use core::panic;
use std::{fmt::Debug, io::{Read, Seek,Write}, ops::Range};

use crate::{encoder::{EncInst, VCDEncoder, WindowHeader}, reader::{Header, WinIndicator}, translator::{DisCopy, Inst, Sequence, SparseInst, VCDTranslator}, ADD, COPY, RUN};

pub struct VCDMerger<R,W> {
    sequential_patches: Vec<VCDTranslator<R>>,
    cur_o_pos: u64,
    ///This is manually managed to ensure it matches cur_o_pos + cur_inst.len() BEFORE applying instructions
    crsr_o_pos: u64,
    cur_win: Vec<MergedInst>,
    win_sum: Option<WindowHeader>,
    sink: VCDEncoder<W>,
    max_u_size: usize,
    cur_inst:Option<ResolveCopy>,
    ///store copy inst. buf[0] is from patch[patches.len()-1]
    copy_buffer:Vec<InitialCopyInst>,
}

impl<R: Read + Seek, W:Write> VCDMerger<R,W> {
    pub fn new(sequential_patches: Vec<VCDTranslator<R>>,sink:W,max_u_size:Option<usize>,) -> std::io::Result<Self> {
        let header = Header::default();
        Ok(Self {
            copy_buffer:Vec::with_capacity(sequential_patches.len()),
            sequential_patches,
            cur_o_pos: 0,
            crsr_o_pos: 0,
            cur_win: Vec::new(),
            sink: VCDEncoder::new(sink,header)?,
            max_u_size:max_u_size.unwrap_or(1<<28), //256MB
            win_sum: None,
            cur_inst:None,
        })
    }
    fn incr_crsr(&mut self,amt:u64){
        dbg!(amt,self.crsr_o_pos);
        self.crsr_o_pos += amt;
    }
    fn incr_o_pos(&mut self,amt:u64){
        self.cur_o_pos += amt;
    }
    pub fn merge_patches(mut self) -> std::io::Result<W> {
        // Main loop: Iterate through positions until all patches are applied
        while let Some(instruction) = self.find_cntl_inst_for_position()? {
            dbg!(&instruction,self.crsr_o_pos,&self.cur_inst);
            self.merge_inner(instruction)?;
        }
        if self.cur_inst.is_some(){
            self.apply_cur_inst()?;
        }

        // Final flush to write out any remaining instructions in the current window
        self.flush_window()?;
        Ok(self.sink.finish()?)
    }
    fn merge_inner(&mut self,instruction: Result<ControlInst,CopyRootResolve>)->std::io::Result<()>{
        match (instruction,&self.cur_inst){
            (Ok(inst), None) => {
                //we are at the top level loop, inst is root (Ok(_))
                self.crsr_o_pos = self.cur_o_pos + inst.inst.len() as u64;
                self.apply_instruction(inst)?;
            },
            (Ok(inst), Some(_)) => {
                //we are in a copy resolve, but found a direct add/run
                self.end_copy_resolution(inst)?;
            },
            (Err(copy), None) => {
                //top level loop and found a copy
                let len = copy.inst.len();
                self.set_cur_inst(copy);
                if len == 1 {
                    self.apply_cur_inst()?;
                }
            },
            (Err(copy), Some(_)) => {
                //in copy resolve, and encountered a copy
                //it might be the same copy from last loop, or a copy from a sequence from last loop
                //it doesn't matter, we need to decide what to do with our state machine.
                //this should either: call end_copy, OR inc o_pos and stay in resolve mode
                self.continue_copy_resolution(copy)?;
            },
        }
        // Write out and reset the current window if necessary
        if self.current_window_size() >= self.max_u_size {
            self.flush_window()?;
        }
        assert!(
            self.cur_inst.is_none() && self.crsr_o_pos == self.cur_o_pos
            ||
            self.cur_inst.is_some() && self.crsr_o_pos != self.cur_o_pos,
            "Invalid state: cur_inst: {:?}, crsr_o_pos: {}, cur_o_pos: {}",
            self.cur_inst, self.crsr_o_pos, self.cur_o_pos
        );
        Ok(())
    }
    fn continue_copy_resolution(&mut self,next_inst:CopyRootResolve)->std::io::Result<()>{
        //we are in a copy resolve, and encountered a copy-like control inst
        //we call start copy resolution to get the root inst for this inst
        //we then interrogate our next position in our cur_inst
        dbg!(&self.cur_inst);
        let cur_inst = self.cur_inst.as_mut().expect("cur_inst should be set here");
        let cur_cmp_inst = cur_inst.inst.compare_form(cur_inst.cur_check_len).expect("cur_inst should not be used up yet");
        //if these don't yield exactly identical inst, then we end the copy resolution
        let next_cmp_inst = next_inst.inst.compare_form(0).expect("inst should not be zero length here");
        //dbg!("cont_res",cur_cmp_inst,next_cmp_inst);
        if next_cmp_inst != cur_cmp_inst{
            self.apply_cur_inst()?;
            return self.merge_inner(Err(next_inst));
        }else{
            //if they are the same, we increment our cur_inst and crsr_o_pos
            //we discard next_inst since it is the same as our cur_inst for this byte.
            cur_inst.cur_check_len += 1;
            let inst_done = cur_inst.inst.len() as u32 == cur_inst.cur_check_len;
            self.incr_crsr(1);
            if inst_done{
                //if we are at the end of our cur_inst, we apply it
                self.apply_cur_inst()?;
            }

        }
        Ok(())
    }
    fn start_copy_resolution(&mut self)->CopyRootResolve{
        //start at the latest patch and find our first root inst (in seq)
        //if no root found (all Copy), then we put the copy from the latest patch in cur_inst
        //inc crsr +1 and let the main loop continue
        let mut latest_copy = None;
        self.copy_buffer.reverse();
        //now in same order as the patches, last is latest.
        //dbg!(&self.copy_buffer);
        while let Some(last_patch_inst) = self.copy_buffer.pop(){
            let InitialCopyInst{ inst, patch_index,..} = last_patch_inst;
            //first we interrogate position 0
            //if it is not a copy, we are done
            //if it is store it in cur_inst
            let inter_value = inst.interrogate(0).expect("inst should not be zero length here");
            //dbg!(&inst,inter_value);
            let value = CopyRootResolve{inst:inter_value,patch_index};
            match inter_value{
                RootInst::Add(_) | RootInst::Run(_) => {
                    dbg!("start_copy_resolution->Add/Run",&latest_copy);
                    return value; //early return, we are done here
                },
                _ => {
                    if latest_copy.is_none(){
                        latest_copy = Some(value);//we want the latest inst
                    }
                }
            }
        }
        //if we are here, then no add/run controls this byte in any patch
        //cur_inst will store the Copy from the last patch
        //dbg!("start_copy_resolution->Copy",&latest_copy);
        return latest_copy.expect("Should have a copy inst here");

    }
    fn set_cur_inst(&mut self,inst:CopyRootResolve){
        let CopyRootResolve { inst, patch_index } = inst;
        self.cur_inst = Some(ResolveCopy{cur_check_len:1,inst,patch_index});
        self.incr_crsr(1);
    }
    fn apply_cur_inst(&mut self)->std::io::Result<()>{
        //take our cur_inst, truncate it per the o_start of this root inst
        let ResolveCopy{mut inst, cur_check_len, patch_index} = self.cur_inst.take().expect("cur_inst should be set here");
        let trunc = inst.len() as u32 - cur_check_len;
        inst.trunc(trunc as u32);   // Truncate per the position of the root inst
        //self.incr_crsr(1);
        debug_assert_eq!(self.crsr_o_pos, self.cur_o_pos + inst.len() as u64, "crsr_o_pos: {} cur_o_pos: {}", self.crsr_o_pos, self.cur_o_pos);
        self.apply_instruction(ControlInst{inst, patch_index,output_start_pos:self.cur_o_pos})
    }
    fn end_copy_resolution(&mut self,instruction: ControlInst)->std::io::Result<()>{
        //make crsr == to cur_o_pos + cur_inst_len
        //THEN apply the cur_inst (to ensure flushing works crsr == o_pos)
        //make crsr == to root_inst + cur_inst_len
        //THEN apply the root inst
        dbg!("end_copy_resolution");
        self.apply_cur_inst()?;
        self.incr_crsr(instruction.inst.len() as u64);

        debug_assert_eq!(self.crsr_o_pos, self.cur_o_pos + instruction.inst.len() as u64, "crsr_o_pos: {} cur_o_pos: {}", self.crsr_o_pos, self.cur_o_pos);
        self.apply_instruction(instruction)
    }
    ///This is the 'terminal' command for moving the merger forward.
    fn apply_instruction(&mut self, instruction: ControlInst) ->std::io::Result<()>{
        //this needs to several things:
        //see if our sss/ssp values need to be changed,
        // if so, will the change make us exceed our MaxWinSize?
        //   if so, we need to flush the window
        //   else we just update them
        let mut cur_inst = Some(instruction);
        while let Some(ci) = cur_inst.take(){
            let (cur_s,_) = self.cur_win_sizes();
            let inst_s = ci.inst.src_range();
            let remaining_size = self.max_u_size as u64 - self.current_u_size() as u64;
            match (cur_s,inst_s) {
                (Some((sss,ssp)), Some(r)) => {
                    if let Some(disjoint) = get_disjoint_range(ssp..ssp+sss, r){
                        let disjoin_len = disjoint.end - disjoint.start;
                        //if this is larger than our remaining window, we need to flush, splitting won't help
                        if disjoin_len > remaining_size{
                            self.flush_window()?;
                            //we have invalidated our variables so loop again
                            cur_inst = Some(ci);
                            continue;
                        }
                    }//else we overlap partially, so splitting would help
                },
                //splitting will help the rest of the cases.
                _ => (),
            }
            //if we are here we can naively check if we need to split the inst
            let trunc = ci.inst.max_u_trunc_amt(remaining_size as u32);
            if trunc > 0{
                let (first,second) = Self::split_inst(ci,trunc);
                self.add_to_window(first);
                cur_inst = Some(second);
            }else{//add it as-is
                self.add_to_window(ci);
                //this is our terminal case, out loop will exit.
            }
        }
        //only if o_pos == crsr_pos do we check again if we should flush the window.
        //we give it an extra 5 bytes so we don't truncate down to a inst len of 1;
        if self.cur_o_pos == self.crsr_o_pos && self.current_window_size() + 5 >= self.max_u_size as usize{
            self.flush_window()?;
        }
        Ok(())

    }
    fn add_to_window(&mut self,inst:ControlInst){
        let ControlInst { output_start_pos, inst, patch_index } = inst;
        debug_assert!(output_start_pos == self.cur_o_pos, "inst output start pos: {} cur_o_pos: {}", output_start_pos, self.cur_o_pos);
        //adjust our current window
        let (src_range,trgt_win_size) = self.new_win_sizes(&inst);
        //dbg!("add_to_window",inst,output_start_pos);
        let ws = self.win_sum.get_or_insert(Default::default());
        if let Some((ssp,sss)) = src_range {
            ws.source_segment_position = Some(ssp);
            ws.source_segment_size = Some(sss);
            if ws.win_indicator != WinIndicator::VCD_SOURCE {
                ws.win_indicator = WinIndicator::VCD_SOURCE;
            }
        }
        ws.size_of_the_target_window = trgt_win_size;
        self.cur_o_pos += inst.len() as u64;

        self.cur_win.push(MergedInst{inst,patch_index});
    }
    fn split_inst(inst:ControlInst, trunc_amt:u32)->(ControlInst,ControlInst){
        let skip = inst.inst.len() - trunc_amt;
        let mut first = inst.clone();
        first.inst.trunc(trunc_amt);
        let mut second = inst;
        second.inst.skip(skip);
        second.output_start_pos += skip as u64;
        (first,second)
    }
    ///(ssp,sss, size_of_the_target_window (T))
    fn new_win_sizes(&self,inst:&RootInst)->(Option<(u64,u64)>,u64){
        let src_range = inst.src_range();
        let ws = self.win_sum.clone().unwrap_or(Default::default());
        let ss = if let Some(r) = src_range{
            let ssp = ws.source_segment_position.unwrap_or(r.start);
            let sss = ws.source_segment_size.unwrap_or(r.end - r.start);
            let new_r = get_superset(r, ssp..ssp+sss);
            //first figure out the change in size of our window
            let new_sss = new_r.end - new_r.start;
            Some((new_r.start,new_sss))
        }else{
            None
        };
        (ss,ws.size_of_the_target_window+inst.len() as u64)
    }
    fn cur_win_sizes(&self)->(Option<(u64,u64)>,u64){
        let ws = self.win_sum.clone().unwrap_or(Default::default());
        let ss = ws.source_segment_position.map(|start| (start,ws.source_segment_size.unwrap()));
        (ss,ws.size_of_the_target_window)
    }

    fn flush_window(&mut self) -> std::io::Result<()> {
        // Write out the current window
        let ws = self.win_sum.take().expect("win_sum should be set here");
        let output_ssp = ws.source_segment_position.clone();
        dbg!(&ws);
        self.sink.start_new_win(ws)?;
        for MergedInst { inst, patch_index } in self.cur_win.drain(..) {
            match inst {
                RootInst::Add(ADD { len, p_pos }) => {
                    let mut bytes = vec![0; len as usize];
                    let r = self.sequential_patches[patch_index].get_reader(p_pos)?;
                    r.read_exact(&mut bytes)?;
                    self.sink.next_inst(EncInst::ADD(bytes)).unwrap();
                },
                RootInst::Run(r) => {
                    self.sink.next_inst(EncInst::RUN(r)).unwrap();
                },
                RootInst::Copy(DisCopy { copy:COPY { len, u_pos }, ssp,sss }) => {
                    //we need to translate this so our u_pos is correct
                    let output_ssp = output_ssp.expect("output_ssp should be set here");
                    //our ssp is within (or coincident) to the bounds of the source segment
                    let zero_u_pos = u_pos as u64 + ssp;
                    let copy_inst = if output_ssp > ssp{
                        let neg_shift = output_ssp - ssp;
                        COPY { len, u_pos: u_pos - neg_shift as u32 }
                    }else{
                        let pos_shift = ssp - output_ssp;
                        COPY { len, u_pos: u_pos + pos_shift as u32 }
                    };
                    // let shift = zero_u_pos - output_ssp; //lower bound should be 0;
                    // let copy_inst = COPY { len, u_pos: u_pos - shift as u32 };
                    dbg!(copy_inst,zero_u_pos,output_ssp,ssp,sss);
                    self.sink.next_inst(EncInst::COPY(copy_inst)).unwrap();
                },
            }
        }
        Ok(())
    }

    /// Finds the effective instruction for a given position in the output stream, with precedence rules.
    fn find_cntl_inst_for_position(&mut self) -> std::io::Result<Option<Result<ControlInst,CopyRootResolve>>> {
        self.copy_buffer.clear();
        let o_position = self.crsr_o_pos;
        dbg!(self.crsr_o_pos);

        // Iterate through the patches in reverse order, as the most recent patch has the highest precedence.
        for (i,translator) in self.sequential_patches.iter_mut().enumerate().rev() {
            if let Some(SparseInst { o_start, mut inst }) = translator.interrogate(o_position)? {
                //dbg!(o_position,o_start,&inst,i);
                inst.skip((o_position - o_start) as u32);
                if inst.is_copy(){
                    self.copy_buffer.push(InitialCopyInst {inst:inst.to_copy(), patch_index: i});
                }else{
                    self.copy_buffer.clear(); //we use empty buffer as signal elsewhere.
                    return Ok(Some(Ok(ControlInst{inst:inst.to_root(), patch_index: i, output_start_pos:o_position})))
                }
            }
        }
        // If no ADD or RUN instruction was found that affects the position we need to return copy.
        // If no copy, we must be End of all the patches.
        if self.copy_buffer.is_empty(){
            Ok(None)
        }else{
            Ok(Some(Err(self.start_copy_resolution()))) //signal that we have copy inst
        }
    }
    fn current_window_size(&self) -> usize {
        self.win_sum.as_ref().map(|h| h.size_of_the_target_window as usize).unwrap_or(0)
    }
    fn current_u_size(&self) -> usize {
        self.win_sum.as_ref().map(|h| h.size_of_the_target_window as usize + h.source_segment_size.unwrap_or(0) as usize).unwrap_or(0)
    }


}


//This needs to be adjusted when we flush to the output stream.

#[derive(Debug,Clone)]
struct ResolveCopy{
    inst: RootInst,
    //CopyInst is either a Copy from the first patch, or Seq with root inst, in a later patch
    patch_index:usize,
    //as we move through the resolution process, we keep track of the 'current len' of this inst
    //this is always at least one, else if it were 0, we would have checked for a root inst.
    //we use this to interrogate the CopyInst for the controlling instruction.
    cur_check_len: u32,
}

impl Sequence{
    //returns a non-sequence instruction with the proper skip applied already.
    fn interrogate(&self,seq_offset:usize)->Option<RootInst>{
        let mut offset = 0;
        let mut seq_pos = 0;
        loop{
            for inst in self.inst.iter(){
                let len = inst.len();
                let inst_seq_start = seq_pos;
                let seq_end_pos = seq_pos + len;
                seq_pos += len; //our seq_pos always moves the whole len of the inst
                if seq_end_pos <= self.skip{
                    continue; //we are before the skip
                }else if offset >= self.len(){
                    return None;
                }
                //if we are here we are in the right range
                let skip_amt = if inst_seq_start < self.skip && seq_end_pos > self.skip{
                    self.skip -inst_seq_start
                }else{//we are adjusting for the skip here
                    0
                };

                let adj_len = len - skip_amt;
                let offset_start = offset;
                let offset_end = offset + adj_len;

                //dbg!(&self,skip_amt,adj_len,inst_seq_start,seq_offset,offset_start,offset_end);
                let inst_cntl_range = offset_start..offset_end;
                if inst_cntl_range.contains(&(seq_offset as u32)){
                    let output = inst.clone();
                    match output{
                        Inst::Sequence(seq) => {
                            //recursively resolve
                            return seq.interrogate(skip_amt as usize);
                        },
                        mut a => {
                            a.skip(skip_amt);
                            return Some(a.to_root())
                        },
                    }
                }
                offset += adj_len;
            }
        }
    }
}


#[derive(Clone, Debug, PartialEq, Eq)]
enum CopyInst{
    Copy(DisCopy),
    Seq(Sequence)
}
impl CopyInst{
    fn len(&self)->u32{
        match self{
            CopyInst::Copy(DisCopy{copy,..}) => copy.len_in_u(), //in this context this is len_in_t actually.
            CopyInst::Seq(seq) => seq.len(),
        }
    }
    fn interrogate(&self,offset:usize)->Option<RootInst>{
        if self.len() <= offset as u32{
            return None;
        }
        match self{
            CopyInst::Copy(DisCopy{copy,sss,ssp}) => {
                //here the offset is literally the skip amount
                let mut copy = copy.clone();
                copy.skip(offset as u32);
                Some(RootInst::Copy(DisCopy{copy, sss:*sss, ssp:*ssp}))
            },
            CopyInst::Seq(seq) => seq.interrogate(offset),
        }
    }

}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct CopyRootResolve{
    inst: RootInst,
    patch_index:usize,
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RootInst{
    Add(ADD),
    Run(RUN),
    Copy(DisCopy),
}

impl RootInst {
    fn src_range(&self)->Option<Range<u64>>{
        match self{
            RootInst::Copy(copy) => Some(copy.min_src()),
            _ => None,
        }
    }
    fn len(&self)->u32{
        match self{
            RootInst::Add(add) => add.len(),
            RootInst::Run(run) => run.len(),
            RootInst::Copy(copy) => copy.copy.len_in_u(),
        }
    }
    fn compare_form(&self,offset:u32)->Option<Self>{
        if self.len() <= offset{
            return None;
        }
        let mut ret = self.clone();
        match &mut ret{
            RootInst::Copy(copy) => {
                copy.copy.skip(offset);
                let Range{start,end} = copy.min_src();
                copy.ssp = start;
                copy.sss = end-start;
            },
            RootInst::Add(add) => add.skip(offset),
            RootInst::Run(run) => run.skip(offset),

        }
        Some(ret)
    }
    fn skip(&mut self,skip:u32){
        match self{
            RootInst::Copy(copy) => copy.copy.skip(skip),
            RootInst::Add(add) => add.skip(skip),
            RootInst::Run(run) => run.skip(skip),
        }
    }
    fn trunc(&mut self,amt:u32){
        match self{
            RootInst::Copy(copy) => copy.copy.trunc(amt),
            RootInst::Add(add) => add.trunc(amt),
            RootInst::Run(run) => run.trunc(amt),
        }

    }
    fn max_u_trunc_amt(&self,max_space_avail:u32)->u32{
        match self{
            RootInst::Copy(copy) => copy.max_u_trunc_amt(max_space_avail),
            RootInst::Add(a) => if a.len() <= max_space_avail {0} else {a.len() - max_space_avail},
            RootInst::Run(a) => if a.len() <= max_space_avail {0} else {a.len() - max_space_avail},

        }
    }
}
impl DisCopy {
    fn min_src(&self)->Range<u64>{
        let Self{copy:COPY { len, u_pos },sss,ssp} = self;
        assert!(u_pos + len <= (ssp+sss) as u32); //make sure we are within the bounds of S in U
        let new_ssp = *ssp + *u_pos as u64;
        let new_end = new_ssp + *len as u64;
        new_ssp..new_end
    }
    fn max_u_trunc_amt(&self,max_space_avail:u32)->u32{
        //can we figure out how much to truncate to fit in the space?
        //every change in len, also shrinks the sss
        let min = self.min_src();
        let cur_s_len = min.end - min.start;
        let cur_u_len = self.copy.len + cur_s_len as u32;
        if cur_u_len <= max_space_avail {
            return 0;
        }else{
            (cur_u_len - max_space_avail) / 2
        }
    }
}
impl Inst {
    fn is_copy(&self) -> bool {
        matches!(self, Inst::Copy{..} | Inst::Sequence(_) )
    }
    fn to_copy(self) ->CopyInst{
        match self{
            Inst::Copy(c) => CopyInst::Copy(c),
            Inst::Sequence(seq) => CopyInst::Seq(seq),
            _ => panic!("Expected Copy or Sequence, got {:?}", self),
        }
    }
    fn to_root(self) -> RootInst{
        match self{
            Inst::Add(add) => RootInst::Add(add),
            Inst::Run(run) => RootInst::Run(run),
            Inst::Copy(c) => RootInst::Copy(c),
            _ => panic!("Expected Add, Run, or Copy, got {:?}", self),
        }
    }
    fn skip(&mut self, amt: u32) {
        if amt == 0 {return}
        match self {

            Self::Sequence(inst) => {inst.skip(amt);},
            Self::Add(inst) => {inst.skip(amt);},
            Self::Run(inst) => {inst.skip(amt);},
            Self::Copy (copy) => {copy.copy.skip(amt);},
        }
    }
}

/// The output from the translator, will never have implicit sequences, they are all made explicit.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ControlInst{
    output_start_pos: u64,
    inst: RootInst,
    patch_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InitialCopyInst{
    inst: CopyInst,
    patch_index: usize,
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct MergedInst{
    inst: RootInst,
    patch_index: usize,
}

fn get_superset<T: Ord>(range1: Range<T>, range2: Range<T>) -> Range<T> {
    // Find the minimum start point
    let start = std::cmp::min(range1.start, range2.start);

    // Find the maximum end point
    let end = std::cmp::max(range1.end, range2.end);

    // Return the superset range
    start..end
}

///Returns the inner disjoint range, if any, between two ranges.
fn get_disjoint_range<T: Ord + Debug>(range1: Range<T>, range2: Range<T>) -> Option<Range<T>> {
    use std::cmp::Ordering;
    match (range1.start.cmp(&range2.start), range1.end.cmp(&range2.end)) {
        // range1 completely before range2
        (Ordering::Less, Ordering::Less) => {
            if range1.end < range2.start {
                Some(range1.end..range2.start)
            } else {
                None
            }
        }
        // range2 completely before range1
        (Ordering::Greater, Ordering::Greater) => {
            if range2.end < range1.start {
                Some(range2.end..range1.start)
            } else {
                None
            }
        }
        // Overlapping ranges
        _ => None,
    }
}

#[cfg(test)]
mod test_super {
    use crate::{applicator::apply_patch, decoder::VCDDecoder, reader::{DeltaIndicator, VCDReader}};

    use super::*;
    /*
    All the merger tests will start with a src file of 'hello'
    We will then create a series of patches that will make certain *changes* to the file.
    That is, we want to be able to apply them in different orders for different effects.
    To this end, all of the target windows must be the same size.
    We will pick 10 bytes as our target window size. This is twice the length of 'hello'

    We need to test the following:
    Sequence unrolling
    Copy Passthrough
    Add/Run precedence

    For the seq:
    We will simply Copy the first byte and then last four bytes and sequence them to len 10.
    This should turn '01234' into '0123412341'

    For the copy:
    We will make a patch that will copy the first five bytes to the last five bytes.
    This should turn '01234' into '0123401234'

    For the add/run:
    We will make a patch that will insert 'A' (ADD) at first pos Copy next 2, Then 'XXX'(Run) + 'YZ'(Add) The COpy rem
    This should turn '01234' into 'A12XXXYZ34'

    Then we do a patch with multiple transforms internally
    Complex:
    We will Add 'Y' Run(2) 'Z' CopySeq last in S u_pos 4 len 6 (len in t=3) Copy u_pos 1 len 4
    This should turn '01234' into 'YZZ4YZ1234'

    We can then mix and match these patches and we should be able to reason about the outputs.
    */
    const HDR:Header = Header { hdr_indicator: 0, secondary_compressor_id: None, code_table_data: None };
    const WIN_HDR:WindowHeader = WindowHeader {
        win_indicator: WinIndicator::VCD_SOURCE,
        source_segment_size: Some(5),
        source_segment_position: Some(0),
        size_of_the_target_window: 10,
        delta_indicator: DeltaIndicator(0),
    };
    use std::io::Cursor;
    fn make_translator(patch_bytes:Vec<u8>)->VCDTranslator<Cursor<Vec<u8>>>{
        VCDTranslator::new(VCDDecoder::new(VCDReader::new(Cursor::new(patch_bytes)).unwrap())).unwrap()
    }
    fn seq_patch() -> VCDTranslator<Cursor<Vec<u8>>> {
        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WIN_HDR).unwrap();
        // Instructions
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 0 })).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 14, u_pos: 1 })).unwrap();
        // Force encoding
        let w = encoder.finish().unwrap().into_inner();
        make_translator(w)
    }
    fn copy_patch() -> VCDTranslator<Cursor<Vec<u8>>> {
        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WIN_HDR).unwrap();
        // Instructions
        encoder.next_inst(EncInst::COPY(COPY { len: 5, u_pos: 0 })).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 5, u_pos: 0 })).unwrap();
        // Force encoding
        let w = encoder.finish().unwrap().into_inner();
        make_translator(w)
    }
    fn add_run_patch() -> VCDTranslator<Cursor<Vec<u8>>> {
        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WIN_HDR).unwrap();
        // Instructions
        encoder.next_inst(EncInst::ADD("A".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 2, u_pos: 1 })).unwrap();
        encoder.next_inst(EncInst::RUN(RUN { len: 3, byte: b'X' })).unwrap();
        encoder.next_inst(EncInst::ADD("YZ".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 2, u_pos: 3 })).unwrap();

        // Force encoding
        let w = encoder.finish().unwrap().into_inner();
        make_translator(w)
    }
    fn complex_patch()->VCDTranslator<Cursor<Vec<u8>>>{
        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WIN_HDR).unwrap();
        // Instructions
        encoder.next_inst(EncInst::ADD("Y".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::RUN(RUN { len: 2, byte: b'Z' })).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 3, u_pos: 4 })).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 4, u_pos: 1 })).unwrap();
        // Force encoding
        let w = encoder.finish().unwrap().into_inner();
        make_translator(w)
    }
    const SRC:&[u8] = b"01234";
    #[test]
    fn test_copy_seq(){
        //01234 Copy-> 0123401234 Seq-> 0123412341
        let answer = b"0123412341";
        let copy = copy_patch();
        let seq = seq_patch();
        let merger = VCDMerger::new(vec![copy,seq],Vec::new(),None).unwrap();
        let merged_patch = merger.merge_patches().unwrap();
        let mut cursor = Cursor::new(merged_patch);
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(Cursor::new(SRC.to_vec())), &mut output).unwrap();
        //print output as a string
        let as_str = std::str::from_utf8(&output).unwrap();
        println!("{}",as_str);
        assert_eq!(output,answer);
    }
    #[test]
    fn test_seq_copy(){
        //01234 Seq-> 0123412341 Seq-> 0123401234
        let answer = b"0123401234";
        let copy = copy_patch();
        let seq = seq_patch();
        let merger = VCDMerger::new(vec![seq,copy],Vec::new(),None).unwrap();
        let merged_patch = merger.merge_patches().unwrap();
        let mut cursor = Cursor::new(merged_patch);
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(Cursor::new(SRC.to_vec())), &mut output).unwrap();
        //print output as a string
        let as_str = std::str::from_utf8(&output).unwrap();
        println!("{}",as_str);
        assert_eq!(output,answer);
    }
    #[test]
    fn test_seq_copy_add(){
        //01234 Seq->Copy 0123412341 Add-> A12XXXYZ34
        let answer = b"A12XXXYZ34";
        let seq = seq_patch();
        let copy = copy_patch();
        let add_run = add_run_patch();
        let merger = VCDMerger::new(vec![seq,copy,add_run],Vec::new(),None).unwrap();
        let merged_patch = merger.merge_patches().unwrap();
        let mut cursor = Cursor::new(merged_patch);
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(Cursor::new(SRC.to_vec())), &mut output).unwrap();
        //print output as a string
        let as_str = std::str::from_utf8(&output).unwrap();
        println!("{}",as_str);
        assert_eq!(output,answer);
    }
    #[test]
    fn test_copy_seq_add(){
        //01234 Copy->Seq 0123412341 Add-> A12XXXYZ34
        let answer = b"A12XXXYZ34";
        let seq = seq_patch();
        let copy = copy_patch();
        let add_run = add_run_patch();
        let merger = VCDMerger::new(vec![copy,seq,add_run],Vec::new(),None).unwrap();
        let merged_patch = merger.merge_patches().unwrap();
        let mut cursor = Cursor::new(merged_patch);
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(Cursor::new(SRC.to_vec())), &mut output).unwrap();
        //print output as a string
        let as_str = std::str::from_utf8(&output).unwrap();
        println!("{}",as_str);
        assert_eq!(output,answer);
    }
    #[test]
    fn test_add_complex(){
        //01234 Add-> A12XXXYZ34 Compl YZZXYZ12XX
        let answer = b"YZZXYZ12XX";
        let add_run = add_run_patch();
        let comp = complex_patch();
        let merger = VCDMerger::new(vec![add_run,comp],Vec::new(),None).unwrap();
        let merged_patch = merger.merge_patches().unwrap();
        let mut cursor = Cursor::new(merged_patch);
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(Cursor::new(SRC.to_vec())), &mut output).unwrap();
        //print output as a string
        let as_str = std::str::from_utf8(&output).unwrap();
        println!("{}",as_str);
        assert_eq!(output,answer);
    }
    #[test]
    fn test_disjoint_ranges() {
        let range1 = 1..5;
        let range2 = 8..12;
        let expected_disjoint = Some(5..8);

        let result = get_disjoint_range(range1, range2);
        assert_eq!(result, expected_disjoint);
    }

    #[test]
    fn test_overlapping_ranges() {
        let range1 = 3..9;
        let range2 = 7..12;

        let result = get_disjoint_range(range1, range2);
        assert_eq!(result, None);
    }

    #[test]
    fn test_adjacent_ranges() {
        let range1 = 1..5;
        let range2 = 5..9;

        let result = get_disjoint_range(range1, range2);
        assert_eq!(result, None);
    }

    #[test]
    fn test_equal_ranges() {
        let range1 = 2..6;
        let range2 = 2..6;

        let result = get_disjoint_range(range1, range2);
        assert_eq!(result, None);
    }
    #[test]
    fn test_get_superset() {
        // Test Case 1: range1 supersedes range2
        let range1 = 10..20;
        let range2 = 12..18;
        let expected_superset = 10..20;
        let result = get_superset(range1, range2);
        assert_eq!(result, expected_superset);

        // Test Case 2: range2 supersedes range1
        let range1 = 0..2;
        let range2 = 7..10;
        let expected_superset = 0..10;
        let result = get_superset(range1, range2);
        assert_eq!(result, expected_superset);

        // Test Case 3: Overlapping ranges
        let range1 = 0..15;
        let range2 = 10..20;
        let expected_superset = 0..20;
        let result = get_superset(range1, range2);
        assert_eq!(result, expected_superset);

        // Test Case 4: start > end within a range (should never happen)
        // Option 1: Force the issue if your logic allows it
        // Option 2: Add a check in the `get_superset` function and expect
        //         a panic or a Result type return
    }
}