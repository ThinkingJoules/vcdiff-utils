use core::panic;
use std::{fmt::Debug, io::{Read, Seek,Write}, ops::Range};

use crate::{encoder::{EncInst, VCDEncoder, WindowHeader}, reader::{Header, WinIndicator}, translator::{DisCopy, Inst, Sequence, SparseInst, VCDTranslator}, ADD, COPY, RUN};

struct MergeState<R,W>{
    cur_o_pos: u64,
    max_u_size: usize,
    cur_win: Vec<MergedInst>,
    win_sum: Option<WindowHeader>,
    sequential_patches: Vec<VCDTranslator<R>>,
    sink: VCDEncoder<W>,
}

impl<R:Read+Seek,W:Write> MergeState<R,W> {
    ///This is the 'terminal' command for moving the merger forward.
    fn apply_instruction(&mut self, instruction: MergedInst) ->std::io::Result<()>{
        //this needs to do several things:
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
                (Some((ssp,sss)), Some(r)) => {
                    if let Some(disjoint) = get_disjoint_range(ssp..ssp+sss, r.clone()){
                        let disjoint_len = disjoint.end - disjoint.start;
                        //if this is larger than our remaining window, we need to flush, splitting won't help
                        if disjoint_len > remaining_size{
                            // println!("flushing (disjoint), cur_o_pos: {} cur_win_size: {} max_u_size: {}", self.cur_o_pos, self.current_window_size(), self.max_u_size);
                            // println!("sss: {} ssp: {} r: {:?} disjoin_len {:?} remaining {}", sss,ssp,r,disjoint_len,remaining_size);
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
            let trunc = ci.inst.max_u_trunc_amt((self.max_u_size - self.current_u_size()) as u32);
            debug_assert!(trunc < ci.inst.len() as u32, "trunc: {} len: {}",trunc,ci.inst.len());
            if trunc > 0{
                let (first,second) = Self::split_inst(ci,trunc);
                self.add_to_window(first);
                cur_inst = Some(second);
            }else{//add it as-is
                self.add_to_window(ci);
                //this is our terminal case, our loop will exit.
            }
        }
        //only if o_pos == crsr_pos do we check again if we should flush the window.
        //we give it an extra 5 bytes so we don't truncate down to a inst len of 1;
        if self.current_window_size() + 5 >= self.max_u_size as usize{
            // println!("flushing (normal), cur_o_pos: {} cur_win_size: {} max_u_size: {}", self.cur_o_pos, self.current_window_size(), self.max_u_size);
            self.flush_window()?;
        }
        Ok(())

    }
    fn add_to_window(&mut self,inst:MergedInst){
        let MergedInst {  inst, patch_index } = inst;
        //debug_assert!(output_start_pos == self.cur_o_pos, "inst output start pos: {} cur_o_pos: {}", output_start_pos, self.cur_o_pos);
        //adjust our current window
        let (src_range,trgt_win_size) = self.new_win_sizes(&inst);
        let ws = self.win_sum.get_or_insert(Default::default());
        if let Some((ssp,sss)) = src_range {
            ws.source_segment_position = Some(ssp);
            ws.source_segment_size = Some(sss);
            if ws.win_indicator != WinIndicator::VCD_SOURCE {
                ws.win_indicator = WinIndicator::VCD_SOURCE;
            }
        }
        ws.size_of_the_target_window = trgt_win_size;
        if self.cur_o_pos == 166972041{
            println!("cur_o_pos: {} inst: {:?}",self.cur_o_pos,inst);
        }
        self.cur_o_pos += inst.len() as u64;

        self.cur_win.push(MergedInst{inst,patch_index});
    }
    fn split_inst(inst:MergedInst, trunc_amt:u32)->(MergedInst,MergedInst){
        let skip = inst.inst.len() - trunc_amt;
        let mut first = inst.clone();
        first.inst.trunc(trunc_amt);
        let mut second = inst;
        second.inst.skip(skip);
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
                RootInst::Copy(DisCopy { copy:COPY { len, u_pos }, ssp,.. }) => {
                    //we need to translate this so our u_pos is correct
                    let output_ssp = output_ssp.expect("output_ssp should be set here");
                    //our ssp is within (or coincident) to the bounds of the source segment
                    let copy_inst = if output_ssp > ssp{
                        let neg_shift = output_ssp - ssp;
                        COPY { len, u_pos: u_pos - neg_shift as u32 }
                    }else{
                        let pos_shift = ssp - output_ssp;
                        COPY { len, u_pos: u_pos + pos_shift as u32 }
                    };
                    // let shift = zero_u_pos - output_ssp; //lower bound should be 0;
                    // let copy_inst = COPY { len, u_pos: u_pos - shift as u32 };
                    self.sink.next_inst(EncInst::COPY(copy_inst)).unwrap();
                },
            }
        }
        Ok(())
    }
    fn current_window_size(&self) -> usize {
        self.win_sum.as_ref().map(|h| h.size_of_the_target_window as usize).unwrap_or(0)
    }
    fn current_u_size(&self) -> usize {
        self.win_sum.as_ref().map(|h| h.size_of_the_target_window as usize + h.source_segment_size.unwrap_or(0) as usize).unwrap_or(0)
    }
    fn next(&mut self) -> std::io::Result<Option<SparseInst>> {
        self.sequential_patches.last_mut().unwrap().interrogate(self.cur_o_pos)
    }
}

pub fn merge_patches<R: Read + Seek, W:Write>(sequential_patches: Vec<VCDTranslator<R>>,sink:W,max_u_size:Option<usize>,) -> std::io::Result<W> {
    let max_u_size = max_u_size.unwrap_or(1<<28); //256MB
    let header = Header::default();
    let encoder = VCDEncoder::new(sink,header)?;
    let mut state = MergeState{
        cur_o_pos: 0,
        cur_win: Vec::new(),
        sink: encoder,
        sequential_patches,
        max_u_size,
        win_sum: None,
    };
    let cntl_patch = state.sequential_patches.len()-1;
    // we loop through the instructions of the last patch in our main loop
    // when we find a copy, we start a copy resolution that will recursively resolve the Copy back though the patches
    let mut input = [Inst::Run(RUN { len: 0,byte:0 }); 1];
    let mut resolution_buffer = Vec::new();
    while let Some(SparseInst { o_start, inst }) = state.next()? {
        debug_assert!(inst.len() > 0, "{:?}",inst);
        let expected_end = o_start + inst.len() as u64;
        input[0] = inst;
        resolve_list_of_inst(&mut state.sequential_patches,cntl_patch, &input, &mut resolution_buffer)?;
        for inst in resolution_buffer.drain(..){
            debug_assert!(inst.inst.len() > 0, "inst.len() == 0");
            state.apply_instruction(inst)?;
        }
        debug_assert_eq!(state.cur_o_pos, expected_end, "cur_o_pos: {} expected_end: {}", state.cur_o_pos, expected_end);
    }

    state.flush_window()?;
    state.sink.finish()
}
fn resolve_list_of_inst<R: Read + Seek>(
    patches:&mut [VCDTranslator<R>],
    patch_index:usize,
    list:&[Inst],
    output:&mut Vec<MergedInst>
)->std::io::Result<()>{
    //the list might already be resolved (no copy/seq)
    //if it is, we just add it to the output
    //else we resolve the copy/seq and add the resolved inst to the output
    for inst in list{
        let len = inst.len();
        if inst.is_copy(){
            let copy = inst.clone().to_copy();
            match copy{
                CopyInst::Copy(copy) => {
                    if patch_index == 0{ //we are emitting a copy from the first patch
                        let ci = MergedInst { inst: RootInst::Copy(copy), patch_index };
                        output.push(ci);
                    }else{//this copy references some earlier patch output, we must resolve it.
                        let src_o_pos = copy.src_o_pos();
                        let next_patch_index = patch_index - 1;
                        let next_slice = patches[next_patch_index].exact_slice(src_o_pos, len)?;
                        //dbg!(&next_slice);
                        resolve_list_of_inst(patches,next_patch_index,next_slice.as_slice(),output)?;
                    }
                },
                CopyInst::Seq(Sequence { inst, len, skip, trunc }) => {
                    let mut inner_out = Vec::new();
                    resolve_list_of_inst(patches,patch_index,inst.as_slice(),&mut inner_out)?;
                    let effective_len = len - (skip + trunc);
                    flatten_and_trim_sequence(&inner_out, skip, effective_len, output)
                },
            }
        }else{
            let ci = MergedInst { inst: inst.clone().to_root(), patch_index };
            output.push(ci);
        }
    }
    Ok(())
}

fn flatten_and_trim_sequence(seq:&[MergedInst],skip:u32,len:u32,output:&mut Vec<MergedInst>) {
    let mut current_len = 0;
    let mut skipped_bytes = 0;

    // Calculate the effective length after considering truncation
    let effective_len = len;
    while current_len < effective_len {
        for MergedInst { inst:instruction, patch_index } in seq.iter() {
            let mut modified_instruction = instruction.clone();
            let end_pos = current_len + modified_instruction.len();
            if skipped_bytes < skip {
                if skipped_bytes + instruction.len() > skip {
                    // We're in the middle of an instruction, need to apply skip
                    modified_instruction.skip(skip - skipped_bytes);
                    skipped_bytes += modified_instruction.len(); // Update skipped_bytes to reflect the adjusted instruction

                } else {
                    // Entire instruction is skipped
                    skipped_bytes += instruction.len();
                    continue;
                }
            } else {
                // After skip has been accounted for, directly clone and potentially modify for truncation
                // If adding this instruction would exceed the effective length, apply truncation
                if end_pos > effective_len {
                    let trunc_amt = end_pos - effective_len;
                    modified_instruction.trunc(trunc_amt);
                }
            }
            current_len += modified_instruction.len();
            skipped_bytes += modified_instruction.len(); // Update skipped_bytes to reflect the adjusted instruction
            output.push(MergedInst { inst: modified_instruction, patch_index: *patch_index });
            // If we've reached or exceeded the effective length, break out of the loop
            if current_len >= effective_len {
                break;
            }
        }
    }
}



#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct MergedInst{
    inst: RootInst,
    patch_index: usize,
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
        if max_space_avail >= self.len() || max_space_avail == 0 {return 0}
        match self{
            RootInst::Copy(copy) => copy.max_u_trunc_amt(max_space_avail),
            a => a.len() - max_space_avail,

        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CopyInst{
    Copy(DisCopy),
    Seq(Sequence)
}

impl DisCopy {
    fn src_o_pos(&self)->u64{
        self.ssp + self.copy.u_pos as u64
    }
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
    // fn skip(&mut self, amt: u32) {
    //     if amt == 0 {return}
    //     match self {

    //         Self::Sequence(inst) => {inst.skip(amt);},
    //         Self::Add(inst) => {inst.skip(amt);},
    //         Self::Run(inst) => {inst.skip(amt);},
    //         Self::Copy (copy) => {copy.copy.skip(amt);},
    //     }
    // }
    // fn trunc(&mut self, amt: u32) {
    //     if amt == 0 {return}
    //     match self {
    //         Self::Sequence(inst) => {inst.trunc += amt;},
    //         Self::Add(inst) => {inst.trunc(amt);},
    //         Self::Run(inst) => {inst.trunc(amt);},
    //         Self::Copy (copy) => {copy.copy.trunc(amt);},
    //     }
    // }
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
    We will simply Copy the first byte and then 2 bytes and sequence them to len 10.
    This should turn '01234' into '0230230230'

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
        encoder.next_inst(EncInst::COPY(COPY { len: 2, u_pos: 2 })).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 10, u_pos: 5 })).unwrap();
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
        //01234 Copy-> 0123401234 Seq-> 0230230230
        let answer = b"0230230230";
        let copy = copy_patch();
        let seq = seq_patch();
        let merged_patch = merge_patches(vec![copy,seq],Vec::new(),None).unwrap();
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
        //01234 Seq-> 0230230230 Copy-> 0230202302
        let answer = b"0230202302";
        let copy = copy_patch();
        let seq = seq_patch();
        let merged_patch = merge_patches(vec![seq,copy],Vec::new(),None).unwrap();
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
        //01234 Seq->Copy 0230202302 Add-> A23XXXYZ02
        let answer = b"A23XXXYZ02";
        let seq = seq_patch();
        let copy = copy_patch();
        let add_run = add_run_patch();
        let merged_patch = merge_patches(vec![seq,copy,add_run], Vec::new(), None).unwrap();
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
        //01234 Copy->Seq 0230230230 Add-> A23XXXYZ02
        let answer = b"A23XXXYZ02";
        let seq = seq_patch();
        let copy = copy_patch();
        let add_run = add_run_patch();
        let merged_patch = merge_patches(vec![copy,seq,add_run],Vec::new(),None).unwrap();
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
        let merged_patch = merge_patches(vec![add_run,comp],Vec::new(),None).unwrap();
        let mut cursor = Cursor::new(merged_patch);
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(Cursor::new(SRC.to_vec())), &mut output).unwrap();
        //print output as a string
        let as_str = std::str::from_utf8(&output).unwrap();
        println!("{}",as_str);
        assert_eq!(output,answer);
    }
    #[test]
    fn test_all_seq(){
        //01234 Add-> A12XXXYZ34 Compl YZZXYZ12XX -> Copy YZZXYYZZXY -> Seq YZXYZXYZXY
        let answer = b"YZXYZXYZXY";
        let add_run = add_run_patch();
        let comp = complex_patch();
        let copy = copy_patch();
        let seq = seq_patch();
        let merged_patch = merge_patches(vec![add_run,comp,copy,seq],Vec::new(),None).unwrap();
        let mut cursor = Cursor::new(merged_patch);
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(Cursor::new(SRC.to_vec())), &mut output).unwrap();
        //print output as a string
        let as_str = std::str::from_utf8(&output).unwrap();
        println!("{}",as_str);
        assert_eq!(output,answer);
    }
    #[test]
    fn test_kitchen_sink(){
        //"hello" -> "hello world!" -> "Hello! Hello! Hello. hello. hello..."
        //we need to use a series of VCD_TARGET windows and Sequences across multiple patches
        //we should use copy/seq excessively since add/run is simple in the code paths.
        let src = b"hello";
        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(5), source_segment_position: Some(0), size_of_the_target_window:12 , delta_indicator: DeltaIndicator(0) }).unwrap();
        // Instructions
        encoder.next_inst(EncInst::COPY(COPY { len: 5, u_pos: 0 })).unwrap();
        encoder.next_inst(EncInst::ADD(" w".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 9 })).unwrap();
        encoder.next_inst(EncInst::ADD("rld!".as_bytes().to_vec())).unwrap();
        let p1 = encoder.finish().unwrap().into_inner();
        let p1_answer = b"hello world!";
        let mut cursor = Cursor::new(p1.clone());
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(Cursor::new(src.to_vec())), &mut output).unwrap();
        assert_eq!(output,p1_answer); //ensure our instructions do what we think they are.
        let patch_1 = make_translator(p1);
        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(11), source_segment_position: Some(1), size_of_the_target_window:7 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::ADD("H".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 4, u_pos: 0 })).unwrap(); //ello
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 10 })).unwrap(); //'!'
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 4 })).unwrap(); //' '
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(7), source_segment_position: Some(0), size_of_the_target_window:14 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 19, u_pos: 0 })).unwrap(); //Hello! Hello! Hello
        encoder.next_inst(EncInst::ADD(".".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 13 })).unwrap(); // ' '
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(6), source_segment_position: Some(15), size_of_the_target_window:7, delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::ADD("h".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 6, u_pos: 0 })).unwrap(); //'ello. '
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(6), source_segment_position: Some(21), size_of_the_target_window:8 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 6, u_pos: 0 })).unwrap(); //'hello.'
        encoder.next_inst(EncInst::COPY(COPY { len: 3, u_pos: 11 })).unwrap(); //Seq '.' == Run(3) '.'
        let p2 = encoder.finish().unwrap().into_inner();
        let p2_answer = b"Hello! Hello! Hello. hello. hello...";
        let mut cursor = Cursor::new(p2.clone());
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(Cursor::new(p1_answer.to_vec())), &mut output).unwrap();
        assert_eq!(output,p2_answer);
        let patch_2 = make_translator(p2);
        let merged_patch = merge_patches(vec![patch_1,patch_2],Vec::new(),None).unwrap();
        let mut cursor = Cursor::new(merged_patch);
        let mut output = Vec::new();
        let answer = b"Hello! Hello! Hello. hello. hello...";
        apply_patch(&mut cursor, Some(Cursor::new(src.to_vec())), &mut output).unwrap();
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

    }

    #[test]
    fn test_skip_sequence() {
        let inst = MergedInst { inst: RootInst::Add(ADD {len:5, p_pos:0 }), patch_index: 0 };
        let seq = vec![inst];
        let mut output = Vec::new();
        flatten_and_trim_sequence(&seq, 3, 7, &mut output);
        let result = vec![MergedInst { inst: RootInst::Add(ADD {len:2, p_pos:3 }), patch_index: 0 },inst];
        assert_eq!(output, result, "Output should contain two copies of the instruction");
    }

    #[test]
    fn test_truncate_sequence() {
        let seq = vec![MergedInst { inst: RootInst::Add(ADD { len: 10, p_pos:0 }), patch_index: 0 }];
        let mut output = Vec::new();
        flatten_and_trim_sequence(&seq, 0, 5, &mut output);
        let result = vec![MergedInst { inst: RootInst::Add(ADD { len: 5, p_pos:0 }), patch_index: 0 }];
        assert_eq!(output, result, "Output should contain a truncated instruction");
    }

    #[test]
    fn test_skip_and_trim() {
        let seq = vec![
            MergedInst { inst: RootInst::Add(ADD { len: 3, p_pos: 0 }), patch_index: 0 },
            MergedInst { inst: RootInst::Run(RUN {len:4, byte: 0 }), patch_index: 0 },
            MergedInst { inst: RootInst::Copy(DisCopy {copy:COPY{len: 5, u_pos: 0 }, sss: 5, ssp: 0 }), patch_index: 0 },
        ];
        let mut output = Vec::new();
        flatten_and_trim_sequence(&seq, 3, 14, &mut output);
        let result = vec![
            MergedInst { inst: RootInst::Run(RUN {len:4, byte: 0 }), patch_index: 0 },
            MergedInst { inst: RootInst::Copy(DisCopy {copy:COPY{len: 5, u_pos: 0 }, sss: 5, ssp: 0 }), patch_index: 0 },
            MergedInst { inst: RootInst::Add(ADD { len: 3, p_pos: 0 }), patch_index: 0 },
            MergedInst { inst: RootInst::Run(RUN {len:2, byte: 0 }), patch_index: 0 },
        ];
        assert_eq!(output, result, "Output should contain a truncated instruction");
    }
    #[test]
    fn test_trunc_and_trim() {
        let seq = vec![
            MergedInst { inst: RootInst::Add(ADD { len: 3, p_pos: 0 }), patch_index: 0 },
            MergedInst { inst: RootInst::Run(RUN {len:4, byte: 0 }), patch_index: 0 },
            MergedInst { inst: RootInst::Copy(DisCopy {copy:COPY{len: 5, u_pos: 0 }, sss: 5, ssp: 0 }), patch_index: 0 },
        ];
        let mut output = Vec::new();
        flatten_and_trim_sequence(&seq, 3, 12, &mut output);
        let result = vec![
            MergedInst { inst: RootInst::Run(RUN {len:4, byte: 0 }), patch_index: 0 },
            MergedInst { inst: RootInst::Copy(DisCopy {copy:COPY{len: 5, u_pos: 0 }, sss: 5, ssp: 0 }), patch_index: 0 },
            MergedInst { inst: RootInst::Add(ADD { len: 3, p_pos: 0 }), patch_index: 0 },
        ];
        assert_eq!(output, result, "Output should contain a truncated instruction");
    }
}