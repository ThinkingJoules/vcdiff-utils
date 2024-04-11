use std::{fmt::Debug, io::{Read, Seek,Write}, num::NonZeroU32, ops::Range};

use crate::{encoder::{EncInst, VCDEncoder, WindowHeader}, extractor::{extract_patch_instructions, get_exact_slice, sum_len, CopyInst, CopyType, DisCopy, DisInst, ExAdd, ExInstType, InstType, Instruction, VcdExtract}, reader::{Header, WinIndicator}, COPY};

/*
In theory we only ever need to compare two patch files at once.
We have an 'earlier' and a 'later' (terminal) patch.
We need to figure out the precedence rules for merging.

Any Add/Run found in 'later' is preserved.
Copy's get more difficult.
We technically have several types of copy commands.
We have a Copy in S (in superstring U) that is ultimately sourced from the Src Doc
    lets call this CopySS.
Then we have Copy in S and the S component of U is from our Output Doc;
    lets call this CopySO.
Finally we have a Copy that is in T of U.
That is we are copying bytes from some earlier operation(s) applied.
This one has two sub variants.
It can be treated like a normal Copy, or it might be an implicit sequence;
lets call the first CopyTU, and the implicit sequence CopyTS.

Currently we resolve all Copy's to CopySS so we can merge them.
We must unwind any implicit sequences so that we can merge within them as well.
Since we don't have the actual data, we can't encode anything other than CopySS.
This means the merge patch will only ever have win_indicators of VCD_SOURCE (or NEITHER).
*/

pub fn deref_non_copy_ss(extracted:Vec<VcdExtract>)->Vec<VcdExtract>{
    let mut output:Vec<VcdExtract> = Vec::with_capacity(extracted.len());
    let mut cur_o_pos = 0;
    for DisInst { inst, .. } in extracted {
        let (o_start,slice_len,seq_len) = match inst.inst_type(){
            InstType::Copy { copy_type:CopyType::CopyS, vcd_trgt:false } |
            InstType::Run |
            InstType::Add => {
                let o_pos_start = cur_o_pos;
                cur_o_pos += inst.len_in_o() as u64;
                output.push(DisInst { o_pos_start, inst });
                continue;
            },
            InstType::Copy { copy_type:CopyType::CopyQ { len_o }, .. } => {
                let slice_len = inst.len_in_u() - len_o;
                let o_start = cur_o_pos - slice_len as u64;
                (o_start,slice_len,len_o)
            },
            InstType::Copy { copy_type:CopyType::CopyT { inst_u_pos_start }, .. } => {
                let copy = inst.clone().take_copy().unwrap();
                let offset = inst_u_pos_start - copy.u_pos;
                let o_start = cur_o_pos - offset as u64;
                (o_start,copy.len_in_u(),0)
            },
            InstType::Copy { copy_type:CopyType::CopyS, vcd_trgt } => {
                debug_assert!(vcd_trgt, "We should only be resolving Trgt sourced Copys here");
                let copy = inst.clone().take_copy().unwrap();
                let o_start = copy.ssp + copy.u_pos as u64;
                (o_start,copy.len_in_u(),0)
            },

        };

        let resolved = get_exact_slice(output.as_slice(), o_start, slice_len).unwrap();
        if seq_len > 0 {
            expand_sequence(&resolved, seq_len,&mut cur_o_pos, &mut output);
        }else{
            for resolved_inst in resolved {
                let o_pos_start = cur_o_pos;
                cur_o_pos += resolved_inst.inst.len_in_o() as u64;
                output.push(DisInst { o_pos_start, inst: resolved_inst.inst });
            }
        }
    }
    output
}

fn find_copy_s(extract:&[VcdExtract],shift:usize,dest:&mut Vec<usize>){
    for (i,ext) in extract.iter().enumerate(){
        match ext.inst_type(){
            InstType::Copy { copy_type:CopyType::CopyS, vcd_trgt:false } => dest.push(i+shift),
            _ => (),
        }
    }
}


#[derive(Clone, Debug)]
pub struct Merger{
    ///The summary patch that will be written to the output.
    terminal_patch: Vec<VcdExtract>,
    ///If this is empty, merging a patch will have no effect.
    ///These are where TerminalInst::CopySS are found.
    terminal_copy_indices: Vec<usize>,
    final_size: u64,
}

impl Merger {
    pub fn new<R:Read + Seek>(terminal_patch:R) -> std::io::Result<Result<Merger,SummaryPatch>> {
        let (terminal_patch,stats) = extract_patch_instructions(terminal_patch)?;
        if stats.copy_bytes == 0{
            return Ok(Err(SummaryPatch(terminal_patch,stats.output_size as u64)));
        }
        let mut terminal_copy_indices = Vec::new();
        //we for sure need to translate local. I think translate global isn't needed??
        //will need to check this.
        let terminal_patch = deref_non_copy_ss(terminal_patch);
        find_copy_s(&terminal_patch,0,&mut terminal_copy_indices);
        debug_assert!(!terminal_copy_indices.is_empty(), "terminal_copy_indices should not be empty");
        //dbg!(&terminal_patch);
        Ok(Ok(Merger{terminal_patch,terminal_copy_indices,final_size:stats.output_size as u64}))
    }
    pub fn merge<R:Read + Seek>(mut self, predecessor_patch:R) -> std::io::Result<Result<Merger,SummaryPatch>> {
        debug_assert!({
            let mut x = 0;
            for inst in self.terminal_patch.iter(){
                assert_eq!(x,inst.o_pos_start);
                x += inst.inst.len_in_o() as u64;
            }
            true
        });
        let (mut predecessor_patch,stats) = extract_patch_instructions(predecessor_patch)?;
        if stats.has_copy(){
            predecessor_patch = deref_non_copy_ss(predecessor_patch);
        }
        let mut terminal_copy_indices = Vec::with_capacity(self.terminal_copy_indices.len());
        let mut inserts = Vec::with_capacity(self.terminal_copy_indices.len());
        let mut shift = 0;
        for i in self.terminal_copy_indices{
            let DisInst { inst,.. } = self.terminal_patch[i].clone();
            let copy = inst.take_copy().expect("Expected Copy");
            //this a src window copy that we need to resolve from the predecessor patch.
            debug_assert!(copy.in_s());
            debug_assert!(!copy.vcd_trgt());
            let o_start = copy.ssp + copy.u_pos as u64; //ssp is o_pos, u is offset from that.
            let resolved = get_exact_slice(&predecessor_patch, o_start, copy.len_in_u()).unwrap();
            debug_assert_eq!(sum_len(&resolved), copy.len_in_o() as u64, "resolved: {:?} copy: {:?}",resolved,copy);
            find_copy_s(&resolved, i+shift, &mut terminal_copy_indices);
            shift += resolved.len() - 1;
            inserts.push((i, resolved));

        }
        //now we expand the old copy values with the derefd instructions.
        self.terminal_patch = expand_elements(self.terminal_patch, inserts);
        debug_assert_eq!(sum_len(&self.terminal_patch), self.final_size, "final size: {} sum_len: {}",self.final_size,sum_len(&self.terminal_patch));
        if terminal_copy_indices.is_empty(){
            Ok(Err(SummaryPatch(self.terminal_patch,self.final_size)))
        }else{
            self.terminal_copy_indices = terminal_copy_indices;
            Ok(Ok(self))
        }
    }
    pub fn finish(self)->SummaryPatch{
        SummaryPatch(self.terminal_patch,self.final_size)
    }

}

///This is returned when a terminal patch has no CopySS instructions.
///Merging additional patches will have no effect.
#[derive(Debug)]
pub struct SummaryPatch(Vec<VcdExtract>,u64);
impl SummaryPatch{
    pub fn write<W:Write>(self,sink:W,max_u_size:Option<usize>)->std::io::Result<W>{
        let max_u_size = max_u_size.unwrap_or(1<<28); //256MB
        let header = Header::default();
        let encoder = VCDEncoder::new(sink,header)?;
        let mut state = EncoderState{
            cur_o_pos: 0,
            max_u_size,
            cur_win: Vec::new(),
            sink: encoder,
            win_sum: Some(Default::default()),
        };
        let mut len = 0;
        for inst in self.0.into_iter(){
            len += inst.inst.len_in_o() as u64;
            state.apply_instruction(inst.inst)?;
        }
        debug_assert!(len == self.1, "before apply: {} final_size: {}",len,self.1);
        debug_assert!(state.cur_o_pos == self.1, "cur_o_pos: {} final_size: {}",state.cur_o_pos,self.1);
        state.flush_window()?;
        state.sink.finish()
    }
}

struct EncoderState<W>{
    cur_o_pos: u64,
    max_u_size: usize,
    cur_win: Vec<ExInstType<DisCopy>>,
    win_sum: Option<WindowHeader>,
    sink: VCDEncoder<W>,
}

impl<W:Write> EncoderState<W> {
    ///This is the 'terminal' command for moving the merger forward.
    fn apply_instruction(&mut self, instruction: ExInstType<DisCopy>) ->std::io::Result<()>{
        //this needs to do several things:
        //see if we our new instruction is the same win_indicator as our current window
        //see if our sss/ssp values need to be changed,
        // if so, will the change make us exceed our MaxWinSize?
        //   if so, we need to flush the window
        //   else we just update them
        self.handle_indicator(&instruction)?;//might flush
        let mut cur_inst = Some(instruction);
        while let Some(ci) = cur_inst.take(){
            let (cur_s,_) = self.cur_win_sizes();
            let inst_s = ci.src_range();
            let remaining_size = self.max_u_size as u64 - self.current_u_size() as u64;
            let remaining_size = if remaining_size == 0{
                self.flush_window()?;
                cur_inst = Some(ci);
                continue;
            }else{
                NonZeroU32::new(remaining_size as u32).unwrap()
            };
            match (cur_s,inst_s) {
                (Some((ssp,sss)), Some(r)) => {
                    if let Some(disjoint) = get_disjoint_range(ssp..ssp+sss, r.clone()){
                        let disjoint_len = (disjoint.end - disjoint.start) as u32;
                        //if this is larger than our remaining window, we need to flush, splitting won't help
                        if disjoint_len > remaining_size.into(){
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
            let split_at = ci.will_fit_window(remaining_size);
            match split_at {
                Some(len) =>{
                    debug_assert!(len.get() < ci.len_in_o() as u32, "split at: {} len: {}",len,ci.len_in_o());
                    let (first,second) = ci.split_at(len.get());
                    self.add_to_window(first);
                    debug_assert!(second.len_in_o()>0, "second len: {}",second.len_in_o());
                    cur_inst = Some(second);
                }
                None => self.add_to_window(ci),
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
    fn add_to_window(&mut self,next_inst:ExInstType<DisCopy>){
        //dbg!(&next_inst);

        //debug_assert!(output_start_pos == self.cur_o_pos, "inst output start pos: {} cur_o_pos: {}", output_start_pos, self.cur_o_pos);
        //adjust our current window
        let (src_range,trgt_win_size) = self.new_win_sizes(&next_inst);
        let ws = self.win_sum.get_or_insert(Default::default());
        if let Some((ssp,sss)) = src_range {
            ws.source_segment_position = Some(ssp);
            ws.source_segment_size = Some(sss);
            if ws.win_indicator != WinIndicator::VCD_SOURCE {
                ws.win_indicator = WinIndicator::VCD_SOURCE;
            }
        }
        ws.size_of_the_target_window = trgt_win_size;
        self.cur_o_pos += next_inst.len_in_o() as u64;
        self.cur_win.push(next_inst);
    }
    fn handle_indicator(&mut self,inst:&ExInstType<DisCopy>)->std::io::Result<()>{
        let win_sum = self.win_sum.get_or_insert(Default::default());

        match (win_sum.win_indicator,inst.inst_type().comp_indicator(&win_sum.win_indicator)){
            (_, None) => (),
            (WinIndicator::Neither, Some(set)) => {
                win_sum.win_indicator = set;
            },
            (WinIndicator::VCD_TARGET, Some(next)) |
            (WinIndicator::VCD_SOURCE, Some(next)) => {
                self.flush_window()?;
                let mut h = WindowHeader::default();
                h.win_indicator = next;
                self.win_sum = Some(h);
            },
        }
        Ok(())
    }
    ///(ssp,sss, size_of_the_target_window (T))
    fn new_win_sizes(&self,inst:&ExInstType<DisCopy>)->(Option<(u64,u64)>,u64){
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
        (ss,ws.size_of_the_target_window+inst.len_in_o() as u64)
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
        for inst in self.cur_win.drain(..) {
            match inst {
                ExInstType::Add(ExAdd{bytes}) => {
                    self.sink.next_inst(EncInst::ADD(bytes)).unwrap();
                },
                ExInstType::Run(r) => {
                    self.sink.next_inst(EncInst::RUN(r)).unwrap();
                },
                ExInstType::Copy(DisCopy { len_u, u_pos , ssp,.. }) => {
                    //we need to translate this so our u_pos is correct
                    let output_ssp = output_ssp.expect("output_ssp should be set here");
                    //our ssp is within (or coincident) to the bounds of the source segment
                    let copy_inst = if output_ssp > ssp{
                        let neg_shift = output_ssp - ssp;
                        COPY { len:len_u, u_pos: u_pos - neg_shift as u32 }
                    }else{
                        let pos_shift = ssp - output_ssp;
                        COPY { len:len_u, u_pos: u_pos + pos_shift as u32 }
                    };
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
}

fn expand_sequence(seq:&[VcdExtract],len:u32,cur_o_pos:&mut u64,output:&mut Vec<VcdExtract>) {
    let mut current_len = 0;
    // Calculate the effective length after considering truncation
    while current_len < len {
        for DisInst {  inst, .. } in seq.iter().cloned() {
            let end_pos = current_len + inst.len_in_o();
            let mut modified_instruction = inst;
            // After skip has been accounted for, directly clone and potentially modify for truncation
            // If adding this instruction would exceed the effective length, apply truncation
            if end_pos > len {
                let trunc_amt = end_pos - len;
                modified_instruction.trunc(trunc_amt);
            }
            let inst_len = modified_instruction.len_in_o();
            output.push(DisInst { o_pos_start:*cur_o_pos, inst: modified_instruction });
            current_len += inst_len;
            *cur_o_pos += inst_len as u64;
            // If we've reached or exceeded the effective length, break out of the loop
            if current_len >= len {
                debug_assert_eq!(current_len, len, "current_len: {} len: {}", current_len, len);
                break;
            }
        }
    }
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
fn expand_elements(mut target: Vec<VcdExtract>, inserts: Vec<(usize, Vec<VcdExtract>)>) -> Vec<VcdExtract>{
    // Calculate the total number of elements to be inserted to determine the new vector's length.
    let total_insertions: usize = inserts.iter().map(|(_, ins)| ins.len()).sum();
    let final_length = target.len() + total_insertions;

    // Allocate a new vector with the final required size.
    let mut result = Vec::with_capacity(final_length);

    // Sort inserts by position to process them in order.
    let mut sorted_inserts = inserts;
    sorted_inserts.sort_by_key(|k| k.0);

    target.reverse();
    // Trackers for the current position in the original vector and the inserts.
    let mut cur_idx = 0;
    let mut cur_o_pos = 0;
    for (insert_pos, insert_vec) in sorted_inserts {
        // Copy elements from the current position up to the insert position.
        while cur_idx < insert_pos {
            match target.pop() {
                Some(mut elem) => {
                    let len = elem.len_in_o();
                    elem.o_pos_start = cur_o_pos;
                    cur_o_pos += len as u64;
                    result.push(elem);
                    cur_idx += 1;
                }
                None => break,
            }
        }
        // Insert the new elements.
        for mut elem in insert_vec {
            let len = elem.len_in_o();
            elem.o_pos_start = cur_o_pos;
            cur_o_pos += len as u64;
            result.push(elem);
        }
        target.pop();//we get rid of the expanded element.
        cur_idx += 1;
    }

    // After processing all inserts, copy any remaining elements from the original vector.
    while let Some(mut elem) = target.pop() {
        let len = elem.len_in_o();
        elem.o_pos_start = cur_o_pos;
        cur_o_pos += len as u64;
        result.push(elem);
    }
    result
}


#[cfg(test)]
mod test_super {
    use crate::{applicator::apply_patch, reader::DeltaIndicator, RUN};

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
    fn make_patch_reader(patch_bytes:Vec<u8>)->Cursor<Vec<u8>>{
        Cursor::new(patch_bytes)
    }
    fn seq_patch() -> Cursor<Vec<u8>> {
        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WIN_HDR).unwrap();
        // Instructions
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 0 })).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 2, u_pos: 2 })).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 10, u_pos: 5 })).unwrap();
        // Force encoding
        let w = encoder.finish().unwrap().into_inner();
        make_patch_reader(w)
    }
    fn copy_patch() -> Cursor<Vec<u8>> {
        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WIN_HDR).unwrap();
        // Instructions
        encoder.next_inst(EncInst::COPY(COPY { len: 5, u_pos: 0 })).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 5, u_pos: 0 })).unwrap();
        // Force encoding
        let w = encoder.finish().unwrap().into_inner();
        make_patch_reader(w)
    }
    fn add_run_patch() -> Cursor<Vec<u8>> {
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
        make_patch_reader(w)
    }
    fn complex_patch()->Cursor<Vec<u8>>{
        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WIN_HDR).unwrap();
        // Instructions
        encoder.next_inst(EncInst::ADD("Y".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::RUN(RUN { len: 2, byte: b'Z' })).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 4 })).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 2, u_pos: 5 })).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 4, u_pos: 1 })).unwrap();
        // Force encoding
        let w = encoder.finish().unwrap().into_inner();
        make_patch_reader(w)
    }
    const SRC:&[u8] = b"01234";
    #[test]
    fn test_copy_seq(){
        //01234 Copy-> 0123401234 Seq-> 0230230230
        let answer = b"0230230230";
        let copy = copy_patch();
        let seq = seq_patch();
        let merger = Merger::new(seq).unwrap().unwrap();
        let merger = merger.merge(copy).unwrap().unwrap();
        let merged_patch = merger.finish().write(Vec::new(), None).unwrap();
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
        let merger = Merger::new(copy).unwrap().unwrap();
        let merger = merger.merge(seq).unwrap().unwrap();
        let merged_patch = merger.finish().write(Vec::new(), None).unwrap();
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
        let merger = Merger::new(add_run).unwrap().unwrap();
        let merger = merger.merge(copy).unwrap().unwrap();
        let merger = merger.merge(seq).unwrap().unwrap();
        let merged_patch = merger.finish().write(Vec::new(), None).unwrap();
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
        let merger = Merger::new(add_run).unwrap().unwrap();
        let merger = merger.merge(seq).unwrap().unwrap();
        let merger = merger.merge(copy).unwrap().unwrap();
        let merged_patch = merger.finish().write(Vec::new(), None).unwrap();
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
        let merger = Merger::new(comp).unwrap().unwrap();
        let merger = merger.merge(add_run).unwrap().unwrap();
        let merged_patch = merger.finish().write(Vec::new(), None).unwrap();
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
        let merger = Merger::new(seq).unwrap().unwrap();
        let merger = merger.merge(copy).unwrap().unwrap();
        let merger = merger.merge(comp).unwrap().unwrap();
        let merger = merger.merge(add_run).unwrap().unwrap_err();
        let merged_patch = merger.write(Vec::new(), None).unwrap();
        let mut cursor = Cursor::new(merged_patch);
        let mut output = Vec::new();
        //We don't need Src, since the last merge yielded SummaryPatch
        apply_patch(&mut cursor, None, &mut output).unwrap();
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
        let src = b"hello!";
        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(5), source_segment_position: Some(0), size_of_the_target_window:5 , delta_indicator: DeltaIndicator(0) }).unwrap();
        // Instructions
        encoder.next_inst(EncInst::COPY(COPY { len: 5, u_pos: 0 })).unwrap();
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(1), source_segment_position: Some(4), size_of_the_target_window:6 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::ADD(" w".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 0 })).unwrap();
        encoder.next_inst(EncInst::ADD("rld".as_bytes().to_vec())).unwrap();
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(1), source_segment_position: Some(5), size_of_the_target_window:1 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 0 })).unwrap();
        let p1 = encoder.finish().unwrap().into_inner();
        let p1_answer = b"hello world!";
        let mut cursor = Cursor::new(p1.clone());
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(Cursor::new(src.to_vec())), &mut output).unwrap();
        assert_eq!(output,p1_answer); //ensure our instructions do what we think they are.
        let patch_1 = make_patch_reader(p1);
        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(11), source_segment_position: Some(1), size_of_the_target_window:7 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::ADD("H".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 4, u_pos: 0 })).unwrap(); //ello
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 10 })).unwrap(); //'!'
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 4 })).unwrap(); //' '
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(7), source_segment_position: Some(0), size_of_the_target_window:14 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 7, u_pos: 0 })).unwrap(); //'Hello! '
        encoder.next_inst(EncInst::COPY(COPY { len: 12, u_pos: 7 })).unwrap(); //'Hello! Hello'
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
        let patch_2 = make_patch_reader(p2);
        let merger = Merger::new(patch_2).unwrap().unwrap();
        let merger = merger.merge(patch_1).unwrap().unwrap();
        let merged_patch = merger.finish().write(Vec::new(), None).unwrap();
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
    fn test_expand_seq(){
        let seq = vec![
            VcdExtract { o_pos_start: 0, inst: ExInstType::Copy(DisCopy { u_pos: 0, len_u: 2, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            VcdExtract { o_pos_start: 2, inst: ExInstType::Copy(DisCopy { u_pos: 4, len_u: 6, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            VcdExtract { o_pos_start: 8, inst: ExInstType::Copy(DisCopy { u_pos: 12, len_u: 4, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
        ];
        let mut output = Vec::new();
        expand_sequence(&seq, 15, &mut 0,&mut output);
        let result = vec![
            VcdExtract { o_pos_start: 0, inst: ExInstType::Copy(DisCopy { u_pos: 0, len_u: 2, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            VcdExtract { o_pos_start: 2, inst: ExInstType::Copy(DisCopy { u_pos: 4, len_u: 6, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            VcdExtract { o_pos_start: 8, inst: ExInstType::Copy(DisCopy { u_pos: 12, len_u: 4, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            VcdExtract { o_pos_start: 12, inst: ExInstType::Copy(DisCopy { u_pos: 0, len_u: 2, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            VcdExtract { o_pos_start: 14, inst: ExInstType::Copy(DisCopy { u_pos: 4, len_u: 1, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
        ];
        assert_eq!(output, result, "Output should contain a truncated instruction");
    }
}