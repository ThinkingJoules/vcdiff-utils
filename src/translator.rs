use std::{io::{Read, Seek}, ops::Range};

use crate::{decoder::{DecInst, VCDDecoder, VCDiffDecodeMsg}, reader::{read_header, read_window_header, WinIndicator, WindowSummary}, ADD, COPY, RUN};



pub fn gather_summaries<R: Read + Seek>(mut patch_data:R)-> std::io::Result<Vec<WindowSummary>>{
    let header = read_header(&mut patch_data)?;
    let mut summaries = Vec::new();
    while let Ok(ws) = read_window_header(&mut patch_data, header.encoded_size() as u64) {
        summaries.push(ws);
    }
    Ok(summaries)
}

fn find_dep_ranges(summaries: &[WindowSummary])->Vec<Range<u64>>{
    let mut ranges = Vec::new();
    for ws in summaries.iter().rev() {
        if let WinIndicator::VCD_TARGET = ws.win_indicator {
            let ssp = ws.source_segment_position.unwrap() as u64;
            let sss = ws.source_segment_size.unwrap() as u64;
            ranges.push(ssp..ssp+sss);
        }
    }
    //sort with the smallest last
    ranges.sort_by(|a,b|a.start.cmp(&b.start));
    ranges
}
/*
When a handle encounters a TrgtSourcedWindow, it must resolve the source data from a SrcSourcedWindow.
We scanned the file for all the summaries so we know which instructions to retain that will be ref'd later
The merge handle can only return Src oriented instructions, since Trgt is synthetic because we are merging.
When a handle is interrogated for a position, it must resolve the source data for the TrgtSourcedWindow
This mean returning one or more instructions that represent the translation.
These instructions will not exist anywhere, we need to adapt existing ones to match the Target Copy command.
If the instruction is in a source (or null) window we will only ever return a single instruction.
COPYs will be the most mutilated during merging, as ADD/RUNs take precedence.
COPYs might be merged if there are no ADD/RUNs in any of the patches for a run of bytes.?????
NO. A COPY in the latest patch only ends because of a ADD/RUN, so we cannot merge them.
So it will never get larger, only smaller if older patches have ADD/RUNs that overlap.
I think that is right.

When we ask for a byte in MergeHandle, it will return a modified instruction that starts at that byte.
If it is a TrgtSourcedWindow, it will return a Vec of instructions that represent the translation.
The idea is that the caller will be working instruction by instruction, but also byte by byte.
Only Copies will defer to earlier patches trying to find an ADD/RUN that covers the byte.
So, it only works byte by byte when the last patch is a COPY. We need to find the last ADD/RUN that covers the byte.
We will probably only ever end up truncating COPY commands. Not sure when a merge or lengthening would happen.
*/

///This orchestrates a single patch file.
pub struct VCDTranslator<R>{
    ///The reader for the VCDIFF file
    decoder: VCDDecoder<R>,
    ///These are sorted, so we just look at 'last' and pop it off once we have passed the end position.
    dependencies: Vec<Range<u64>>,
    windows: Vec<WindowSummary>,
    ///Used to resolve dependencies for TrgtSourcedWindows
    src_data: Vec<SparseInst>,
    cur_window_inst: Vec<Inst>,
    cur_o_pos: u64,
    cur_u_pos: usize,
}

impl<R:Read+Seek> VCDTranslator<R> {
    pub fn new(decoder: VCDDecoder<R>, patch_reader:R) -> std::io::Result<Self> {
        let windows = gather_summaries(patch_reader)?;
        let dependencies = find_dep_ranges(&windows);
        Ok(VCDTranslator {
            decoder,
            dependencies,
            //we throw out the windows, the decoder will forward them to us
            windows:Vec::with_capacity(windows.len()),
            src_data: Vec::new(),
            cur_window_inst: Vec::new(),
            cur_o_pos: 0,
            cur_u_pos: 0,
        })
    }
    pub fn cur_win(&self)->Option<&WindowSummary>{
        self.windows.last()
    }
    pub fn cur_t_start(&self)->Option<usize>{
        if let Some(ws) = self.cur_win() {
            let s = ws.source_segment_size.map(|x|x as usize);
            return s;
        }
        None
    }
    pub fn interrogate(&mut self, o_position: u64) -> std::io::Result<Option<SparseInst>> {
        while self.cur_o_pos < o_position {//will fall through if it is a prev inst.
            let count = self.next_op()?;
            if count == 0 {
                return Ok(None);
            }
        }
        Ok(Some(self.interrogate_prev(o_position)))
    }
    fn interrogate_prev(&self, o_position: u64)->SparseInst{
        let mut pos = self.cur_o_pos;
        for inst in self.cur_window_inst.iter().rev() {
            let cur_inst_start = pos - inst.len() as u64;
            if (cur_inst_start..pos).contains(&o_position) {
                return SparseInst {
                    o_start: cur_inst_start,
                    inst: inst.clone(),
                };
            }
            pos = cur_inst_start;
        }
        panic!("o_position must be within the range of the current window");
    }
    ///This returns a 'root' instruction that controls the byte at the given position in the output stream.
    ///This operates one op code at a time (one or two instructions).
    ///Returns the number of instructions that were advanced.
    ///0 is returned if there are no more instructions to advance (end of file).
    ///This can return more than 2, since we might need to resolve a TrgtSourcedWindow or a COPY in T (or Both).
    fn next_op(&mut self)->std::io::Result<usize>{
        //a form of double entry accounting to make sure all the translations are correct.
        assert!(self.cur_o_pos == self.decoder.position(), "Decoder and Merge Handle are out of sync!");
        loop {
            match self.decoder.next()?{
                VCDiffDecodeMsg::WindowSummary(ws) => {
                    self.cur_u_pos = ws.source_segment_size.map(|x|x as usize).unwrap_or(0);
                    self.windows.push(ws);
                },
                VCDiffDecodeMsg::Inst { o_start,first, second } => {
                    //here we need to resolve instructions for merging
                    assert!(o_start == self.cur_o_pos, "We are out of sync with the output stream");
                    let mut count = 0;
                    count += self.resolve_inst(first);
                    if let Some(second) = second {
                        count += self.resolve_inst(second);
                    }
                    return Ok(count);
                },
                VCDiffDecodeMsg::EndOfWindow => {
                    self.cur_window_inst.clear();
                },
                VCDiffDecodeMsg::EndOfFile => return Ok(0),
            };
        }
    }
    fn resolve_inst(&mut self,inst:DecInst)->usize{
        if inst.is_copy() {
            let c = inst.take_copy();
            self.resolve_copy(c)
        }else{
            self.store_inst(Inst::from_dec(inst));
            1
        }
    }

    fn resolve_copy(&mut self, mut copy: COPY) -> usize {
        let CopyInfo { mut state, seq, copy_addr } = self.determine_copy_type(&copy);
        // handle the sequence first? We would reduce the len of the COPY to the pattern
        // then we would resolve the instructions.
        if let Some(ImplicitSeq { pattern, .. }) = seq {
            copy.len = pattern as u32;
        }
        let initial = Inst::Copy { copy, addr:copy_addr };
        let mut buf = Vec::with_capacity(15);
        buf.push(initial);

        loop {
            match state {
                CopyState::TranslateLocal | CopyState::TranslateLocalThenGlobal=> {
                    self.resolve_local(&mut buf);
                }
                CopyState::TranslateGlobal => {
                    self.resolve_global(&mut buf);
                }
                CopyState::Resolved => {
                    break;
                },
            }
            state.next()
        }
        //buf should now contain all the resolved instructions for the given COPY input
        //this might have been a no-op, but we still need to handle the sequence case
        let count = buf.len();
        if let Some(ImplicitSeq { len, .. }) = seq {
            self.store_inst(Inst::Sequence(Sequence { inst: buf, len: len as u32, skip: 0, trunc: 0}));
        }else{
            self.cur_window_inst.reserve(count);
            for inst in buf.drain(..) {
                self.store_inst(inst);
            }
        }
        count
    }
    // This handles translation using the current window's data
    fn resolve_local(&mut self, input_buffer:&mut Vec<Inst>){
        //it is possible to have window with no reference source
        //In this case we will always get TranslateLocal, but there is no S to transfer to.
        //In that case our resulting vec will only have ADD/RUNs and nothing else
        //(or Sequence, but it will only have ADD/RUNs)
        //This is probably unlikely, but happens always if the patch source is null (compression only)

        //The more likely case is that we have ref src (Src Or Trgt), and we need to translate to commands from S.

        //For each (COPY) instruction we find in the buffer, we need to find another COPY in S or an ADD/RUN (anywhere)
        //Some instructions might already be in S and need no translation
        let mut to_drain = Vec::new();
        to_drain.reserve(input_buffer.capacity() + 15);
        //to_drain now has input, and input is now our 'output'
        std::mem::swap(input_buffer, &mut to_drain);
        //change name for sanity
        let output_buffer = input_buffer;

        for inst in to_drain.drain(..){
            match inst.has_copy(){
                InstCopy::No(a) => output_buffer.push(a),
                InstCopy::InSequence(mut s) => {
                    //recursively resolve the inner instructions
                    //this happens when we have had an earlier implicit sequence
                    //This is possible if an earlier COPY in T was a seq, and we are COPYing those bytes again
                    //So this would require resolving those inner instructions as they might also be in T
                    //This is fine, as long as we get eventually get COPYs in S or ADD/RUNs.
                    self.resolve_local(&mut s.inst);
                    output_buffer.push(Inst::Sequence(s));
                },
                InstCopy::IsCopy { copy:COPY { len, u_pos }, addr } => {
                    if matches!(&addr, CopyAddr::Source {..}) {
                        //we are doing a local translation, and it is already in S
                        //we do nothing.
                        output_buffer.push(Inst::Copy { copy:COPY { len, u_pos }, addr });
                        continue;
                    }
                    //here u_pos should partly be in T of U
                    debug_assert!(u_pos < self.cur_u_pos as u32, "The COPY position is not before our current output position in U");
                    let mut slice = self.get_local_slice(u_pos, len);
                    //sanity check for len
                    debug_assert!(slice.iter().map(|x|x.len()).sum::<u32>() == len, "The slice length does not match the COPY length");
                    output_buffer.append(&mut slice);
                },
            }
        }
    }

    fn get_local_slice(&self, u_pos: u32, len: u32) -> Vec<Inst> {
        let mut slice = Vec::new();
        let mut pos = 0;
        let slice_end = u_pos + len;
        for inst in self.cur_window_inst.iter() {
            let len = inst.len();
            let cur_inst_end = pos + len;
            if cur_inst_end <= u_pos {
                //we are not there yet
            } else if pos >= u_pos && slice.is_empty() {
                //we are in the middle of the first instruction
                let mut start = inst.clone();
                let skip = u_pos - pos;
                start.skip(skip);
                slice.push(start);
            }else if cur_inst_end >= slice_end { //non-aligned end state
                let mut end = inst.clone();
                let truc = cur_inst_end - slice_end;
                end.trunc(truc);
                slice.push(end);
                break;
            } else if cur_inst_end < slice_end {
                //take the whole instruction
                debug_assert!(!slice.is_empty(), "We should have started the slice");
                slice.push(inst.clone());
            }
            pos += len;
        }
        slice
    }

    fn get_global_slice(&self, u_pos: u32, len: u32) -> Vec<Inst> {
        //first we need to get our source window size and position
        let cur_win = self.cur_win().unwrap();
        debug_assert!( cur_win.is_vcd_target(), "We can only translate from a TargetSourcedWindow");
        let ssp = cur_win.source_segment_position.unwrap();
        let o_pos = ssp + u_pos as u64;
        //sanity check that we are indeed in S.
        debug_assert!(u_pos + len <= cur_win.source_segment_size.unwrap() as u32);
        let mut slice = Vec::new();
        let slice_end = o_pos + len as u64;
        let mut last = 0; //to assert contiguous SparseInst
        for SparseInst { o_start, inst } in self.src_data.iter() {
            debug_assert!(if !slice.is_empty(){last == *o_start}else{true});
            let len = inst.len();
            let cur_inst_end = o_start + len as u64;
            if cur_inst_end <= o_pos {
                //we are not there yet
            } else if *o_start >= o_pos && slice.is_empty() {
                //we are in the middle of the first instruction
                let mut start = inst.clone();
                let skip = o_pos - o_start;
                start.skip(skip as u32);
                slice.push(start);
            } else if cur_inst_end >= slice_end { //non-aligned end state
                let mut end = inst.clone();
                let trunc = cur_inst_end - slice_end;
                end.trunc(trunc as u32);
                slice.push(end);
                break;
            }else if cur_inst_end < slice_end {// whole instruction mid-slice
                //take the whole instruction
                debug_assert!(!slice.is_empty(), "We should have started the slice");
                slice.push(inst.clone());
            }
            last = cur_inst_end;
        }
        slice
    }

    // This handles cross-window translation using previously resolved instructions
    fn resolve_global(&mut self,input_buffer:&mut Vec<Inst>){
        //the buffer here only contains instructions in S or they are ADD/RUNs

        //we need to translate COPYs since our cur_win is TargetSourced.
        let mut to_drain = Vec::new();
        to_drain.reserve(input_buffer.capacity() + 15);
        //to_drain now has input, and input is now our 'output'
        std::mem::swap(input_buffer, &mut to_drain);
        //change name for sanity
        let output_buffer = input_buffer;

        for inst in to_drain.drain(..){
            match inst.has_copy(){
                InstCopy::No(a) => output_buffer.push(a),
                InstCopy::InSequence(mut s) => {
                    //recursively resolve the inner instructions
                    //this happens when we have had an earlier implicit sequence
                    self.resolve_global(&mut s.inst);
                    output_buffer.push(Inst::Sequence(s));
                },
                InstCopy::IsCopy { copy:COPY { len, u_pos }, addr } => {
                    assert!(matches!(addr, CopyAddr::Unresolved), "The COPY address should always be unresolved in a TrgtSourcedWindow");
                    //here u_pos should already be in S (resolve local should be ran first)
                    //S references our Output stream
                    let mut slice = self.get_global_slice(u_pos, len);
                    //sanity check for len
                    debug_assert!(slice.iter().map(|x|x.len()).sum::<u32>() == len, "The slice length does not match the COPY length");
                    output_buffer.append(&mut slice);
                },
            }
        }
    }
    fn determine_copy_type(&self,copy:&COPY)->CopyInfo{
        //Every COPY can have between 0 and 2 layers of indirection.
        //0: COPY entirely in S in U and VCD_SOURCE
        //1: COPY in U that overflows or is entirely in T and VCD_SOURCE
        //1: COPY in entirely in S in U and VCD_TARGET
        //2: COPY in U that overflows or is entirely in T and VCD_TARGET
        //There is an implicit Sequence if COPYs length exceeds our current output position in U
        //This is regardless of where the COPY starts, only where it ends.
        let COPY { len, u_pos } = copy;
        let u_end = u_pos + len;
        let t_start = self.cur_t_start().unwrap_or(0) as u32;
        let cur_u = self.cur_u_pos as u32;
        let ws = self.cur_win().unwrap();
        let is_trgt = ws.is_vcd_target();
        let sss = ws.source_segment_size;
        let ssp = ws.source_segment_position;
        let len = *len as usize;
        if u_end <= t_start { // COPY entirely in S in U
            return CopyInfo {
                state: if is_trgt { CopyState::TranslateGlobal } else { CopyState::Resolved },
                seq: None,
                copy_addr: if is_trgt { CopyAddr::Unresolved  } else { CopyAddr::Source { sss: sss.unwrap(), ssp:ssp.unwrap() } }
            };
        }
        if u_end <= cur_u { // COPY in U that overflows or is entirely in T
            return CopyInfo {
                state: if is_trgt { CopyState::TranslateLocalThenGlobal } else { CopyState::TranslateLocal },
                seq: None,
                copy_addr: CopyAddr::Unresolved
            };
        }
        // Remaining case: u_end > t_start && u_end > cur_u
        let pattern = (cur_u - u_pos) as usize;
        CopyInfo {
            state: if is_trgt { CopyState::TranslateLocalThenGlobal } else { CopyState::TranslateLocal },
            seq: Some(ImplicitSeq { pattern, len }),
            copy_addr: CopyAddr::Unresolved
        }
    }

    fn update_deps(&mut self){
        while let Some(r) = self.dependencies.last() {
            if r.end <= self.cur_o_pos {
                self.dependencies.pop();
            }else{
                break;
            }
        }
    }
    fn store_inst(&mut self,inst:Inst){
        self.update_deps();//we have added instructions since last call, make sure last is correct.
        let len = inst.len();
        let new_o_pos = self.cur_o_pos + len as u64;
        //do later things depend on this?
        if let Some(r) = self.dependencies.last() {
            if r.start <= self.cur_o_pos || r.end > new_o_pos {
                self.src_data.push(SparseInst {
                    o_start: self.cur_o_pos,
                    inst: inst.clone(),
                });
            }
        }
        self.cur_u_pos += len as usize;
        self.cur_o_pos = new_o_pos;
        self.cur_window_inst.push(inst);
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct ImplicitSeq{
    pattern:usize,
    len:usize,
}
struct CopyInfo{
    state:CopyState,
    seq:Option<ImplicitSeq>,
    copy_addr:CopyAddr,
}

/// States:
/// Resolved -> DONE
///
/// TranslateLocal -> TranslateGlobal -> DONE
///
/// TranslateGlobal -> DONE
enum CopyState{
    ///Buffer contains translated instructions
    Resolved,
    ///Buffer contains one or more instructions that need internal resolution
    TranslateLocal,
    ///Buffer contains one or more instructions that need cross-window resolution
    TranslateGlobal,
    ///Buffer contains one or more instructions that need both forms of resolution
    TranslateLocalThenGlobal,
}

impl CopyState {
    fn next(&mut self){
        match self {
            CopyState::TranslateLocal => *self = CopyState::Resolved,
            CopyState::TranslateGlobal => *self = CopyState::Resolved,
            CopyState::TranslateLocalThenGlobal => *self = CopyState::TranslateGlobal,
            _ => unreachable!(),
        }
    }

}


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sequence{pub inst:Vec<Inst>, pub len:u32, pub skip:u32, pub trunc:u32}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Inst{
    ///This is a source instruction
    Add(ADD),
    Run(RUN),
    Copy{copy:COPY, addr:CopyAddr},
    Sequence(Sequence),
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CopyAddr{
    Unresolved,
    Source{sss:u64, ssp:u64},
}
impl Inst {
    fn from_dec(inst:DecInst)->Self{
        match inst {
            DecInst::Add(a) => Inst::Add(a),
            DecInst::Copy(c) => Inst::Copy{copy:c, addr:CopyAddr::Unresolved},
            DecInst::Run(r) => Inst::Run(r),
        }
    }
    fn len(&self) -> u32 {
        match self {
            Inst::Add(inst) => {inst.len()},
            Inst::Run(inst) => {inst.len()},
            Inst::Copy { copy, .. } => {copy.len()},
            Inst::Sequence(Sequence { len, skip, trunc, ..}) => {
                //cannot be zero len, we have a logical error somewhere
                debug_assert!(skip + trunc < *len);
                len - (skip + trunc)
            },
        }

    }
    fn skip(&mut self, amt: u32) {
        if amt == 0 {return}
        match self {

            Inst::Sequence(Sequence { skip, .. }) => {*skip += amt as u32;},
            Inst::Add(inst) => {inst.skip(amt);},
            Inst::Run(inst) => {inst.skip(amt);},
            Inst::Copy { copy, .. } => {copy.skip(amt);},
        }
    }
    fn trunc(&mut self, amt: u32){
        if amt == 0 {return}
        match self {
            Inst::Add(inst) => {inst.trunc(amt);},
            Inst::Run(inst) => {inst.trunc(amt);},
            Inst::Copy { copy, .. } => {copy.trunc(amt);},
            Inst::Sequence(Sequence {trunc ,..}) => {*trunc += amt as u32;},
        }
    }
    fn has_copy(self)->InstCopy{
        match self{
            Inst::Copy { copy, addr  } => {
                //always
                InstCopy::IsCopy { copy, addr }
            },
            Inst::Sequence(Sequence { inst, len, skip, trunc }) => {
                //sometimes
                if inst.iter().any(|x|matches!(x,Inst::Copy{..})) {
                    InstCopy::InSequence(Sequence { inst, len, skip, trunc })
                }else{
                    InstCopy::No(Inst::Sequence(Sequence { inst, len, skip, trunc }))
                }
            },
            a => InstCopy::No(a)
        }
    }
}
enum InstCopy{
    No(Inst),
    InSequence(Sequence),
    IsCopy{ copy:COPY, addr: CopyAddr  }
}
pub struct SparseInst{
    ///The position in the target output stream where this instruction starts outputting bytes
    pub o_start: u64,
    pub inst: Inst,
}

impl ADD{
    fn len(&self)->u32{
        self.len
    }
    fn skip(&mut self,amt:u32){
        self.len-=amt;
        self.p_pos+=amt as u64;
    }
    fn trunc(&mut self,amt:u32){
        self.len-=amt;
    }
}
impl RUN{
    fn len(&self)->u32{
        self.len
    }
    fn skip(&mut self,amt:u32){
        self.len-=amt as u32;
    }
    fn trunc(&mut self,amt:u32){
        self.len-=amt;
    }
}
impl COPY{
    fn len(&self)->u32{
        self.len
    }
    fn skip(&mut self,amt:u32){
        self.len-=amt;
        self.u_pos+=amt;
    }
    fn trunc(&mut self,amt:u32){
        self.len-=amt;
    }
}
// impl DecInst{
//     fn skip(&mut self,amt:u32){
//         match self {
//             DecInst::Add(ADD { len, p_pos }) =>{*len-=amt; *p_pos+=amt as u64;},
//             DecInst::Copy(COPY { len, u_pos }) => {*len-=amt; *u_pos+=amt;},
//             DecInst::Run(RUN{ len, .. }) => {*len-=amt as u32;},
//         }
//     }
//     fn trunc(&mut self,amt:u32){
//         match self {
//             DecInst::Add(ADD { len, .. }) |
//             DecInst::Copy(COPY { len, .. }) |
//             DecInst::Run(RUN{ len, .. }) => {*len-=amt;},
//         }
//     }
// }

