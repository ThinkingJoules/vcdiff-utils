use std::{io::{Read, Seek}, ops::Range};

use crate::{decoder::{DecInst, VCDDecoder, VCDiffDecodeMsg}, reader::{read_header, read_window_header, WinIndicator, WindowSummary}, ADD, COPY, RUN};



pub fn gather_summaries<R: Read + Seek>(patch_data:&mut R)-> std::io::Result<Vec<WindowSummary>>{
    let header = read_header(patch_data)?;
    let mut summaries = Vec::new();
    let mut win_start_pos = header.encoded_size() as u64;
    while let Ok(ws) = read_window_header(patch_data, win_start_pos) {
        win_start_pos = ws.end_of_window();
        summaries.push(ws);
        patch_data.seek(std::io::SeekFrom::Start(win_start_pos))?;
    }
    Ok(summaries)
}

pub fn find_dep_ranges(summaries: &[WindowSummary])->Vec<Range<u64>>{
    let mut ranges = Vec::new();
    for ws in summaries.iter().rev() {
        if let WinIndicator::VCD_TARGET = ws.win_indicator {
            let ssp = ws.source_segment_position.unwrap() as u64;
            let sss = ws.source_segment_size.unwrap() as u64;
            ranges.push(ssp..ssp+sss);
        }
    }
    let mut ranges = merge_ranges(ranges);
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
    cur_window_inst: Vec<(ProcessInst,u32)>,
    cur_o_pos: u64,
    cur_u_pos: usize,
}

impl<R:Read+Seek> VCDTranslator<R> {
    pub fn new(mut decoder: VCDDecoder<R>) -> std::io::Result<Self> {
        let windows = gather_summaries(&mut decoder.reader().get_reader(0)?)?;
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
    pub fn get_reader(&mut self, at_from_start:u64)->std::io::Result<&mut R>{
        self.decoder.reader().get_reader(at_from_start)
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
        while self.cur_o_pos <= o_position {//will fall through if it is a prev inst.
            let count = self.next_op()?;
            if count == 0 {
                break;
            }
        }
        Ok(self.interrogate_prev(o_position))
    }
    fn interrogate_prev(&self, o_position: u64)->Option<SparseInst>{
        let mut pos = self.cur_o_pos;
        for (inst,len_in_t) in self.cur_window_inst.iter().rev() {
            let cur_inst_start = pos - *len_in_t as u64;
            if (cur_inst_start..pos).contains(&o_position) {
                return Some(inst.to_sparse_inst(cur_inst_start));
            }
            pos = cur_inst_start;
        }
        None
    }
    ///This returns a 'root' instruction that controls the byte at the given position in the output stream.
    ///This operates one op code at a time (one or two instructions).
    ///Returns the number of instructions that were advanced.
    ///0 is returned if there are no more instructions to advance (end of file).
    ///This can return more than 2, since we might need to resolve a TrgtSourcedWindow or a COPY in T (or Both).
    fn next_op(&mut self)->std::io::Result<usize>{
        //a form of double entry accounting to make sure all the translations are correct.
        assert!(self.cur_o_pos == self.decoder.position(), "Decoder and Merge Handle are out of sync! {} != {}", self.cur_o_pos, self.decoder.position());
        loop {
            match self.decoder.next()?{
                VCDiffDecodeMsg::WindowSummary(ws) => {
                    self.cur_u_pos = ws.source_segment_size.map(|x|x as usize).unwrap_or(0);
                    self.windows.push(ws);
                },
                VCDiffDecodeMsg::Inst { o_start,first, second } => {
                    //here we need to resolve instructions for merging
                    assert!(o_start == self.cur_o_pos, "We are out of sync with the output stream {} != {}", self.cur_o_pos, self.decoder.position());
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
            self.store_inst(ProcessInst::from_dec(inst));
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
        let initial = ProcessInst::Copy { copy, addr:copy_addr };
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
            self.store_inst(ProcessInst::Sequence(ProcSequence { inst: buf, len: len as u32, skip: 0, trunc: 0}));
        }else{
            self.cur_window_inst.reserve(count);
            for inst in buf.drain(..) {
                self.store_inst(inst);
            }
        }
        count
    }
    // This handles translation using the current window's data
    fn resolve_local(&mut self, input_buffer:&mut Vec<ProcessInst>){
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
                    output_buffer.push(ProcessInst::Sequence(s));
                },
                InstCopy::IsCopy { copy:COPY { len, u_pos }, addr } => {
                    if matches!(&addr, CopyAddr::Source {..}) {
                        //we are doing a local translation, and it is already in S
                        //we do nothing.
                        output_buffer.push(ProcessInst::Copy { copy:COPY { len, u_pos }, addr });
                        continue;
                    }
                    //here u_pos should partly be in T of U
                    debug_assert!(u_pos < self.cur_u_pos as u32, "The COPY position is not before our current output position in U");
                    let mut slice = self.get_local_slice(u_pos, len);
                    // //sanity check for len
                    // debug_assert!(slice.iter().map(|x|x.len_in_u()).sum::<u32>() == len, "The slice length does not match the COPY length");
                    output_buffer.append(&mut slice);
                },
            }
        }
    }

    fn get_local_slice(&self, u_pos: u32, len: u32) -> Vec<ProcessInst> {
        let mut slice = Vec::new();
        let mut pos = 0;
        let src_size = self.cur_t_start().unwrap_or(0) as u32;
        let t_pos = u_pos - src_size;
        let slice_end = t_pos + len;
        let range = t_pos..slice_end;
        for (inst,len_in_t) in self.cur_window_inst.iter() {
            let len = len_in_t;//inst.len_in_t(src_size+pos);
            let cur_inst_end = pos + len;
            if let Some(overlap) = range_overlap(&(pos..cur_inst_end), &range) {
                let mut cur_inst = inst.clone();
                if overlap.start >= u_pos {
                    let skip = u_pos - pos;
                    cur_inst.skip(skip);
                }
                if cur_inst_end >= slice_end {
                    let trunc = cur_inst_end - slice_end;
                    cur_inst.trunc(trunc);
                }
                slice.push(cur_inst);
            }
            if cur_inst_end >= slice_end {
                break;
            }
            pos += len;
        }
        slice
    }

    fn get_global_slice(&self, u_pos: u32, len: u32) -> Vec<ProcessInst> {
        //first we need to get our source window size and position
        let cur_win = self.cur_win().unwrap();
        debug_assert!( cur_win.is_vcd_target(), "We can only translate from a TargetSourcedWindow");
        let ssp = cur_win.source_segment_position.unwrap();
        let o_pos = ssp + u_pos as u64;
        //sanity check that we are indeed in S.
        debug_assert!(u_pos + len <= cur_win.source_segment_size.unwrap() as u32);
        let slice_end = o_pos + len as u64;
        let range = o_pos as u64..slice_end;

        let mut slice = Vec::new();
        let mut last = 0; //to assert contiguous SparseInst
        for SparseInst { o_start, inst } in self.src_data.iter() {
            debug_assert!(if !slice.is_empty(){last == *o_start}else{true});

            let len = inst.len();
            let cur_inst_end = o_start + len as u64;
            if let Some(overlap) = range_overlap(&(*o_start..cur_inst_end), &range) {
                let mut cur_inst = inst.as_proc_inst();

                if overlap.start > *o_start {
                    let skip = overlap.start - *o_start;
                    cur_inst.skip(skip as u32);
                }
                if overlap.end < cur_inst_end {
                    let trunc = cur_inst_end - overlap.end;
                    cur_inst.trunc(trunc as u32);
                }
                slice.push(cur_inst);
            }
            if cur_inst_end >= slice_end {
                break;
            }
            last = cur_inst_end;
        }
        slice
    }

    // This handles cross-window translation using previously resolved instructions
    fn resolve_global(&mut self,input_buffer:&mut Vec<ProcessInst>){
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
                    output_buffer.push(ProcessInst::Sequence(s));
                },
                InstCopy::IsCopy { copy:COPY { len, u_pos }, addr } => {
                    assert!(matches!(addr, CopyAddr::Unresolved), "The COPY address should always be unresolved in a TrgtSourcedWindow");
                    //here u_pos should already be in S (resolve local should be ran first)
                    //S references our Output stream
                    let mut slice = self.get_global_slice(u_pos, len);
                    // //sanity check for len
                    // debug_assert!(slice.iter().map(|x|x.len_in_u()).sum::<u32>() == len, "The slice length does not match the COPY length");
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
        let len = (u_end - cur_u) as usize;
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
    fn store_inst(&mut self,inst:ProcessInst){
        self.update_deps();//we have added instructions since last call, make sure last is correct.
        let len = inst.len_in_t(self.cur_u_pos as u32);
        let new_o_pos = self.cur_o_pos + len as u64;
        //do later things depend on this?
        if let Some(r) = self.dependencies.last() {
            if r.start <= self.cur_o_pos || r.end > new_o_pos {
                self.src_data.push(inst.to_sparse_inst(self.cur_o_pos));
            }
        }
        self.cur_u_pos += len as usize;
        self.cur_o_pos = new_o_pos;
        self.cur_window_inst.push((inst,len));
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
#[derive(Debug)]
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
struct ProcSequence{inst:Vec<ProcessInst>, len:u32, skip:u32, trunc:u32}

impl ProcSequence {
    fn to_seq(&self) -> Sequence {
        Sequence {
            inst: self.inst.iter().map(|x|x.to_inst()).collect(),
            len: self.len,
            skip: self.skip,
            trunc: self.trunc,
        }
    }
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sequence{pub inst:Vec<Inst>, pub len:u32, pub skip:u32, pub trunc:u32}

impl Sequence {
    fn to_proc_seq(&self) -> ProcSequence {
        ProcSequence {
            inst: self.inst.iter().map(|x|x.as_proc_inst()).collect(),
            len: self.len,
            skip: self.skip,
            trunc: self.trunc,
        }
    }
    pub(crate) fn skip(&mut self, amt: u32) {
        self.skip += amt as u32;
    }
    pub(crate) fn len(&self) -> u32 {
        //cannot be zero len, we have a logical error somewhere
        debug_assert!(self.skip + self.trunc < self.len);
        self.len - (self.skip + self.trunc)
    }
}
///Disassociated Copy (from the window it was found in).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DisCopy{pub copy:COPY, pub sss:u64, pub ssp:u64}


#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Inst{
    ///This is a source instruction
    Add(ADD),
    Run(RUN),
    ///The source window this COPY command refers to
    Copy(DisCopy),
    Sequence(Sequence),
}

impl Inst {
    fn as_proc_inst(&self) -> ProcessInst {
        match self {
            Inst::Add(inst) => ProcessInst::Add(inst.clone()),
            Inst::Run(inst) => ProcessInst::Run(inst.clone()),
            Inst::Copy (DisCopy{ copy, sss, ssp }) => ProcessInst::Copy { copy: copy.clone(), addr: CopyAddr::Source { sss: *sss, ssp: *ssp } },
            Inst::Sequence(seq) => ProcessInst::Sequence(seq.to_proc_seq()),
        }
    }
    ///We don't worry about T/U distinction since Sequences are now explicit.
    pub(crate) fn len(&self) -> u32 {
        match self {
            Self::Add(inst) => {inst.len()},
            Self::Run(inst) => {inst.len()},
            Self::Copy (copy) => {copy.copy.len_in_u()},
            Self::Sequence(Sequence { len, skip, trunc, ..}) => {
                //cannot be zero len, we have a logical error somewhere
                debug_assert!(skip + trunc < *len);
                len - (skip + trunc)
            },
        }

    }

}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ProcessInst{
    ///This is a source instruction
    Add(ADD),
    Run(RUN),
    Copy{copy:COPY, addr:CopyAddr},
    Sequence(ProcSequence),
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CopyAddr{
    Unresolved,
    Source{sss:u64, ssp:u64},
}
impl ProcessInst {
    fn to_sparse_inst(&self, o_start: u64) -> SparseInst {
        SparseInst {
            o_start,
            inst: self.to_inst(),
        }
    }
    fn to_inst(&self) -> Inst {
        match self {
            ProcessInst::Add(inst) => Inst::Add(inst.clone()),
            ProcessInst::Run(inst) => Inst::Run(inst.clone()),
            ProcessInst::Copy { copy, addr:CopyAddr::Source{ sss, ssp } } => Inst::Copy(DisCopy { copy: copy.clone(), sss:*sss, ssp:*ssp }),
            ProcessInst::Sequence(seq) => Inst::Sequence(seq.to_seq()),
            ProcessInst::Copy { .. } => panic!("Cannot convert unresolved COPY to Inst"),
        }
    }
    fn from_dec(inst:DecInst)->Self{
        match inst {
            DecInst::Add(a) => ProcessInst::Add(a),
            DecInst::Copy(c) => ProcessInst::Copy{copy:c, addr:CopyAddr::Unresolved},
            DecInst::Run(r) => ProcessInst::Run(r),
        }
    }
    fn has_copy(self)->InstCopy{
        match self{
            Self::Copy { copy, addr  } => {
                //always
                InstCopy::IsCopy { copy, addr }
            },
            Self::Sequence(ProcSequence { inst, len, skip, trunc }) => {
                //sometimes
                if inst.iter().any(|x|matches!(x,Self::Copy{..})) {
                    InstCopy::InSequence(ProcSequence { inst, len, skip, trunc })
                }else{
                    InstCopy::No(Self::Sequence(ProcSequence { inst, len, skip, trunc }))
                }
            },
            a => InstCopy::No(a)
        }
    }
    fn len_in_t(&self, inst_u_start:u32) -> u32 {
        match self {
            Self::Add(inst) => {inst.len()},
            Self::Run(inst) => {inst.len()},
            Self::Copy { copy:COPY { len, u_pos }, .. } => {
                let u_pos = *u_pos as u32;
                let end = u_pos + *len;
                if end > inst_u_start {
                    end - inst_u_start
                }else{
                    *len
                }
            },
            Self::Sequence(ProcSequence { len, skip, trunc, ..}) => {
                //cannot be zero len, we have a logical error somewhere
                debug_assert!(skip + trunc < *len);
                len - (skip + trunc)
            },
        }

    }
    fn skip(&mut self, amt: u32) {
        if amt == 0 {return}
        match self {

            Self::Sequence(ProcSequence { skip, .. }) => {*skip += amt as u32;},
            Self::Add(inst) => {inst.skip(amt);},
            Self::Run(inst) => {inst.skip(amt);},
            Self::Copy { copy, .. } => {copy.skip(amt);},
        }
    }
    fn trunc(&mut self, amt: u32){
        if amt == 0 {return}
        match self {
            Self::Add(inst) => {inst.trunc(amt);},
            Self::Run(inst) => {inst.trunc(amt);},
            Self::Copy { copy, .. } => {copy.trunc(amt);},
            Self::Sequence(ProcSequence {trunc ,..}) => {*trunc += amt as u32;},
        }
    }
}
enum InstCopy{
    No(ProcessInst),
    InSequence(ProcSequence),
    IsCopy{ copy:COPY, addr: CopyAddr  }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseInst{
    ///The position in the target output stream where this instruction starts outputting bytes
    pub o_start: u64,
    pub inst: Inst,
}

pub(crate) fn merge_ranges<T: Ord + Copy>(ranges: Vec<Range<T>>) -> Vec<Range<T>> {
    let mut result: Vec<Range<T>> = Vec::new();
    let mut sorted_ranges = ranges;

    // 1. Sort the ranges by their start values
    sorted_ranges.sort_by(|a, b| a.start.cmp(&b.start));

    // 2. Iterate through the sorted ranges
    for range in sorted_ranges {
        // 3. If the result list is empty or the current range doesn't overlap
        //    with the last range in the result, simply add it to the result.
        if result.is_empty() || range.start > result.last().unwrap().end {
            result.push(range);
        } else {
            // 4. Overlap exists: Extend the end of the last range in the result
            //    to the maximum end of the overlapping ranges.
            let last_index = result.len() - 1;
            result[last_index].end = std::cmp::max(result[last_index].end, range.end);
        }
    }

    result
}
pub(crate) fn range_overlap<T: Ord + Copy>(range1: &Range<T>, range2: &Range<T>) -> Option<Range<T>> {
    let start = std::cmp::max(range1.start, range2.start);
    let end = std::cmp::min(range1.end, range2.end);

    if start < end {
        Some(Range { start, end })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::reader::VCDReader;

    use super::*;
    use std::io::Cursor;
    #[test]
    fn test_kitchen_sink_translate_detail(){
        // "hello" -> "Hello! Hello! Hell..."
        //from encoder tests
        let patch = vec![
            214,195,196,0, //magic
            0, //hdr_indicator
            0, //win_indicator Neither
            7, //delta window size
            1, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            1, //length of instructions and size
            0, //length of addr
            72, //data section 'H (i=12)
            2, //ADD1
            1, //win_indicator VCD_SOURCE
            4, //SSS
            1, //SSP
            8, //delta window size
            5, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            1, //length of instructions and size
            1, //length of addr
            33, //data section '!' i=23
            253, //COPY4_mode5 ADD1
            0, //addr 0
            2, //win_indicator VCD_TARGET
            6, //SSS
            0, //SSP
            9, //delta window size
            7, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            2, //length of instructions and size
            1, //length of addr
            32, //data section ' ' (i=35)
            2, //ADD1 NOOP
            118, //COPY6_mode6 NOOP
            0, //addr 0
            2, //win_indicator VCD_TARGET
            5, //SSS
            6, //SSP
            12, //delta window size
            8, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            4, //length of instructions and size
            2, //length of addr
            46, //data section '.' (i=48)
            117, //ADD1 COPY5_mode6
            2, //Add1 NOOP
            35, //COPY0_mode1
            3, //...size
            0, //addr 0
            1, //addr 1
        ];
        let patch = Cursor::new(patch);
        let reader = VCDReader::new(patch).unwrap();
        let mut translator = VCDTranslator::new(VCDDecoder::new(reader)).unwrap();
        let msg = translator.interrogate(0).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 12 })); //H
        let msg = translator.interrogate(1).unwrap().unwrap();
        let same_copy = Inst::Copy (DisCopy{ copy: COPY { len: 4, u_pos: 0 }, sss: 4, ssp: 1 });
        assert_eq!(msg.inst, same_copy); //ello

        let msg = translator.interrogate(2).unwrap().unwrap();
        assert_eq!(msg.inst, same_copy); //ello
        let msg = translator.interrogate(3).unwrap().unwrap();
        assert_eq!(msg.inst, same_copy); //ello
        let msg = translator.interrogate(4).unwrap().unwrap();
        assert_eq!(msg.inst, same_copy); //ello

        let msg = translator.interrogate(5).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 23 })); // '!'
        let msg = translator.interrogate(6).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 35 })); // ' '
        let msg = translator.interrogate(7).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 12 })); //H
        let msg = translator.interrogate(8).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Copy (DisCopy{ copy: COPY { len: 4, u_pos: 0 }, sss: 4, ssp: 1 }));//ello
        let msg = translator.interrogate(12).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 23 })); // '!'
        let msg = translator.interrogate(13).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 35 })); // ' '
        let msg = translator.interrogate(14).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 12 })); //H
        let msg = translator.interrogate(15).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Copy (DisCopy{ copy: COPY { len: 3, u_pos: 0 }, sss: 4, ssp: 1 })); //ell
        let msg = translator.interrogate(18).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 48 }));
        let msg = translator.interrogate(19).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Sequence(Sequence { inst: vec![Inst::Add(ADD { len: 1, p_pos: 48 })], len: 2, skip: 0, trunc: 0 }));

    }
    #[test]
    fn test_seq(){
        // Instructions -> "" -> "tererest'

        let patch = vec![
            214, 195, 196, 0,  //magic
            0,  //hdr_indicator
            0, //win_indicator
            13, //size_of delta window
            8, //size of target window
            0, //delta indicator
            5, //length of data for ADDs and RUNs
            2, //length of instructions and sizes
            1, //length of addresses for COPYs
            116, 101, 114, 115, 116, //data section b"terst"
            200, //ADD size3 & COPY5_mode0
            3, //ADD size 2
            1, //addr for copy
        ];
        let patch = Cursor::new(patch);
        let reader = VCDReader::new(patch).unwrap();
        let mut translator = VCDTranslator::new(VCDDecoder::new(reader)).unwrap();
        let msg = translator.interrogate(0).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 3, p_pos: 12 }));
        let msg = translator.interrogate(3).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Sequence(Sequence { inst: vec![Inst::Add(ADD { len: 2, p_pos: 13 })], len: 3, skip: 0, trunc: 0 }));
        let msg = translator.interrogate(6).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 2, p_pos: 15 }));
        let msg = translator.interrogate(8).unwrap();
        assert_eq!(msg, None);
    }
    #[test]
    fn test_consecutive_ranges() {
        let ranges = vec![
            Range { start: 1, end: 3 },
            Range { start: 3, end: 5 },
        ];
        let expected = vec![Range { start: 1, end: 5 }];
        assert_eq!(merge_ranges(ranges), expected);
    }

    #[test]
    fn test_overlapping_ranges() {
        let ranges = vec![
            Range { start: 5, end: 10 },
            Range { start: 3, end: 8 },
            Range { start: 10, end: 12 },
        ];
        let expected = vec![Range { start: 3, end: 12 }];
        assert_eq!(merge_ranges(ranges), expected);
    }

    #[test]
    fn test_nested_ranges() {
        let ranges = vec![
            Range { start: 1, end: 10 },
            Range { start: 2, end: 5 },
        ];
        let expected = vec![Range { start: 1, end: 10 }];
        assert_eq!(merge_ranges(ranges), expected);
    }

    #[test]
    fn test_disjoint_ranges() {
        let ranges = vec![
            Range { start: 1, end: 3 },
            Range { start: 5, end: 7 },
            Range { start: 10, end: 12 },
        ];
        let expected = ranges.clone(); // No merging happens
        assert_eq!(merge_ranges(ranges), expected);
    }

    #[test]
    fn test_all(){
        let ranges = vec![
            Range { start: 5, end: 10 },
            Range { start: 3, end: 8 },
            Range { start: 9, end: 12 },
            Range { start: 1, end: 10 },
            Range { start: 2, end: 5 },
            Range { start: 1, end: 3 },
            Range { start: 5, end: 7 },
            Range { start: 10, end: 12 },
            Range { start: 15, end: 30 },
        ];
        let expected = vec![Range { start: 1, end: 12 },Range { start: 15, end: 30 }];
        assert_eq!(merge_ranges(ranges), expected);
    }

    #[test]
    fn test_empty_input() {
        let ranges: Vec<Range<u64>> = vec![];
        let expected: Vec<Range<u64>> = vec![];
        assert_eq!(merge_ranges(ranges), expected);
    }
}
