use core::panic;
use std::{io::{Read, Seek}, ops::Range};

use crate::{decoder::{DecInst, VCDDecoder, VCDiffDecodeMsg}, reader::{read_header, read_window_header, WindowSummary}, ADD, COPY, RUN};



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
/*
The translator must process the file sequentially to find any implicit sequences.
We store all instructions so that we can resolve Copys in T or TargetSourcedWindows (or both).
An Copy or sequence found in a Sparse instruction is already normalized and can be merged.
In theory, we only translate the 'earlier' patch file.
The 'later' patch, we are simply trying to 'fill in' COPYs with a src window reference.
If it is in T, we just need to resolve copys in S.

The main goal would be to get this refactored to be more efficient.

For starters, we need to more efficiently find our run of inst for a given slice
Binary search in the vec might not be the fastest, maybe btree?
*/

///This orchestrates a single patch file. For Merging purposes.
pub struct VCDTranslator<R>{
    ///The reader for the VCDIFF file
    decoder: VCDDecoder<R>,
    windows: Vec<WindowSummary>,
    win_index: usize,
    ///Completed Windows are stored here
    src_data: Vec<SparseInst>,
    ///This is from completed windows in src_data
    cur_o_len: u64,
    ///In src_data, where is the first inst for this window
    cur_win_start_idx: usize,
    cur_u_pos: usize,
}

impl<R:Read+Seek> VCDTranslator<R> {
    pub fn new(mut decoder: VCDDecoder<R>) -> std::io::Result<Self> {
        let windows = gather_summaries(&mut decoder.reader().get_reader(0)?)?;

        Ok(VCDTranslator {
            decoder,
            cur_u_pos: windows[0].source_segment_size.map(|x|x as usize).unwrap_or(0),
            windows,
            win_index: 0,
            src_data: Vec::new(),
            cur_win_start_idx: 0,
            cur_o_len: 0,
        })
    }
    pub fn cur_win_start_o_pos(&self)->u64{
        self.cur_o_len
    }
    pub fn into_inner(self)->VCDDecoder<R>{
        self.decoder
    }
    pub fn get_reader(&mut self, at_from_start:u64)->std::io::Result<&mut R>{
        self.decoder.reader().get_reader(at_from_start)
    }
    pub fn cur_win(&self)->Option<&WindowSummary>{
        self.windows.get(self.win_index)
    }
    pub fn cur_t_start(&self)->usize{
        self.cur_win().unwrap().source_segment_size.map(|x|x as usize).unwrap_or(0)
    }
    fn load_up_to(&mut self, o_position:u64)->std::io::Result<()>{
        while self.cur_o_len <= o_position {//will fall through if it is a prev inst.
            let count = self.next_op()?;
            if count == 0 {
                break;
            }
        }
        Ok(())
    }
    pub fn exact_slice(&mut self, o_position: u64, len: u32) -> std::io::Result<Vec<Inst>> {
        self.load_up_to(o_position + len as u64)?;
        Ok(self.retrieve_exact_slice_from_src_data(o_position, len).into_iter().map(|x|x.inst).collect())
    }
    fn controlling_inst_idx(&self,o_position: u64)->Option<usize>{
        let inst = self.src_data.binary_search_by(|probe|{
            let end = probe.o_start + probe.inst.len() as u64;
            if (probe.o_start..end).contains(&o_position){
                return std::cmp::Ordering::Equal
            }else if probe.o_start > o_position {
                return std::cmp::Ordering::Greater
            }else{
                return std::cmp::Ordering::Less
            }
        });
        if let Ok(idx) = inst {
            Some(idx)
        }else {
            None
        }
    }
    pub fn interrogate(&mut self, o_position: u64) -> std::io::Result<Option<SparseInst>> {
        self.load_up_to(o_position)?;
        if let Some(idx) = self.controlling_inst_idx(o_position){
            return Ok(Some(self.src_data[idx].clone()));
        }else {
            return Ok(None)
        }

    }
    fn set_new_window_state(&mut self,ws:&WindowSummary){
        self.cur_win_start_idx = self.src_data.len();
        self.cur_u_pos = ws.source_segment_size.map(|x|x as usize).unwrap_or(0);
    }
    ///This returns a 'root' instruction that controls the byte at the given position in the output stream.
    ///This operates one op code at a time (one or two instructions).
    ///Returns the number of instructions that were advanced.
    ///0 is returned if there are no more instructions to advance (end of file).
    ///This can return more than 2, since we might need to resolve a TrgtSourcedWindow or a COPY in T (or Both).
    fn next_op(&mut self)->std::io::Result<usize>{
        //a form of double entry accounting to make sure all the translations are correct.
        //assert!(self.cur_o_pos == self.decoder.position(), "Decoder and Merge Handle are out of sync! {} != {}", self.cur_o_pos, self.decoder.position());
        loop {
            match self.decoder.next()?{
                VCDiffDecodeMsg::WindowSummary(ws) => {
                    self.set_new_window_state(&ws);
                },
                VCDiffDecodeMsg::Inst { u_start ,first, second } => {
                    //here we need to resolve instructions for merging
                    assert!(u_start == self.cur_u_pos as u64, "We are out of sync with the output stream {} != {}", self.cur_u_pos, u_start);
                    let mut count = 0;
                    count += self.resolve_inst(first);
                    if let Some(second) = second {
                        count += self.resolve_inst(second);
                    }
                    return Ok(count);
                },
                VCDiffDecodeMsg::EndOfWindow => {
                    self.win_index += 1;
                    continue;
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
            self.store_inst(ProcessInst::from_dec(inst,false));
            1
        }
    }

    fn resolve_copy(&mut self, mut copy: COPY) -> usize {
        let mut buf = Vec::with_capacity(15);
        let cur_win = self.cur_win().unwrap();
        let mut is_seq = None;
        let vcd_trgt = cur_win.is_vcd_target();
        let mut state = if vcd_trgt { CopyState::TranslateGlobal } else { CopyState::Resolved };
        if copy.u_pos + copy.len <= self.cur_t_start() as u32 { //this cannot be seq
            let sss = cur_win.source_segment_size.unwrap();
            let ssp = cur_win.source_segment_position.unwrap();
            buf.push(ProcessInst::Copy { copy, addr: CopyAddr::InS { sss, ssp },vcd_trgt });
        }else{
            //the next might have seq, we are in s+t
            let cur_u = self.cur_u_pos as u32;
            if copy.u_pos + copy.len > cur_u{
                let pattern = (cur_u - copy.u_pos) as usize;
                let len_in_t = copy.u_pos + copy.len - cur_u;
                is_seq = Some(len_in_t);
                let trunc = copy.len - pattern as u32;
                copy.trunc(trunc)
            }
            if copy.u_pos + copy.len <= self.cur_t_start() as u32 {
                let sss = cur_win.source_segment_size.unwrap();
                let ssp = cur_win.source_segment_position.unwrap();
                buf.push(ProcessInst::Copy { copy, addr: CopyAddr::InS { sss, ssp },vcd_trgt });
            }else if copy.u_pos < self.cur_t_start() as u32{
                //This case is technically disallowed by the spec
                let sss = cur_win.source_segment_size.unwrap();
                let ssp = cur_win.source_segment_position.unwrap();

                let mut in_s = copy.clone();
                let len = copy.len as u32;
                let trunc = copy.u_pos + copy.len - sss as u32;
                in_s.trunc(trunc);
                buf.push(ProcessInst::Copy { copy: in_s, addr: CopyAddr::InS { sss, ssp },vcd_trgt });
                let skip = len - trunc;
                copy.skip(skip);
                buf.push(ProcessInst::Copy { copy, addr: CopyAddr::InT ,vcd_trgt});
                state.back(vcd_trgt)
            }else{
                buf.push(ProcessInst::Copy { copy, addr: CopyAddr::InT,vcd_trgt });
                state.back(vcd_trgt)
            }
        }
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
        if let Some(len) = is_seq {
            self.store_inst(ProcessInst::Sequence(ProcSequence { inst: buf, len: len as u32, skip: 0, trunc: 0}));
        }else{
            self.src_data.reserve(count);
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
            match inst.has_copy(false){
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
                InstCopy::IsCopy { copy:COPY { len, u_pos }, addr,vcd_trgt } => {
                    if matches!(&addr, CopyAddr::InS {..}) {
                        //we are doing a local translation, and it is already in S
                        //we do nothing. It might still be in target, but we will handle that later
                        output_buffer.push(ProcessInst::Copy { copy:COPY { len, u_pos }, addr,vcd_trgt });
                        continue;
                    }
                    //here u_pos should partly be in T of U
                    debug_assert!(u_pos < self.cur_u_pos as u32, "The COPY position is not before our current output position in U");
                    //local_slice will only handle COPY entirely in T
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
        let src_size = self.cur_t_start() as u32;
        let mut pos = src_size;
        let slice_end = u_pos + len;
        let range = u_pos..slice_end;
        for SparseInst { inst, .. } in self.src_data[self.cur_win_start_idx..].iter() {
            let len = inst.len();//len_in_t;//inst.len_in_t(src_size+pos);
            let cur_inst_end = pos + len;
            if let Some(_) = range_overlap(&(pos..cur_inst_end), &range) {
                let mut cur_inst = inst.as_proc_inst(false);
                if u_pos >= pos {
                    let skip = u_pos - pos;
                    cur_inst.skip(skip);
                }
                if cur_inst_end >= slice_end {
                    let trunc = cur_inst_end - slice_end;
                    cur_inst.trunc(trunc);
                }
                debug_assert!(cur_inst.len() > 0);
                slice.push(cur_inst);
            }
            if cur_inst_end >= slice_end {
                break;
            }
            pos += len;
        }
        slice
    }
    fn retrieve_exact_slice_from_src_data(&self,o_position: u64,len: u32)->Vec<SparseInst>{
        let mut last = 0;
        let mut slice = Vec::new();
        let end_pos = o_position + len as u64;
        let start_inst_idx = self.controlling_inst_idx(o_position).unwrap();
        for SparseInst { o_start, inst } in self.src_data[start_inst_idx..].iter() {
            let inst_len = inst.len();
            let cur_inst_end = o_start + inst_len as u64;
            debug_assert!(if !slice.is_empty(){last == *o_start}else{true});
            let mut cur_inst = inst.clone();

            if o_position > *o_start {
                let skip = o_position - *o_start;
                cur_inst.skip(skip as u32);
            }
            if end_pos < cur_inst_end {
                let trunc = cur_inst_end - end_pos;
                cur_inst.trunc(trunc as u32);
            }
            debug_assert!(cur_inst.len() > 0, "The instruction length is zero");
            slice.push(SparseInst { o_start: *o_start, inst: cur_inst });

            last = cur_inst_end;
            if cur_inst_end >= end_pos {
                break;
            }
        }
        debug_assert!(last >= end_pos, "The last instruction end is before the requested end position");
        slice
    }
    // This handles cross-window translation using previously resolved instructions
    fn resolve_global(&mut self,input_buffer:&mut Vec<ProcessInst>){
        //the buffer here only contains instructions in S or they are ADD/RUNs

        //we need to translate COPYs since our cur_win is TargetSourced.
        let mut to_drain = Vec::new();
        to_drain.reserve(input_buffer.capacity() + 2);
        //to_drain now has input, and input is now our 'output'
        std::mem::swap(input_buffer, &mut to_drain);
        //change name for sanity
        let output_buffer = input_buffer;

        for inst in to_drain.drain(..){
            match inst.has_copy(true){
                InstCopy::No(a) => output_buffer.push(a),
                InstCopy::InSequence(mut s) => {
                    //recursively resolve the inner instructions
                    //this happens when we have had an earlier implicit sequence
                    self.resolve_global(&mut s.inst);
                    output_buffer.push(ProcessInst::Sequence(s));
                },
                InstCopy::IsCopy { copy:COPY { len, u_pos }, addr, .. } => {
                    //here u_pos should already be in S (resolve local should be ran first)
                    //S references our Output stream
                    match addr {
                        CopyAddr::InS {  ssp,.. } => {
                            let o_pos = ssp + u_pos as u64;
                            let slice = self.retrieve_exact_slice_from_src_data(o_pos, len);
                            output_buffer.extend(slice.into_iter().map(|x|x.inst.as_proc_inst(false)));
                        },
                        _ => panic!("The COPY address should always be translated locally in a TrgtSourcedWindow"),

                    }
                },
            }
        }
    }

    fn store_inst(&mut self,inst:ProcessInst){
        let len = inst.len();
        let new_o_pos = self.cur_o_len + len as u64;
        self.src_data.push(inst.to_sparse_inst(self.cur_o_len));
        self.cur_u_pos += len as usize;
        self.cur_o_len = new_o_pos;
    }
}


#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct ImplicitSeq{
    pattern:usize,
    len:usize,
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
    fn back(&mut self, in_trgt:bool){
        if in_trgt{
            match self{
                CopyState::Resolved => *self = CopyState::TranslateGlobal,
                CopyState::TranslateLocal => *self = CopyState::TranslateLocalThenGlobal,
                CopyState::TranslateGlobal => *self = CopyState::TranslateLocalThenGlobal,
                _ => panic!("Invalid state transition"),
            }

        }else{
            match self{
                CopyState::Resolved => *self = CopyState::TranslateLocal,
                _ => panic!("Invalid state transition"),
            }
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
    fn to_proc_seq(&self,vcd_trgt:bool) -> ProcSequence {
        ProcSequence {
            inst: self.inst.iter().map(|x|x.as_proc_inst(vcd_trgt)).collect(),
            len: self.len,
            skip: self.skip,
            trunc: self.trunc,
        }
    }
    // pub(crate) fn skip(&mut self, amt: u32) {
    //     self.skip += amt as u32;
    // }
    // pub(crate) fn len(&self) -> u32 {
    //     //cannot be zero len, we have a logical error somewhere
    //     debug_assert!(self.skip + self.trunc < self.len);
    //     self.len - (self.skip + self.trunc)
    // }
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
    fn as_proc_inst(&self,vcd_trgt:bool) -> ProcessInst {
        match self {
            Inst::Add(inst) => ProcessInst::Add(inst.clone()),
            Inst::Run(inst) => ProcessInst::Run(inst.clone()),
            Inst::Copy (DisCopy{ copy, sss, ssp }) => ProcessInst::Copy { copy: copy.clone(), addr: CopyAddr::InS { sss: *sss, ssp: *ssp },vcd_trgt },
            Inst::Sequence(seq) => ProcessInst::Sequence(seq.to_proc_seq(vcd_trgt)),
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
    fn skip(&mut self, amt: u32) {
        if amt == 0 {return}
        match self {

            Self::Sequence(Sequence { skip, .. }) => {*skip += amt as u32;},
            Self::Add(inst) => {inst.skip(amt);},
            Self::Run(inst) => {inst.skip(amt);},
            Self::Copy (copy) => {copy.copy.skip(amt);},
        }
    }
    fn trunc(&mut self, amt: u32){
        if amt == 0 {return}
        match self {
            Self::Add(inst) => {inst.trunc(amt);},
            Self::Run(inst) => {inst.trunc(amt);},
            Self::Copy(copy) => {copy.copy.trunc(amt);},
            Self::Sequence(Sequence {trunc ,..}) => {*trunc += amt as u32;},
        }
    }

}

///Sequence is explicit so its length is always known.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ProcessInst{
    ///This is a source instruction
    Add(ADD),
    Run(RUN),
    Copy{copy:COPY, addr:CopyAddr, vcd_trgt:bool},
    Sequence(ProcSequence),
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CopyAddr{
    InT,
    InS{sss:u64, ssp:u64},
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
            ProcessInst::Copy { copy, addr:CopyAddr::InS{ sss, ssp },.. } => Inst::Copy(DisCopy { copy: copy.clone(), sss:*sss, ssp:*ssp }),
            ProcessInst::Sequence(seq) => Inst::Sequence(seq.to_seq()),
            ProcessInst::Copy { .. } => panic!("Cannot convert unresolved COPY to Inst"),
        }
    }
    fn from_dec(inst:DecInst,vcd_trgt:bool)->Self{
        match inst {
            DecInst::Add(a) => ProcessInst::Add(a),
            DecInst::Copy(c) => ProcessInst::Copy{copy:c, addr:CopyAddr::InT,vcd_trgt},
            DecInst::Run(r) => ProcessInst::Run(r),
        }
    }
    fn has_copy(self,global_translate:bool)->InstCopy{
        match self{
            Self::Copy { copy, addr, vcd_trgt } => {
                if global_translate && !vcd_trgt {
                    InstCopy::No(Self::Copy { copy, addr, vcd_trgt })
                }else{
                    InstCopy::IsCopy { copy, addr, vcd_trgt }
                }
            },
            Self::Sequence(ProcSequence { inst, len, skip, trunc }) => {
                //sometimes
                if inst.iter().any(|x|matches!(x,Self::Copy{vcd_trgt,..} if global_translate && !vcd_trgt)) {
                    InstCopy::InSequence(ProcSequence { inst, len, skip, trunc })
                }else{
                    InstCopy::No(Self::Sequence(ProcSequence { inst, len, skip, trunc }))
                }
            },
            a => InstCopy::No(a)
        }
    }
    fn len(&self)->u32{
        match self {
            Self::Add(inst) => {inst.len()},
            Self::Run(inst) => {inst.len()},
            Self::Copy { copy, .. } => {copy.len},
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
    IsCopy{ copy:COPY, addr: CopyAddr, vcd_trgt:bool }
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
    fn test_merge_overlapping_ranges() {
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

    #[test]
    fn test_overlapping_ranges() {
        // Fully overlapping
        let range1 = 1..10;
        let range2 = 5..8;
        assert_eq!(range_overlap(&range1, &range2), Some(5..8));

        // Partially overlapping (start point)
        let range1 = 0..6;
        let range2 = 5..10;
        assert_eq!(range_overlap(&range1, &range2), Some(5..6));

        // Partially overlapping (end point)
        let range1 = 5..10;
        let range2 = 3..10;
        assert_eq!(range_overlap(&range1, &range2), Some(5..10));
    }

    #[test]
    fn test_non_overlapping_ranges() {
        // Adjacent ranges
        let range1 = 1..5;
        let range2 = 5..10;
        assert_eq!(range_overlap(&range1, &range2), None);

        // Disjoint ranges
        let range1 = 1..5;
        let range2 = 10..15;
        assert_eq!(range_overlap(&range1, &range2), None);
    }

    #[test]
    fn test_edge_cases() {
        // Identical ranges
        let range1 = 3..7;
        let range2 = 3..7;
        assert_eq!(range_overlap(&range1, &range2), Some(3..7));

    }

    use crate::encoder::{VCDEncoder, EncInst,WindowHeader};
    use crate::reader::{Header,WinIndicator, DeltaIndicator};
    const HDR:Header = Header { hdr_indicator: 0, secondary_compressor_id: None, code_table_data: None };

    fn make_translator(patch_bytes:Vec<u8>)->VCDTranslator<Cursor<Vec<u8>>>{
        VCDTranslator::new(VCDDecoder::new(VCDReader::new(Cursor::new(patch_bytes)).unwrap())).unwrap()
    }
    #[test]
    fn test_kitchen_sink(){
        //"hello world!" -> "Hello! Hello! Hello. hello. hello..."
        //we need to use a series of VCD_TARGET windows and Sequences across multiple patches
        //we should use copy/seq excessively since add/run is simple in the code paths.


        let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(11), source_segment_position: Some(1), size_of_the_target_window:7 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::ADD("H".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 4, u_pos: 0 })).unwrap(); //ello
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 10 })).unwrap(); //'!'
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 4 })).unwrap(); //' '
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(7), source_segment_position: Some(0), size_of_the_target_window:14 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 19, u_pos: 0 })).unwrap(); //Hello! Hello! Hello
        encoder.next_inst(EncInst::ADD(".".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 13 })).unwrap(); // ' ' idx =20
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(6), source_segment_position: Some(15), size_of_the_target_window:7, delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::ADD("h".as_bytes().to_vec())).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 6, u_pos: 0 })).unwrap(); //'ello. '
        encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(6), source_segment_position: Some(21), size_of_the_target_window:8 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(EncInst::COPY(COPY { len: 6, u_pos: 0 })).unwrap(); //'hello.'
        encoder.next_inst(EncInst::COPY(COPY { len: 3, u_pos: 11 })).unwrap(); //Seq '.' == Run(3) '.'
        let p2 = encoder.finish().unwrap().into_inner();
        let mut translator = make_translator(p2);
        let msg = translator.interrogate(0).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 14 })); //H
        let msg = translator.interrogate(1).unwrap().unwrap();
        let same_copy = Inst::Copy (DisCopy{ copy: COPY { len: 4, u_pos: 0 }, sss: 11, ssp: 1 });
        assert_eq!(msg.inst, same_copy); //ello

        let msg = translator.interrogate(5).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Copy (DisCopy{ copy: COPY { len: 1, u_pos: 10 }, sss: 11, ssp: 1 })); // '!'
        let msg = translator.interrogate(6).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Copy (DisCopy{ copy: COPY { len: 1, u_pos: 4 }, sss: 11, ssp: 1 })); // ' '
        let msg = translator.interrogate(7).unwrap().unwrap();
        let big_seq = Inst::Sequence(Sequence {
            inst: vec![
                Inst::Add(ADD { len: 1, p_pos: 14 }), // H
                Inst::Copy(DisCopy{copy: COPY { len: 4, u_pos: 0 }, sss: 11, ssp: 1 } ),// ello
                Inst::Copy(DisCopy{copy: COPY { len: 1, u_pos: 10 }, sss: 11, ssp: 1 } ),// !
                Inst::Copy(DisCopy{copy: COPY { len: 1, u_pos: 4 }, sss: 11, ssp: 1 } ),// ' '
            ],
            len: 12, skip: 0, trunc: 0
        });
        assert_eq!(msg.inst, big_seq); //Hello! Hello
        let msg = translator.interrogate(19).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 32 }));// .
        let msg = translator.interrogate(20).unwrap().unwrap();
        let big_seq_space = Inst::Sequence(Sequence {
            inst: vec![
                Inst::Add(ADD { len: 1, p_pos: 14 }), // H
                Inst::Copy(DisCopy{copy: COPY { len: 4, u_pos: 0 }, sss: 11, ssp: 1 } ),// ello
                Inst::Copy(DisCopy{copy: COPY { len: 1, u_pos: 10 }, sss: 11, ssp: 1 } ),// !
                Inst::Copy(DisCopy{copy: COPY { len: 1, u_pos: 4 }, sss: 11, ssp: 1 } ),// ' '
            ],
            len: 12, skip: 6, trunc: 5 // ' '
        });
        assert_eq!(msg.inst, big_seq_space); // ' '
        let msg = translator.interrogate(21).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 49 })); // h
        let big_seq_ello = Inst::Sequence(Sequence {
            inst: vec![
                Inst::Add(ADD { len: 1, p_pos: 14 }), // H
                Inst::Copy(DisCopy{copy: COPY { len: 4, u_pos: 0 }, sss: 11, ssp: 1 } ),// ello
                Inst::Copy(DisCopy{copy: COPY { len: 1, u_pos: 10 }, sss: 11, ssp: 1 } ),// !
                Inst::Copy(DisCopy{copy: COPY { len: 1, u_pos: 4 }, sss: 11, ssp: 1 } ),// ' '
            ],
            len: 12, skip: 8, trunc: 0 // 'ello'
        });
        let msg = translator.interrogate(22).unwrap().unwrap();
        assert_eq!(msg.inst, big_seq_ello); // 'ello'
        let msg = translator.interrogate(26).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 32 })); //.
        let msg = translator.interrogate(27).unwrap().unwrap();
        assert_eq!(msg.inst, big_seq_space); // ' '
        let msg = translator.interrogate(28).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 49 })); // h
        let msg = translator.interrogate(29).unwrap().unwrap();
        assert_eq!(msg.inst, big_seq_ello); // 'ello'
        let msg = translator.interrogate(33).unwrap().unwrap();
        assert_eq!(msg.inst, Inst::Add(ADD { len: 1, p_pos: 32 })); //.
        let msg = translator.interrogate(34).unwrap().unwrap();
        let seq_dot = Inst::Sequence(Sequence {
            inst: vec![
                Inst::Add(ADD { len: 1, p_pos: 32 })
            ],
            len: 2, skip: 0, trunc: 0 // '..'
        });
        assert_eq!(msg.inst, seq_dot); //..

        let msg = translator.interrogate(11).unwrap().unwrap();
        let big_seq = Inst::Sequence(Sequence {
            inst: vec![
                Inst::Add(ADD { len: 1, p_pos: 14 }), // H
                Inst::Copy(DisCopy{copy: COPY { len: 4, u_pos: 0 }, sss: 11, ssp: 1 } ),// ello
                Inst::Copy(DisCopy{copy: COPY { len: 1, u_pos: 10 }, sss: 11, ssp: 1 } ),// !
                Inst::Copy(DisCopy{copy: COPY { len: 1, u_pos: 4 }, sss: 11, ssp: 1 } ),// ' '
            ],
            len: 12, skip: 0, trunc: 0
        });
        assert_eq!(msg.inst, big_seq); //Hello! Hello



    }
}
