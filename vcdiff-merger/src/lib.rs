

use std::{fmt::Debug, io::{Read, Seek, Write}, num::NonZeroU32, ops::Range};

use vcdiff_common::{CopyType, Header, Inst, InstType, Instruction, WinIndicator, ADD, COPY, RUN};
use vcdiff_reader::{VCDReader, VCDiffReadMsg};
use vcdiff_writer::{VCDWriter, WriteInst, WriteWindowHeader};


///Disassociated Copy (from the window it was found in).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DCopy{
    pub u_pos:u32,
    pub len_u:u32,
    pub sss:u32,
    pub ssp:u64,
    pub vcd_trgt:bool,
    pub copy_type:CopyType,
}
impl MergeInst for DCopy{
    fn skip(&mut self,amt:u32){
        self.u_pos += amt;
        self.len_u -= amt;
        match self.copy_type{
            CopyType::CopyQ{..} => {
                panic!("CopyQ should not be skipped!");
            },
            _ => {}
        }
    }
    fn trunc(&mut self,amt:u32){
        self.len_u = self.len_u - amt;
        match self.copy_type{
            CopyType::CopyQ{..} => {
                panic!("CopyQ should not be truncated!");
            },
            _ => {}
        }
    }
    fn src_range(&self)->Option<Range<u64>>{
        let new_ssp = self.ssp() + self.u_start_pos() as u64;
        let new_end = new_ssp + self.len_in_u() as u64;
        debug_assert!(new_end <= self.ssp() + self.sss() as u64,
            "new_end:{} ssp:{} sss:{}",new_end,self.ssp(),self.sss() as u64
        );
        Some(new_ssp..new_end)
    }

}
impl Instruction for DCopy{
    fn len_in_u(&self)->u32{
        self.len_u
    }

    fn inst_type(&self)->InstType {
        InstType::Copy(self.copy_type)
    }

}
impl DCopy{
    fn from_copy(copy:COPY,ssp:u64,sss:u32,vcd_trgt:bool)->Self{
        let COPY { len, u_pos, copy_type } = copy;
        DCopy{
            u_pos,
            len_u:len,
            sss,
            ssp,
            vcd_trgt,
            copy_type,
        }
    }
    fn u_start_pos(&self)->u32{
        self.u_pos
    }
    fn ssp(&self)->u64{
        self.ssp
    }
    fn sss(&self)->u32{
        self.sss
    }
    fn vcd_trgt(&self)->bool{
        self.vcd_trgt
    }
}

///Extracted Add instruction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExAdd{
    pub bytes:Vec<u8>,
}
impl Instruction for ExAdd{
    fn len_in_u(&self)->u32{
        self.bytes.len() as u32
    }
    fn inst_type(&self)->InstType {
        InstType::Add
    }
}
impl MergeInst for ExAdd{
    fn skip(&mut self,amt:u32){
        self.bytes = self.bytes.split_off(amt as usize);
    }
    fn trunc(&mut self,amt:u32){
        self.bytes.truncate(self.bytes.len() - amt as usize);
    }
    fn src_range(&self)->Option<Range<u64>> {
        None
    }
}
impl MergeInst for RUN{
    fn skip(&mut self,amt:u32){
        self.len -= amt;
    }
    fn trunc(&mut self,amt:u32){
        self.len = self.len - amt;
    }
    fn src_range(&self)->Option<Range<u64>> {
        None
    }
}

///Disassociated Instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DInst{
    Add(ExAdd),
    Run(RUN),
    Copy(DCopy),
}
impl Instruction for DInst{
    fn len_in_u(&self)->u32{
        match self{
            DInst::Add(bytes) => bytes.len_in_u(),
            DInst::Run(run) => run.len,
            DInst::Copy(copy) => copy.len_in_u(),
        }
    }
    fn inst_type(&self)->InstType{
        match self{
            DInst::Add(_) => InstType::Add,
            DInst::Run(_) => InstType::Run,
            DInst::Copy(copy) => copy.inst_type(),
        }
    }
}
impl MergeInst for DInst{
    fn skip(&mut self,amt:u32){
        match self{
            DInst::Add(bytes) => bytes.skip(amt),
            DInst::Run(run) => run.skip(amt),
            DInst::Copy(copy) => copy.skip(amt),
        }
    }
    fn trunc(&mut self,amt:u32){
        match self{
            DInst::Add(bytes) => bytes.trunc(amt),
            DInst::Run(run) => run.trunc(amt),
            DInst::Copy(copy) => copy.trunc(amt),
        }
    }
    fn src_range(&self)->Option<Range<u64>>{
        match self{
            DInst::Add(_) => None,
            DInst::Run(_) => None,
            DInst::Copy(copy) => copy.src_range(),
        }
    }
}
impl DInst {
    pub fn take_copy(self)->Option<DCopy>{
        match self{
            DInst::Copy(copy) => Some(copy),
            _ => None,
        }
    }
    pub fn vcd_trgt(&self)->bool{
        match self{
            DInst::Add(_) => false,
            DInst::Run(_) => false,
            DInst::Copy(copy) => copy.vcd_trgt(),
        }
    }
}


///Extracted Instruction with a starting position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseInst{
    pub o_pos_start:u64,
    pub inst:DInst,
}
impl Instruction for SparseInst{
    fn len_in_u(&self)->u32{
        self.inst.len_in_u()
    }
    fn inst_type(&self)->InstType{
        self.inst.inst_type()
    }
}
impl MergeInst for SparseInst{
    fn skip(&mut self,amt:u32){
        self.o_pos_start += amt as u64;
        self.inst.skip(amt);
    }
    fn trunc(&mut self,amt:u32){
        self.inst.trunc(amt);
    }

    fn src_range(&self)->Option<Range<u64>>{
        self.inst.src_range()
    }
}
impl PosInst for SparseInst{
    fn o_start(&self)->u64{
        self.o_pos_start
    }
}

pub trait PosInst:MergeInst{
    fn o_start(&self)->u64;
}
pub trait MergeInst:Instruction{
    ///Shorten the 'front' of the instruction
    fn skip(&mut self,amt:u32);
    ///Truncate off the 'back' of the instruction
    fn trunc(&mut self,amt:u32);
    ///If this is a Copy, what is the ssp..ssp+sss that would contain exactly this one instruction.
    fn src_range(&self)->Option<Range<u64>>;
    fn will_fit_window(&self,max_space_avail:NonZeroU32)->Option<NonZeroU32>{
        //can we figure out how much to truncate to fit in the space?
        match self.src_range(){
            None => {
                //Add or Run
                let cur_u_len = self.len_in_u();
                if cur_u_len <= max_space_avail.into(){
                    return None;
                }else{
                    Some(max_space_avail)
                }
            },
            Some(min) => {//Copy Math
                //every change in len, also shrinks the sss
                let cur_s_len = min.end - min.start;
                let cur_u_len = self.len_in_u() + cur_s_len as u32;
                if cur_u_len <= max_space_avail.into() {
                    return None;
                }else{
                    let new_len = max_space_avail.get() as u64 - cur_s_len;
                    if new_len == 0{
                        None
                    }else{
                        Some(NonZeroU32::new(new_len as u32).unwrap())
                    }
                }
            }
        }
    }
    fn split_at(mut self,first_inst_len:u32)->(Self,Self){
        assert!(!self.is_implicit_seq());
        let mut second = self.clone();
        self.trunc(self.len_in_u() - first_inst_len);
        second.skip(first_inst_len);
        (self,second)
    }
}


///Determines the next WinIndicator based on the current instruction type, the current WinIndicator, and the instruction's VCD target status.
/// Returns None if the new instruction does not change the current WinIndicator.
/// # Arguments
/// * `inst_type` - The type of the instruction.
/// * `cur_ind` - The current WinIndicator.
/// * `vcd_trgt` - The VCD target status of the instruction.
pub fn comp_indicator(inst_type: &InstType, cur_ind: &WinIndicator, vcd_trgt: bool) -> Option<WinIndicator> {
    match (inst_type, cur_ind, vcd_trgt) {
        (InstType::Copy { .. }, WinIndicator::VCD_SOURCE, true)
        | (InstType::Copy { .. }, WinIndicator::Neither, true) => Some(WinIndicator::VCD_TARGET),
        (InstType::Copy { .. }, WinIndicator::VCD_TARGET, false)
        | (InstType::Copy { .. }, WinIndicator::Neither, false) => Some(WinIndicator::VCD_SOURCE),
        _ => None,
    }
}
///Finds the index of the instruction that controls the given output position.
/// # Arguments
/// * `insts` - The list of instructions to search.
/// * `o_pos` - The output position to find the controlling instruction for.
/// # Returns
/// The index of the controlling instruction, or None if no such instruction exists.
pub fn find_controlling_inst<I:PosInst>(insts:&[I],o_pos:u64)->Option<usize>{
    let inst = insts.binary_search_by(|probe|{
        let end = probe.o_start() + probe.len_in_o() as u64;
        if (probe.o_start()..end).contains(&o_pos){
            return std::cmp::Ordering::Equal
        }else if probe.o_start() > o_pos {
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

///Returns a cloned and clipped subslice of instructions that exactly covers the requested output range.
/// # Arguments
/// * `instructions` - The list of instructions to extract from.
/// * `start` - The output position (output byte offset) to start the slice at.
/// * `len` - The length of the slice in output bytes.
/// # Returns
/// A vector containing the cloned and clipped instructions that exactly cover the requested output range.
/// If the output range is not covered by the instructions, None is returned.
///
/// This does not check that the instructions are sequential.
pub fn get_exact_slice<I:PosInst+Debug>(instructions:&[I],start:u64,len:u32)->Option<Vec<I>>{
    let start_idx = find_controlling_inst(instructions,start)?;
    let end_pos = start + len as u64;
    let mut slice = Vec::new();
    let mut complete = false;

    for inst in instructions[start_idx..].iter() {
        let inst_len = inst.len_in_o();
        let o_start = inst.o_start();
        let cur_inst_end = o_start + inst_len as u64;
        let mut cur_inst = inst.clone();
        if start > o_start {
            let skip = start - o_start;
            cur_inst.skip(skip as u32);
        }
        if end_pos < cur_inst_end {
            let trunc = cur_inst_end - end_pos;
            cur_inst.trunc(trunc as u32);
        }
        debug_assert!(cur_inst.len_in_o() > 0, "The instruction length is zero");
        slice.push(cur_inst);

        if cur_inst_end >= end_pos {
            complete = true;
            //debug_assert!(sum_len_in_o(&slice)==len as u64,"{} != {} start:{} end_pos:{} ... {:?} from {:?}",sum_len_in_o(&slice),len,start,end_pos,&slice,instructions);
            break;
        }
    }
    if !complete {
        return None;
    }
    Some(slice)
}

//Should maybe move this to Reader?
///Stats about the patch file.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats{
    pub add_bytes:usize,
    pub run_bytes:usize,
    pub copy_bytes:usize,
    pub add_cnt:usize,
    pub run_cnt:usize,
    pub copy_s_cnt:usize,
    pub copy_t_cnt:usize,
    pub copy_q_cnt:usize,
    pub output_size:usize,
    pub contains_vcd_target:bool,
}

impl Stats {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn add(&mut self, len:usize){
        self.add_bytes += len;
        self.add_cnt += 1;
        self.output_size += len;
    }
    pub fn run(&mut self, len:usize){
        self.run_bytes += len;
        self.run_cnt += 1;
        self.output_size += len;
    }
    pub fn copy_s(&mut self, len:usize){
        self.copy_bytes += len;
        self.copy_s_cnt += 1;
        self.output_size += len;
    }
    pub fn copy_t(&mut self, len:usize){
        self.copy_bytes += len;
        self.copy_t_cnt += 1;
        self.output_size += len;
    }
    pub fn copy_q(&mut self, len_in_o:usize){
        self.copy_bytes += len_in_o;
        self.copy_q_cnt += 1;
        self.output_size += len_in_o;
    }
    pub fn vcd_trgt(&mut self){
        self.contains_vcd_target = true;
    }
    pub fn has_copy(&self)->bool{
        self.copy_bytes > 0
    }
}

///Extracts all instructions from all windows.
///Memory consumption may be 2-4x the size of the encoded (uncompressed) patch.
pub fn extract_patch_instructions<R:Read + Seek>(patch:R)->std::io::Result<(Vec<SparseInst>, Stats)>{
    let mut insts = Vec::new();
    let mut reader = VCDReader::new(patch)?;
    let mut ssp = None;
    let mut sss = None;
    let mut vcd_trgt = false;
    let mut o_pos_start = 0;
    let mut stats = Stats::new();
    loop{
        match reader.next()?{
            VCDiffReadMsg::WindowSummary(ws) => {
                ssp = ws.source_segment_position;
                sss = ws.source_segment_size;
                if ws.win_indicator == WinIndicator::VCD_TARGET{
                    vcd_trgt = true;
                    stats.vcd_trgt();
                }
            },
            VCDiffReadMsg::Inst { first, second } => {
                for inst in [Some(first), second]{
                    if inst.is_none(){
                        continue;
                    }
                    let inst = inst.unwrap();
                    let len_o = inst.len_in_o() as usize;
                    match inst{
                        Inst::Add(ADD{ len, p_pos }) => {
                            let mut bytes = vec![0; len as usize];
                            reader.read_from_src(p_pos, &mut bytes)?;
                            insts.push(SparseInst{o_pos_start,inst:DInst::Add(ExAdd { bytes })});
                            stats.add(len_o);
                        },
                        Inst::Run(run) => {
                            stats.run(len_o);
                            insts.push(SparseInst{o_pos_start,inst:DInst::Run(run)})
                        },
                        Inst::Copy(copy) =>{
                            let ssp = ssp.expect("SSP not set");
                            let sss = sss.expect("SSS not set");
                            match copy.copy_type{
                                CopyType::CopyQ{..} => {
                                    stats.copy_q(len_o);
                                },
                                CopyType::CopyS => {
                                    stats.copy_s(len_o);
                                },
                                CopyType::CopyT{..} => {
                                    stats.copy_t(len_o);
                                },
                            }
                            insts.push(SparseInst{
                                o_pos_start,
                                inst:DInst::Copy(DCopy::from_copy(copy, ssp, sss as u32, vcd_trgt))
                            });
                        }
                    }
                    o_pos_start += len_o as u64;
                }
            },
            VCDiffReadMsg::EndOfWindow => {
                ssp = None;
                sss = None;
                vcd_trgt = false;
            },
            VCDiffReadMsg::EndOfFile => break,
        }
    }
    Ok((insts,stats))
}

/// This function will dereference all Non-CopySS instructions in the extracted instructions.
pub fn deref_non_copy_ss(extracted:Vec<SparseInst>)->Vec<SparseInst>{
    let mut output:Vec<SparseInst> = Vec::with_capacity(extracted.len());
    let mut cur_o_pos = 0;
    for SparseInst { inst, .. } in extracted {
        let (o_start,slice_len,seq_len) = match (inst.inst_type(),inst.vcd_trgt()){
            (InstType::Copy (CopyType::CopyS),false) |
            (InstType::Run, _) |
            (InstType::Add, _) => {
                let o_pos_start = cur_o_pos;
                cur_o_pos += inst.len_in_o() as u64;
                output.push(SparseInst { o_pos_start, inst });
                continue;
            },
            (InstType::Copy (CopyType::CopyQ { len_o }),_) => {
                let slice_len = inst.len_in_u() - len_o;
                let o_start = cur_o_pos - slice_len as u64;
                (o_start,slice_len,len_o)
            },
            (InstType::Copy (CopyType::CopyT { inst_u_pos_start }),_) => {
                let copy = inst.clone().take_copy().unwrap();
                let offset = inst_u_pos_start - copy.u_pos;
                let o_start = cur_o_pos - offset as u64;
                (o_start,copy.len_in_u(),0)
            },
            (InstType::Copy (CopyType::CopyS),true) => {
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
                output.push(SparseInst { o_pos_start, inst: resolved_inst.inst });
            }
        }
    }
    output
}

fn find_copy_s(extract:&[SparseInst],shift:usize,dest:&mut Vec<usize>){
    for (i,ext) in extract.iter().enumerate(){
        match (ext.inst_type(),ext.inst.vcd_trgt()){
            (InstType::Copy (CopyType::CopyS),false) => dest.push(i+shift),
            _ => (),
        }
    }
}

///Merger struct that can accept merging of additional patches.
#[derive(Clone, Debug)]
pub struct Merger{
    ///The summary patch that will be written to the output.
    terminal_patch: Vec<SparseInst>,
    ///If this is empty, merging a patch will have no effect.
    ///These are where TerminalInst::CopySS are found.
    terminal_copy_indices: Vec<usize>,
    //final_size: u64,
}

impl Merger {
    ///Creates a new Merger from a terminal patch.
    ///This should only be called using the patch that generates the output file you want.
    /// # Arguments
    /// * `terminal_patch` - The terminal patch that will serve as the core set of instructions.
    /// # Returns
    /// If the terminal patch has no Copy instructions, a SummaryPatch is returned.
    /// If the terminal patch has even a single Copy instructions, a Merger is returned.
    pub fn new<R:Read + Seek>(terminal_patch:R) -> std::io::Result<Result<Merger,SummaryPatch>> {
        let (terminal_patch,stats) = extract_patch_instructions(terminal_patch)?;
        if stats.copy_bytes == 0{
            return Ok(Err(SummaryPatch(terminal_patch)));
        }
        let mut terminal_copy_indices = Vec::new();
        //we for sure need to translate local. I think translate global isn't needed??
        //will need to check this.
        let terminal_patch = deref_non_copy_ss(terminal_patch);
        find_copy_s(&terminal_patch,0,&mut terminal_copy_indices);
        debug_assert!(!terminal_copy_indices.is_empty(), "terminal_copy_indices should not be empty");
        Ok(Ok(Merger{
            terminal_patch,
            terminal_copy_indices,
            //final_size:stats.output_size as u64
        }))
    }
    ///Merges a predecessor patch into the terminal patch.
    ///This should be called using proper order of patches.
    /// # Arguments
    /// * `predecessor_patch` - The patch to merge into the current summary patch.
    /// # Returns
    /// If the resulting patch has no Copy instructions, a SummaryPatch is returned.
    /// If the resulting patch has even a single Copy instructions, a Merger is returned.
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
            let SparseInst { inst,.. } = self.terminal_patch[i].clone();
            let copy = inst.take_copy().expect("Expected Copy");
            //this a src window copy that we need to resolve from the predecessor patch.
            debug_assert!(copy.copy_in_s());
            debug_assert!(!copy.vcd_trgt());
            let o_start = copy.ssp + copy.u_pos as u64; //ssp is o_pos, u is offset from that.
            let resolved = get_exact_slice(&predecessor_patch, o_start, copy.len_in_u()).unwrap();
            //debug_assert_eq!(sum_len_in_o(&resolved), copy.len_in_o() as u64, "resolved: {:?} copy: {:?}",resolved,copy);
            find_copy_s(&resolved, i+shift, &mut terminal_copy_indices);
            shift += resolved.len() - 1;
            inserts.push((i, resolved));

        }
        //now we expand the old copy values with the derefd instructions.
        self.terminal_patch = expand_elements(self.terminal_patch, inserts);
        //debug_assert_eq!(sum_len_in_o(&self.terminal_patch), self.final_size, "final size: {} sum_len: {}",self.final_size,sum_len_in_o(&self.terminal_patch));
        if terminal_copy_indices.is_empty(){
            Ok(Err(SummaryPatch(self.terminal_patch)))
        }else{
            self.terminal_copy_indices = terminal_copy_indices;
            Ok(Ok(self))
        }
    }
    pub fn finish(self)->SummaryPatch{
        SummaryPatch(self.terminal_patch)
    }

}

///This is returned when the current summary patch contains no Copy instructions, OR when you are finished with the Merger.
#[derive(Debug)]
pub struct SummaryPatch(Vec<SparseInst>);
impl SummaryPatch{
    ///Writes the summary patch to a sink.
    /// # Arguments
    /// * `sink` - The sink to write the summary patch to.
    /// * `max_u_size` - The maximum size of the super string U. If None, the default is 256MB. This is used to help determine when new windows are created.
    /// # Returns
    /// The sink that was passed in.
    pub fn write<W:Write>(self,sink:W,max_u_size:Option<usize>)->std::io::Result<W>{
        let max_u_size = max_u_size.unwrap_or(1<<28); //256MB
        let header = Header::default();
        let encoder = VCDWriter::new(sink,header)?;
        let mut state = WriterState{
            cur_o_pos: 0,
            max_u_size,
            cur_win: Vec::new(),
            sink: encoder,
            win_sum: Some(Default::default()),
        };
        for inst in self.0.into_iter(){
            state.apply_instruction(inst.inst)?;
        }
        state.flush_window()?;
        state.sink.finish()
    }
}

struct WriterState<W>{
    cur_o_pos: u64,
    max_u_size: usize,
    cur_win: Vec<DInst>,
    win_sum: Option<WriteWindowHeader>,
    sink: VCDWriter<W>,
}

impl<W:Write> WriterState<W> {
    ///This is the 'terminal' command for moving the merger forward.
    fn apply_instruction(&mut self, instruction: DInst) ->std::io::Result<()>{
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
    fn add_to_window(&mut self,next_inst:DInst){
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
    fn handle_indicator(&mut self,inst:&DInst)->std::io::Result<()>{
        let win_sum = self.win_sum.get_or_insert(Default::default());

        match (win_sum.win_indicator,comp_indicator(&inst.inst_type(),&win_sum.win_indicator,inst.vcd_trgt())){
            (_, None) => (),
            (WinIndicator::Neither, Some(set)) => {
                win_sum.win_indicator = set;
            },
            (WinIndicator::VCD_TARGET, Some(next)) |
            (WinIndicator::VCD_SOURCE, Some(next)) => {
                self.flush_window()?;
                let mut h = WriteWindowHeader::default();
                h.win_indicator = next;
                self.win_sum = Some(h);
            },
        }
        Ok(())
    }
    ///(ssp,sss, size_of_the_target_window (T))
    fn new_win_sizes(&self,inst:&DInst)->(Option<(u64,u64)>,u64){
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
                DInst::Add(ExAdd{bytes}) => {
                    self.sink.next_inst(WriteInst::ADD(bytes)).unwrap();
                },
                DInst::Run(r) => {
                    self.sink.next_inst(WriteInst::RUN(r)).unwrap();
                },
                DInst::Copy(DCopy { len_u, u_pos , ssp,copy_type,.. }) => {
                    //we need to translate this so our u_pos is correct
                    let output_ssp = output_ssp.expect("output_ssp should be set here");
                    //our ssp is within (or coincident) to the bounds of the source segment
                    let copy_inst = if output_ssp > ssp{
                        let neg_shift = output_ssp - ssp;
                        COPY { len:len_u, u_pos: u_pos - neg_shift as u32,copy_type }
                    }else{
                        let pos_shift = ssp - output_ssp;
                        COPY { len:len_u, u_pos: u_pos + pos_shift as u32,copy_type }
                    };
                    self.sink.next_inst(WriteInst::COPY(copy_inst)).unwrap();
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

fn expand_sequence(seq:&[SparseInst],len:u32,cur_o_pos:&mut u64,output:&mut Vec<SparseInst>) {
    let mut current_len = 0;
    // Calculate the effective length after considering truncation
    while current_len < len {
        for SparseInst {  inst, .. } in seq.iter().cloned() {
            let end_pos = current_len + inst.len_in_o();
            let mut modified_instruction = inst;
            // After skip has been accounted for, directly clone and potentially modify for truncation
            // If adding this instruction would exceed the effective length, apply truncation
            if end_pos > len {
                let trunc_amt = end_pos - len;
                modified_instruction.trunc(trunc_amt);
            }
            let inst_len = modified_instruction.len_in_o();
            output.push(SparseInst { o_pos_start:*cur_o_pos, inst: modified_instruction });
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
fn expand_elements(mut target: Vec<SparseInst>, inserts: Vec<(usize, Vec<SparseInst>)>) -> Vec<SparseInst>{
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

    use vcdiff_common::DeltaIndicator;
    use vcdiff_decoder::apply_patch;
    use vcdiff_writer::WriteWindowHeader;

    use super::*;
    /*
    Basic merger tests will start with a src file of '01234'
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
    const WIN_HDR:WriteWindowHeader = WriteWindowHeader {
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
        let mut encoder = VCDWriter::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WIN_HDR).unwrap();
        // Instructions
        encoder.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 0,copy_type:CopyType::CopyS })).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 2, u_pos: 2,copy_type:CopyType::CopyS  })).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 10, u_pos: 5,copy_type:CopyType::CopyQ { len_o:7 }  })).unwrap();
        // Force encoding
        let w = encoder.finish().unwrap().into_inner();
        make_patch_reader(w)
    }
    fn copy_patch() -> Cursor<Vec<u8>> {
        let mut encoder = VCDWriter::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WIN_HDR).unwrap();
        // Instructions
        encoder.next_inst(WriteInst::COPY(COPY { len: 5, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 5, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap();
        // Force encoding
        let w = encoder.finish().unwrap().into_inner();
        make_patch_reader(w)
    }
    fn add_run_patch() -> Cursor<Vec<u8>> {
        let mut encoder = VCDWriter::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WIN_HDR).unwrap();
        // Instructions
        encoder.next_inst(WriteInst::ADD("A".as_bytes().to_vec())).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 2, u_pos: 1,copy_type:CopyType::CopyS  })).unwrap();
        encoder.next_inst(WriteInst::RUN(RUN { len: 3, byte: b'X' })).unwrap();
        encoder.next_inst(WriteInst::ADD("YZ".as_bytes().to_vec())).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 2, u_pos: 3,copy_type:CopyType::CopyS  })).unwrap();

        // Force encoding
        let w = encoder.finish().unwrap().into_inner();
        make_patch_reader(w)
    }
    fn complex_patch()->Cursor<Vec<u8>>{
        let mut encoder = VCDWriter::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WIN_HDR).unwrap();
        // Instructions
        encoder.next_inst(WriteInst::ADD("Y".as_bytes().to_vec())).unwrap();
        encoder.next_inst(WriteInst::RUN(RUN { len: 2, byte: b'Z' })).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 4,copy_type:CopyType::CopyS  })).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 2, u_pos: 5,copy_type:CopyType::CopyT { inst_u_pos_start: 9 }  })).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 4, u_pos: 1,copy_type:CopyType::CopyS  })).unwrap();
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
        apply_patch(&mut cursor, Some(&mut Cursor::new(SRC.to_vec())), &mut output).unwrap();
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
        apply_patch(&mut cursor, Some(&mut Cursor::new(SRC.to_vec())), &mut output).unwrap();
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
        apply_patch(&mut cursor, Some(&mut Cursor::new(SRC.to_vec())), &mut output).unwrap();
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
        apply_patch(&mut cursor, Some(&mut Cursor::new(SRC.to_vec())), &mut output).unwrap();
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
        apply_patch(&mut cursor, Some(&mut Cursor::new(SRC.to_vec())), &mut output).unwrap();
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
        let mut encoder = VCDWriter::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WriteWindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(5), source_segment_position: Some(0), size_of_the_target_window:5 , delta_indicator: DeltaIndicator(0) }).unwrap();
        // Instructions
        encoder.next_inst(WriteInst::COPY(COPY { len: 5, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap();
        encoder.start_new_win(WriteWindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(1), source_segment_position: Some(4), size_of_the_target_window:6 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(WriteInst::ADD(" w".as_bytes().to_vec())).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap();
        encoder.next_inst(WriteInst::ADD("rld".as_bytes().to_vec())).unwrap();
        encoder.start_new_win(WriteWindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(1), source_segment_position: Some(5), size_of_the_target_window:1 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap();
        let p1 = encoder.finish().unwrap().into_inner();
        let p1_answer = b"hello world!";
        let mut cursor = Cursor::new(p1.clone());
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(&mut Cursor::new(src.to_vec())), &mut output).unwrap();
        assert_eq!(output,p1_answer); //ensure our instructions do what we think they are.
        let patch_1 = make_patch_reader(p1);
        let mut encoder = VCDWriter::new(Cursor::new(Vec::new()), HDR).unwrap();
        encoder.start_new_win(WriteWindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(11), source_segment_position: Some(1), size_of_the_target_window:7 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(WriteInst::ADD("H".as_bytes().to_vec())).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 4, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap(); //ello
        encoder.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 10,copy_type:CopyType::CopyS  })).unwrap(); //'!'
        encoder.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 4,copy_type:CopyType::CopyS  })).unwrap(); //' '
        encoder.start_new_win(WriteWindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(7), source_segment_position: Some(0), size_of_the_target_window:14 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 7, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap(); //'Hello! '
        encoder.next_inst(WriteInst::COPY(COPY { len: 12, u_pos: 7,copy_type:CopyType::CopyQ { len_o: 5 }  })).unwrap(); //'Hello! Hello'
        encoder.next_inst(WriteInst::ADD(".".as_bytes().to_vec())).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 13,copy_type:CopyType::CopyS  })).unwrap(); // ' '
        encoder.start_new_win(WriteWindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(6), source_segment_position: Some(15), size_of_the_target_window:7, delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(WriteInst::ADD("h".as_bytes().to_vec())).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 6, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap(); //'ello. '
        encoder.start_new_win(WriteWindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(6), source_segment_position: Some(21), size_of_the_target_window:8 , delta_indicator: DeltaIndicator(0) }).unwrap();
        encoder.next_inst(WriteInst::COPY(COPY { len: 6, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap(); //'hello.'
        encoder.next_inst(WriteInst::COPY(COPY { len: 3, u_pos: 11,copy_type:CopyType::CopyQ { len_o: 2 }  })).unwrap(); //Seq '.' == Run(3) '.'
        let p2 = encoder.finish().unwrap().into_inner();
        let p2_answer = b"Hello! Hello! Hello. hello. hello...";
        let mut cursor = Cursor::new(p2.clone());
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(&mut Cursor::new(p1_answer.to_vec())), &mut output).unwrap();
        assert_eq!(output,p2_answer);
        let patch_2 = make_patch_reader(p2);
        let merger = Merger::new(patch_2).unwrap().unwrap();
        let merger = merger.merge(patch_1).unwrap().unwrap();
        let merged_patch = merger.finish().write(Vec::new(), None).unwrap();
        let mut cursor = Cursor::new(merged_patch);
        let mut output = Vec::new();
        let answer = b"Hello! Hello! Hello. hello. hello...";
        apply_patch(&mut cursor, Some(&mut Cursor::new(src.to_vec())), &mut output).unwrap();
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
            SparseInst { o_pos_start: 0, inst: DInst::Copy(DCopy { u_pos: 0, len_u: 2, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            SparseInst { o_pos_start: 2, inst: DInst::Copy(DCopy { u_pos: 4, len_u: 6, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            SparseInst { o_pos_start: 8, inst: DInst::Copy(DCopy { u_pos: 12, len_u: 4, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
        ];
        let mut output = Vec::new();
        expand_sequence(&seq, 15, &mut 0,&mut output);
        let result = vec![
            SparseInst { o_pos_start: 0, inst: DInst::Copy(DCopy { u_pos: 0, len_u: 2, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            SparseInst { o_pos_start: 2, inst: DInst::Copy(DCopy { u_pos: 4, len_u: 6, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            SparseInst { o_pos_start: 8, inst: DInst::Copy(DCopy { u_pos: 12, len_u: 4, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            SparseInst { o_pos_start: 12, inst: DInst::Copy(DCopy { u_pos: 0, len_u: 2, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            SparseInst { o_pos_start: 14, inst: DInst::Copy(DCopy { u_pos: 4, len_u: 1, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
        ];
        assert_eq!(output, result, "Output should contain a truncated instruction");
    }
}