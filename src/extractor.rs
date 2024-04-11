

use std::{fmt::Debug, io::{Read, Seek}, num::NonZeroU32, ops::Range};

use crate::{decoder::{DecInst, VCDDecoder, VCDiffDecodeMsg::*},reader::{VCDReader, WinIndicator}, ADD, COPY, RUN};

///Disassociated Copy (from the window it was found in).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisCopy{
    pub u_pos:u32,
    pub len_u:u32,
    pub sss:u32,
    pub ssp:u64,
    pub vcd_trgt:bool,
    pub copy_type:CopyType,
}
impl Instruction for DisCopy{
    fn len_in_o(&self)->u32{
        match self.copy_type{
            CopyType::CopyQ{len_o} => len_o,
            _ => self.len_u,
        }
    }
    fn len_in_u(&self)->u32{
        self.len_u
    }
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
    fn inst_type(&self)->InstType {
        InstType::Copy{copy_type:self.copy_type,vcd_trgt:self.vcd_trgt}
    }
    fn src_range(&self)->Option<Range<u64>>{
        Some(self.min_src())
    }
}
impl CopyInst for DisCopy{
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CopyType{
    CopyS,
    CopyT{inst_u_pos_start:u32},
    CopyQ{len_o:u32}
}

impl CopyType {
    pub fn in_s(&self)->bool{
        matches!(self, CopyType::CopyS)
    }
    pub fn in_t(&self)->bool{
        matches!(self, CopyType::CopyT{..})
    }
    pub fn is_seq(&self)->bool{
        matches!(self, CopyType::CopyQ{..})
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExAdd{
    pub bytes:Vec<u8>,
}
impl Instruction for ExAdd{
    fn len_in_u(&self)->u32{
        self.bytes.len() as u32
    }
    fn len_in_o(&self)->u32{
        self.bytes.len() as u32
    }
    fn skip(&mut self,amt:u32){
        self.bytes = self.bytes.split_off(amt as usize);
    }
    fn trunc(&mut self,amt:u32){
        self.bytes.truncate(self.bytes.len() - amt as usize);
    }
    fn inst_type(&self)->InstType {
        InstType::Add
    }
    fn src_range(&self)->Option<Range<u64>> {
        None
    }
}
impl Instruction for RUN{
    fn len_in_u(&self)->u32{
        self.len
    }
    fn len_in_o(&self)->u32{
        self.len
    }
    fn skip(&mut self,amt:u32){
        self.len -= amt;
    }
    fn trunc(&mut self,amt:u32){
        self.len = amt;
    }
    fn inst_type(&self)->InstType {
        InstType::Run
    }
    fn src_range(&self)->Option<Range<u64>> {
        None
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExInstType<C>{
    Add(ExAdd),
    Run(RUN),
    Copy(C),
}
impl<C:Instruction> Instruction for ExInstType<C>{
    fn len_in_u(&self)->u32{
        match self{
            ExInstType::Add(bytes) => bytes.len_in_u(),
            ExInstType::Run(run) => run.len,
            ExInstType::Copy(copy) => copy.len_in_u(),
        }
    }
    fn len_in_o(&self)->u32{
        match self{
            ExInstType::Add(bytes) => bytes.len_in_o(),
            ExInstType::Run(run) => run.len,
            ExInstType::Copy(copy) => copy.len_in_o(),
        }
    }
    fn skip(&mut self,amt:u32){
        match self{
            ExInstType::Add(bytes) => bytes.skip(amt),
            ExInstType::Run(run) => run.skip(amt),
            ExInstType::Copy(copy) => copy.skip(amt),
        }
    }
    fn trunc(&mut self,amt:u32){
        match self{
            ExInstType::Add(bytes) => bytes.trunc(amt),
            ExInstType::Run(run) => run.trunc(amt),
            ExInstType::Copy(copy) => copy.trunc(amt),
        }
    }
    fn inst_type(&self)->InstType{
        match self{
            ExInstType::Add(_) => InstType::Add,
            ExInstType::Run(_) => InstType::Run,
            ExInstType::Copy(copy) => copy.inst_type(),
        }
    }
    fn src_range(&self)->Option<Range<u64>>{
        match self{
            ExInstType::Add(_) => None,
            ExInstType::Run(_) => None,
            ExInstType::Copy(copy) => copy.src_range(),
        }
    }
}
impl<C> ExInstType<C>{
    pub fn take_copy(self)->Option<C>{
        match self{
            ExInstType::Copy(copy) => Some(copy),
            _ => None,
        }
    }
}


pub type ExtractedInst = ExInstType<DisCopy>;


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisInst<V>{
    pub o_pos_start:u64,
    pub inst:V,
}
impl<V:Instruction> Instruction for DisInst<V>{
    fn len_in_o(&self)->u32{
        self.inst.len_in_o()
    }
    fn len_in_u(&self)->u32{
        self.inst.len_in_u()
    }
    fn skip(&mut self,amt:u32){
        self.o_pos_start += amt as u64;
        self.inst.skip(amt);
    }
    fn trunc(&mut self,amt:u32){
        self.inst.trunc(amt);
    }
    fn inst_type(&self)->InstType{
        self.inst.inst_type()
    }
    fn src_range(&self)->Option<Range<u64>>{
        self.inst.src_range()
    }
}
impl<V:Instruction> PosInst for DisInst<V>{
    fn o_start(&self)->u64{
        self.o_pos_start
    }
}

pub type VcdExtract = DisInst<ExtractedInst>;
pub trait PosInst:Instruction{
    fn o_start(&self)->u64;
}
pub trait Instruction:Clone{
    fn len_in_u(&self)->u32;
    fn len_in_o(&self)->u32;
    fn skip(&mut self,amt:u32);
    fn trunc(&mut self,amt:u32);
    fn inst_type(&self)->InstType;
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
    fn is_implicit_seq(&self)->bool{
        matches!(self.inst_type(), InstType::Copy{copy_type:CopyType::CopyQ{..},..})
    }
    fn split_at(mut self,first_inst_len:u32)->(Self,Self){
        assert!(!self.is_implicit_seq());
        let mut second = self.clone();
        self.trunc(self.len_in_u() - first_inst_len);
        second.skip(first_inst_len);
        (self,second)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InstType{
    Add,
    Run,
    Copy{copy_type:CopyType,vcd_trgt:bool},
}

impl InstType {
    pub fn comp_indicator(&self,cur_ind:&WinIndicator)->Option<WinIndicator>{
        match (self, cur_ind,self.vcd_trgt()){
            (InstType::Copy { .. }, WinIndicator::VCD_SOURCE,true) |
            (InstType::Copy {.. }, WinIndicator::Neither,true) => Some(WinIndicator::VCD_TARGET),
            (InstType::Copy {.. }, WinIndicator::VCD_TARGET,false) |
            (InstType::Copy {.. }, WinIndicator::Neither,false) => Some(WinIndicator::VCD_SOURCE),
            _ => None,
        }
    }
    pub fn vcd_trgt(&self)->bool{
        match self{
            InstType::Copy{vcd_trgt,..} => *vcd_trgt,
            _ => false,
        }
    }
    pub fn is_copy(&self)->bool{
        matches!(self, InstType::Copy{..})
    }
}
pub trait CopyInst:Instruction{
    fn u_start_pos(&self)->u32;
    fn ssp(&self)->u64;
    fn sss(&self)->u32;
    fn vcd_trgt(&self)->bool;

    fn in_s(&self)->bool{
        (self.u_start_pos() + self.len_in_u()) <= self.sss()
    }
    fn in_t(&self)->bool{
        !self.in_s()
    }
    fn min_src(&self)->Range<u64>{
        let new_ssp = self.ssp() + self.u_start_pos() as u64;
        let new_end = new_ssp + self.len_in_u() as u64;
        new_ssp..std::cmp::min(new_end,new_ssp+self.sss() as u64)
    }
}
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
pub(crate) fn sum_len<I:PosInst>(insts:&[I])->u64{
    insts.iter().map(|i| i.len_in_o() as u64).sum()
}
pub fn get_exact_slice<I:PosInst+Debug>(insts:&[I],start:u64,len:u32)->Option<Vec<I>>{
    let start_idx = find_controlling_inst(insts,start)?;
    let end_pos = start + len as u64;
    let mut slice = Vec::new();

    for inst in insts[start_idx..].iter() {
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

            break;
        }
    }
    debug_assert!(sum_len(&slice)==len as u64,"{} != {} start:{} end_pos:{} ... {:?}",sum_len(&slice),len,start,end_pos,&slice);
    Some(slice)
}
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
pub fn extract_patch_instructions<R:Read + Seek>(patch:R)->std::io::Result<(Vec<VcdExtract>, Stats)>{
    let mut insts = Vec::new();
    let mut dec = VCDDecoder::new(VCDReader::new(patch)?);
    let mut ssp = None;
    let mut sss = None;
    let mut vcd_trgt = false;
    let mut o_pos_start = 0;
    let mut stats = Stats::new();
    loop{
        match dec.next()?{
            WindowSummary(ws) => {
                ssp = ws.source_segment_position;
                sss = ws.source_segment_size;
                if ws.win_indicator == WinIndicator::VCD_TARGET{
                    vcd_trgt = true;
                    stats.vcd_trgt();
                }
            },
            Inst { u_start, first, second } => {
                let mut cur_u = u_start as u32;
                for inst in [Some(first), second]{
                    if inst.is_none(){
                        continue;
                    }
                    let inst = inst.unwrap();
                    let len_u = inst.len_in_u();
                    let len_o = inst.len_in_o(cur_u as usize);
                    //dbg!(len_u, len_o, cur_u, o_pos_start,&inst);
                    match inst{
                        DecInst::Add(ADD{ len, p_pos }) => {
                            let mut bytes = vec![0; len as usize];
                            dec.reader().read_from_src(p_pos, &mut bytes)?;
                            insts.push(VcdExtract{o_pos_start,inst:ExtractedInst::Add(ExAdd { bytes })});
                            stats.add(len as usize);
                        },
                        DecInst::Run(run) => {
                            stats.run(run.len as usize);
                            insts.push(VcdExtract{o_pos_start,inst:ExtractedInst::Run(run)})
                        },
                        DecInst::Copy(COPY{ len, u_pos }) =>{
                            let ssp = ssp.expect("SSP not set");
                            let sss = sss.expect("SSS not set");
                            let len_u = len;
                            let end_pos = u_pos + len;
                            //dbg!(end_pos,cur_u, len_u, sss, ssp);
                            let copy_type = if end_pos > cur_u{//seQ
                                assert!(u_pos as u64 >= sss,"CopyT must be entirely in T!");
                                let len_o = end_pos - cur_u;
                                stats.copy_q(len_o as usize);
                                CopyType::CopyQ{len_o}
                            }else if end_pos as u64 <= sss{//inS
                                stats.copy_s(len_u as usize);
                                CopyType::CopyS
                            }else{//inT
                                debug_assert!(len_u == len_o as u32,"Length Mismatch! Is this a seq?");
                                assert!(u_pos as u64 >= sss,"CopyT must be entirely in T!");
                                stats.copy_t(len_u as usize);
                                CopyType::CopyT{inst_u_pos_start:cur_u}
                            };
                            insts.push(VcdExtract{o_pos_start,inst:ExtractedInst::Copy(DisCopy{
                                u_pos,
                                len_u,
                                ssp,
                                sss:sss as u32,
                                vcd_trgt,
                                copy_type,
                            })});
                        }
                    }
                    o_pos_start += len_o as u64;
                    cur_u += len_u as u32;
                }
            },
            EndOfWindow => {
                ssp = None;
                sss = None;
                vcd_trgt = false;
            },
            EndOfFile => break,
        }
    }
    Ok((insts,stats))
}