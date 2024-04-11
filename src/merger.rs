use core::panic;
use std::{fmt::Debug, io::{Read, Seek,Write}, ops::Range, os::linux::raw::stat};

use crate::{encoder::{EncInst, VCDEncoder, WindowHeader}, extractor::{extract_patch_instructions, get_exact_slice, CopyInst, CopyType, DisCopy, DisInst, ExAdd, ExInstType, ExtractedInst, InstType, Instruction, PosInst, VcdExtract}, reader::{Header, WinIndicator}, COPY};

/*
In theory we only ever need to compare two patch files at once.
We have an 'earlier' and a 'later' patch.
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

We need to understand when these various copy commands have precedence in 'later' vs 'earlier' patch files.

Here are my first thoughts:
TU/TS/SO found in 'later' can be preserved as-is, since these are indirectly references to *other instructions*.
    However, we cannot add any new TU/TS to 'later'.
SS found in 'later' needs to be 'dereferenced' to instructions in 'earlier'.
    We need to replace the 'later' instructions with one or more instructions from 'earlier'.
    This requires us potentially modifying the controlling instructions so that the length matches the copy instruction found.
    This would require 'skipping' bytes on the first instruction, and potentially 'truncating' bytes off the last instruction.
    ...If some of these instructions are TU or TS in 'earlier' we might have to do something more to them?
    TS would need to be 'de-sequenced'. Since, the implicit nature of the the Copy requires the position in U for the sequence to work properly.
    So we would need to walk through the 'earlier' patch to detect and make the implicit seq explicit in our code.
    A TU encountered in 'earlier' can be directly placed in the 'later' patch?
    Or does it need some sort of translation?

A TU in the earlier patch would be referencing commands earlier commands from *its* patch file.
However, this can only happen if we are de-referencing a CopyS* from the 'later' patch.
Since we are selectively merging instructions, moving a TU/TS/SO to 'later' does not semantically follow.
We need to de-reference any non SS Copy so it can be merged semantically in later.
*/

// #[derive(Clone, Debug)]
// pub struct CopyQ{
//     inst:Vec<MInst>,
//     len:u32,
//     skip:u32,
//     trunc:u32,
// }
// impl Instruction for CopyQ{
//     fn len_in_u(&self)->u32 {
//         self.len
//     }

//     fn len_in_o(&self)->u32 {
//         self.inst.iter().map(|f|f.len_in_o()).sum()
//     }

//     fn skip(&mut self,amt:u32) {
//         self.skip = amt;
//     }

//     fn trunc(&mut self,amt:u32) {
//         self.trunc = amt;
//     }


//     fn inst_type(&self)->InstType {
//         unimplemented!("inst_type")
//     }

//     fn src_range(&self)->Option<Range<u64>> {
//         unimplemented!("src_range")
//     }
// }

// #[derive(Clone, Debug)]
// pub enum MergeCopy{
//     ///Any SO/TU/TS found in 'terminal' patch
//     Terminal(DisCopy),
//     ///Any TS found in a predecessor patch
//     PredecessorSeq(CopyQ),
//     ///Any SS found in terminal or a merged predecessor patch
//     SourceCopy(DisCopy),
//     ///Any SO found in a predecessor patch
//     PredecessorSO(DisCopy),
// }
// impl Instruction for MergeCopy{
//     fn len_in_u(&self)->u32 {
//         match self {
//             MergeCopy::Terminal(c) => c.len_in_u(),
//             MergeCopy::PredecessorSeq(c) => c.len_in_u(),
//             MergeCopy::SourceCopy(c) => c.len_in_u(),
//             MergeCopy::PredecessorSO(c) => c.len_in_u(),
//         }
//     }

//     fn len_in_o(&self)->u32 {
//         match self {
//             MergeCopy::Terminal(c) => c.len_in_o(),
//             MergeCopy::PredecessorSeq(c) => c.len_in_o(),
//             MergeCopy::SourceCopy(c) => c.len_in_o(),
//             MergeCopy::PredecessorSO(c) => c.len_in_o(),
//         }
//     }

//     fn skip(&mut self,amt:u32) {
//         match self {
//             MergeCopy::Terminal(c) => c.skip(amt),
//             MergeCopy::PredecessorSeq(c) => c.skip(amt),
//             MergeCopy::SourceCopy(c) => c.skip(amt),
//             MergeCopy::PredecessorSO(c) => c.skip(amt),
//         }
//     }

//     fn trunc(&mut self,amt:u32) {
//         match self {
//             MergeCopy::Terminal(c) => c.trunc(amt),
//             MergeCopy::PredecessorSeq(c) => c.trunc(amt),
//             MergeCopy::SourceCopy(c) => c.trunc(amt),
//             MergeCopy::PredecessorSO(c) => c.trunc(amt),
//         }
//     }

//     fn inst_type(&self)->InstType {
//         unimplemented!("inst_type")
//     }

//     fn src_range(&self)->Option<Range<u64>> {
//         unimplemented!("src_range")
//     }

// }
// impl MergeCopy{
//     fn take_ss(self)->DisCopy{
//         match self{
//             MergeCopy::SourceCopy(c) => c,
//             _ => panic!("Expected SourceCopy"),
//         }
//     }
//     fn take_so(self)->DisCopy{
//         match self{
//             MergeCopy::PredecessorSO(c) => c,
//             _ => panic!("Expected PredecessorSO"),
//         }
//     }
//     fn take_seq(self)->CopyQ{
//         match self{
//             MergeCopy::PredecessorSeq(c) => c,
//             _ => panic!("Expected PredecessorSeq"),
//         }
//     }
//     fn take_discopy(self)->DisCopy{
//         match self{
//             MergeCopy::Terminal(a) |
//             MergeCopy::PredecessorSO(a) |
//             MergeCopy::SourceCopy(a) => a,
//             MergeCopy::PredecessorSeq(_) => panic!("Expected DisCopy"),
//         }
//     }
// }
// pub type MInst = DisInst<ExInstType<MergeCopy>>;
fn translate_and_deref_term(extracted:Vec<VcdExtract>)->Vec<VcdExtract>{
    let mut output:Vec<VcdExtract> = Vec::with_capacity(extracted.len());
    let mut cur_o_pos = 0;
    for DisInst { inst, .. } in extracted {
        //dbg!(&inst);
        match inst {
            ExtractedInst::Copy(copy) => {
                //here we can have a reg copy in T or a seq
                //since we know we are in T either way
                //our o position should be u_pos - sss
                if copy.is_implicit_seq(){//CopyTS
                    let o_start = (copy.u_pos - copy.sss) as u64;
                    let slice_len = copy.len_in_u() - copy.len_in_o();
                    let o_len = copy.len_in_o();
                    //dbg!(&o_start,slice_len);
                    let seq_slice_local = get_exact_slice(output.as_slice(), o_start, slice_len).unwrap();
                    //dbg!(&seq_slice_local);
                    let seq_slice_fully_resolved = global_deref_list(&output, seq_slice_local);
                    //dbg!(&seq_slice_fully_resolved);
                    let output_start = cur_o_pos;
                    cur_o_pos += o_len as u64;
                    //now we need to generate the seq for the output
                    expand_sequence(&seq_slice_fully_resolved, o_len,output_start, &mut output)
                }else if !copy.in_s() || copy.vcd_trgt(){//CopySO/TU
                    let vcd_trgt = copy.vcd_trgt();
                    let in_s = if !copy.in_s(){
                        //dbg!(&copy);
                        let o_start = (copy.u_pos - copy.sss) as u64;
                        get_exact_slice(output.as_slice(), o_start, copy.len_in_u()).unwrap()
                    }else{//CopySO
                        let o_pos_start = cur_o_pos;
                        vec![VcdExtract { o_pos_start, inst: ExtractedInst::Copy(copy) }]
                    };
                    let mut resolved = if vcd_trgt{
                        global_deref_list(&output, in_s)
                    }else{in_s};
                    output.append(&mut resolved);
                }else{//CopySS
                    let o_pos_start = cur_o_pos;
                    cur_o_pos += copy.len_in_o() as u64;
                    output.push(DisInst { o_pos_start, inst: ExtractedInst::Copy(copy) });
                };

            },
            inst => {//pass through no-op
                let o_pos_start = cur_o_pos;
                cur_o_pos += inst.len_in_o() as u64;
                output.push(DisInst { o_pos_start, inst })
            },
        }
    }
    output
}
pub fn translate_and_deref_pred(extracted:Vec<VcdExtract>)->Vec<VcdExtract>{
    let mut output:Vec<VcdExtract> = Vec::with_capacity(extracted.len());
    let mut cur_o_pos = 0;
    for DisInst { inst, .. } in extracted {
        //dbg!(&inst);
        match inst {
            ExtractedInst::Copy(copy) => {
                //here we can have a reg copy in T or a seq
                //since we know we are in T either way
                //our o position should be u_pos - sss
                if copy.is_implicit_seq(){//CopyTS
                    let o_start = (copy.u_pos - copy.sss) as u64;
                    let slice_len = copy.len_in_u() - copy.len_in_o();
                    let o_len = copy.len_in_o();
                    //dbg!(&o_start,slice_len);
                    let seq_slice_local = get_exact_slice(output.as_slice(), o_start, slice_len).unwrap();
                    //dbg!(&seq_slice_local);
                    let seq_slice_fully_resolved = global_deref_list(&output, seq_slice_local);
                    //dbg!(&seq_slice_fully_resolved);
                    let output_start = cur_o_pos;
                    cur_o_pos += o_len as u64;
                    //now we need to generate the seq for the output
                    expand_sequence(&seq_slice_fully_resolved, o_len,output_start, &mut output)
                }else if !copy.in_s() || copy.vcd_trgt(){//CopySO/TU
                    let vcd_trgt = copy.vcd_trgt();
                    let in_s = if !copy.in_s(){
                        dbg!(&copy);
                        let o_start = (copy.u_pos - copy.sss) as u64;
                        get_exact_slice(output.as_slice(), o_start, copy.len_in_u()).unwrap()
                    }else{//CopySO
                        let o_pos_start = cur_o_pos;
                        vec![VcdExtract { o_pos_start, inst: ExtractedInst::Copy(copy) }]
                    };
                    let mut resolved = if vcd_trgt{
                        global_deref_list(&output, in_s)
                    }else{in_s};
                    //get the output start positions correct
                    resolved.drain(..).for_each(|f|{
                        let o_pos_start = cur_o_pos;
                        cur_o_pos += f.inst.len_in_o() as u64;
                        output.push(DisInst { o_pos_start, inst: f.inst });
                    });
                    output.append(&mut resolved);
                }else{//CopySS
                    let o_pos_start = cur_o_pos;
                    cur_o_pos += copy.len_in_o() as u64;
                    output.push(DisInst { o_pos_start, inst: ExtractedInst::Copy(copy) });
                };

            },
            inst => {//pass through no-op
                let o_pos_start = cur_o_pos;
                cur_o_pos += inst.len_in_o() as u64;
                output.push(DisInst { o_pos_start, inst })
            },
        }
    }
    output
}
fn global_deref_list(cur_output:&[VcdExtract],inst:Vec<VcdExtract>)->Vec<VcdExtract>{
    //this sequence should be S normalized already
    //we just need to find the parent slice in O
    //We treat all the Copys found based on their own vcd_trgt value
    let mut output = Vec::with_capacity(inst.len());
    for mi in inst {
        match global_deref_inst(&mi){
            false => output.push(mi),
            true => {
                let copy = mi.inst.take_copy().unwrap();
                debug_assert!(copy.vcd_trgt(), "We should only be resoloving Trgt sourced Copys here");
                let o_start = copy.ssp + copy.u_pos as u64; //ssp is o_pos, u is offset from that.
                let mut resolved = get_exact_slice(cur_output, o_start, copy.len_in_u()).unwrap();
                output.append(&mut resolved);
            },
        }
    }
    output
}
fn global_deref_inst(inst: &VcdExtract)->bool{
    match &inst.inst {
        ExInstType::Copy(copy ) if copy.vcd_trgt() => {
            debug_assert!(copy.in_s(), "We should have resolved all Local Copys by now");
           true
        },
        _ => false,
    }
}
// fn deref_nonq(source:&[MInst],copy:&DisCopy)->Vec<MInst>{
//     debug_assert!(copy.in_s(), "We should have resolved all Local Copys by now");
//     //since we know we are in S either way
//     //our o position should be ssp + u_pos
//     let o_start = copy.ssp + copy.u_pos as u64; //ssp is o_pos, u is offset from that.
//     get_exact_slice(source, o_start, copy.len_in_u()).unwrap()
// }

// pub fn translate_local_only(extracted:Vec<VcdExtract>)->Vec<MInst>{
//     let mut output:Vec<MInst> = Vec::with_capacity(extracted.len());
//     for DisInst { o_pos_start, inst } in extracted {
//         match inst {
//             ExtractedInst::Copy(copy) if !copy.in_s() => {
//                 //here we can have a reg copy in T or a seq
//                 //since we know we are in T either way
//                 //our o position should be u_pos - sss
//                 let o_start = (copy.u_pos - copy.sss) as u64;
//                 if copy.is_implicit_seq(){
//                     let slice_len = copy.len_in_u() - copy.len_in_o();
//                     let slice = get_exact_slice(output.as_slice(), o_start, slice_len).unwrap();
//                     let seq = CopyQ { inst:slice, len: copy.len_in_o(), skip: 0, trunc: 0 };
//                     output.push(MInst { o_pos_start, inst: ExInstType::Copy(MergeCopy::PredecessorSeq(seq)) });
//                 }else{
//                     let mut slice = get_exact_slice(output.as_slice(), o_start, copy.len_in_u()).unwrap();
//                     output.append(&mut slice);
//                 }
//             },
//             a => {
//                 let inst = match a {
//                     ExtractedInst::Run(r) => ExInstType::Run(r),
//                     ExtractedInst::Add(a) => ExInstType::Add(a),
//                     ExInstType::Copy(c) if !c.vcd_trgt() => {
//                         debug_assert!(c.in_s(), "We should have resolved all Local Copys by now");
//                         ExInstType::Copy(MergeCopy::SourceCopy(c))
//                     }
//                     ExInstType::Copy(c) => {
//                         debug_assert!(c.vcd_trgt() && c.in_s(), "At most this should require Global translation");
//                         ExInstType::Copy(MergeCopy::PredecessorSO(c))
//                     },
//                 };
//                 output.push(DisInst { o_pos_start, inst })
//             },
//         }
//     }
//     output
// }
// enum ResolveDecision{
//     NoOp,
//     ResolveNonQ,
//     ResolveQ,
// }
// fn global_deref_cntl(inst: &MInst)->ResolveDecision{
//     match &inst.inst {
//         ExInstType::Copy(MergeCopy::PredecessorSO(copy)) => {
//             debug_assert!(copy.vcd_trgt(), "We should only be resoloving Trgt sourced Copys here");
//             debug_assert!(copy.in_s(), "We should have resolved all Sourced Copys by now");
//             ResolveDecision::ResolveNonQ
//         },
//         ExInstType::Copy(MergeCopy::PredecessorSeq(CopyQ { .. })) => {
//             ResolveDecision::ResolveQ
//         },
//         _ => ResolveDecision::NoOp,
//     }
// }
// fn global_deref_q(cur_output:&[MInst],seq:CopyQ)->CopyQ{
//     //this sequence should be S normalized already
//     //we just need to find the parent slice in O
//     //We treat all the Copys found based on their own vcd_trgt value
//     let CopyQ { inst, len, skip, trunc } = seq;
//     let mut output = Vec::with_capacity(inst.len());
//     for mi in inst {
//         match global_deref_cntl(&mi){
//             ResolveDecision::NoOp => output.push(mi),
//             ResolveDecision::ResolveNonQ => {
//                 let copy = mi.inst.take_copy().unwrap().take_so();
//                 debug_assert!(copy.vcd_trgt(), "We should only be resoloving Trgt sourced Copys here");
//                 let mut resolved = deref_nonq(cur_output, &copy);
//                 output.append(&mut resolved);
//             },
//             ResolveDecision::ResolveQ => {
//                 let DisInst { o_pos_start, inst } = mi;
//                 let resolved = global_deref_q(cur_output,inst.take_copy().unwrap().take_seq());
//                 output.push(DisInst { o_pos_start, inst:ExInstType::Copy( MergeCopy::PredecessorSeq(resolved)) });
//             },
//         }
//     }
//     CopyQ { inst: output, len, skip, trunc }
// }
// pub fn translate_global_only(locally_resolved:Vec<MInst>)->Vec<MInst>{
//     let mut output = Vec::with_capacity(locally_resolved.len());
//     for mi in locally_resolved {
//         match global_deref_cntl(&mi){
//             ResolveDecision::NoOp => output.push(mi),
//             ResolveDecision::ResolveNonQ => {
//                 let mut resolved = deref_nonq(output.as_slice(), &mi.inst.take_copy().unwrap().take_so());
//                 output.append(&mut resolved);
//             },
//             ResolveDecision::ResolveQ => {
//                 let DisInst { o_pos_start, inst } = mi;
//                 let resolved = global_deref_q(&output,inst.take_copy().unwrap().take_seq());
//                 output.push(DisInst { o_pos_start, inst:ExInstType::Copy(MergeCopy::PredecessorSeq(resolved) ) });
//             },
//         }
//     }
//     output
// }

fn find_copy_s(extract:&[VcdExtract],shift:usize,dest:&mut Vec<usize>){
    for (i,ext) in extract.iter().enumerate(){
        match &ext.inst{
            ExInstType::Copy(c) if c.in_s() && !c.vcd_trgt() => dest.push(i+shift),
            _ => (),
        }
    }
}


pub struct Merger{
    ///The summary patch that will be written to the output.
    terminal_patch: Vec<VcdExtract>,
    ///If this is empty, merging a patch will have no effect.
    ///These are where TerminalInst::CopySS are found.
    terminal_copy_indices: Vec<usize>,
}

impl Merger {
    pub fn new<R:Read + Seek>(terminal_patch:R) -> std::io::Result<Result<Merger,SummaryPatch>> {
        let (terminal_patch,stats) = extract_patch_instructions(terminal_patch)?;
        dbg!(stats);
        if stats.copy_bytes == 0{
            return Ok(Err(SummaryPatch(terminal_patch)));
        }
        let mut terminal_copy_indices = Vec::new();
        //we for sure need to translate local. I think translate global isn't needed??
        //will need to check this.
        let terminal_patch = translate_and_deref_pred(terminal_patch);
        find_copy_s(&terminal_patch,0,&mut terminal_copy_indices);
        debug_assert!(!terminal_copy_indices.is_empty(), "terminal_copy_indices should not be empty");
        dbg!(&terminal_patch);
        Ok(Ok(Merger{terminal_patch,terminal_copy_indices}))
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
            predecessor_patch = translate_and_deref_pred(predecessor_patch);
        }
        dbg!(&predecessor_patch);
        dbg!(&self.terminal_patch);
        let mut terminal_copy_indices = Vec::with_capacity(self.terminal_copy_indices.len());
        let mut inserts = Vec::with_capacity(self.terminal_copy_indices.len());
        let mut shift = 0;
        for i in self.terminal_copy_indices{
            let DisInst { inst,.. } = self.terminal_patch[i].clone();
            //dbg!(i);
            let copy = inst.take_copy().expect("Expected Copy");
            //this a src window copy that we need to resolve from the predecessor patch.
            debug_assert!(copy.in_s());
            debug_assert!(!copy.vcd_trgt());
            let o_start = copy.ssp + copy.u_pos as u64; //ssp is o_pos, u is offset from that.
            dbg!(o_start,copy.len_in_u(),&copy);
            let resolved = get_exact_slice(&predecessor_patch, o_start, copy.len_in_u()).unwrap();
            dbg!(&resolved);
            find_copy_s(&resolved, i+shift, &mut terminal_copy_indices);
            shift += resolved.len() - 1;
            //dbg!(&terminal_copy_indices);
            inserts.push((i, resolved));
        }
        //now we expand the old copy values with the derefd instructions.
        self.terminal_patch = expand_elements(self.terminal_patch, inserts);
        dbg!(&self.terminal_patch);
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
///This is returned when a terminal patch has no CopySS instructions.
///Merging additional patches will have no effect.
#[derive(Debug)]
pub struct SummaryPatch(Vec<VcdExtract>);
impl SummaryPatch{
    pub fn write<W:Write>(self,sink:W,max_u_size:Option<usize>)->std::io::Result<W>{
        let max_u_size = max_u_size.unwrap_or(1<<28); //256MB
        let header = Header::default();
        let encoder = VCDEncoder::new(sink,header)?;
        let mut state = MergeState{
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

pub enum EarlyReturn{
    NoMoreCopyInst(SummaryPatch),
    CanContinue(Merger),
}

impl DisCopy{
    // fn src_o_pos(&self)->u64{
    //     self.ssp + self.u_pos as u64
    // }
    // fn max_u_trunc_amt(&self,max_space_avail:u32)->u32{
    //     //can we figure out how much to truncate to fit in the space?
    //     //every change in len, also shrinks the sss
    //     let min = self.min_src();
    //     let cur_s_len = min.end - min.start;
    //     let cur_u_len = self.len_u + cur_s_len as u32;
    //     if cur_u_len <= max_space_avail {
    //         return 0;
    //     }else{
    //         (cur_u_len - max_space_avail) / 2
    //     }
    // }
    // fn to_output_copy(self,cur_win_ssp:u64)->COPY{
    //     let DisCopy { u_pos, len_u, ssp, vcd_trgt, copy_type,.. } = self;
    //     if cur_win_ssp > ssp{
    //         let neg_shift = cur_win_ssp - ssp;
    //         COPY { len:len_u, u_pos: u_pos - neg_shift as u32 }
    //     }else{
    //         let pos_shift = ssp - cur_win_ssp;
    //         COPY { len:len_u, u_pos: u_pos + pos_shift as u32 }
    //     }
    // }
    // fn split_at(self,first_inst_len:u32)->(DisCopy,DisCopy){
    //     let DisCopy { u_pos, len_u, ssp, vcd_trgt, copy_type,sss  } = self;
    //     debug_assert!(first_inst_len > 0, "first_inst_len should be > 0");
    //     debug_assert!(first_inst_len < len_u -1, "first_inst_len should be < len_in_o -1");
    //     if matches!(copy_type, CopyType::CopyQ{..}) {
    //         panic!("Cannot split a CopyQ");
    //     }
    //     let first = DisCopy { u_pos, len_u: first_inst_len, ssp, vcd_trgt, copy_type,sss };
    //     let remainder = len_u - first_inst_len;
    //     let second = DisCopy { u_pos: u_pos + first_inst_len, len_u: remainder, ssp, vcd_trgt, copy_type,sss };
    //     (first,second)
    // }
}

struct MergeState<W>{
    cur_o_pos: u64,
    max_u_size: usize,
    cur_win: Vec<ExInstType<DisCopy>>,
    win_sum: Option<WindowHeader>,
    sink: VCDEncoder<W>,
}

impl<W:Write> MergeState<W> {
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
            let split_at = ci.will_fit_window((self.max_u_size - self.current_u_size()) as u32);
            match split_at {
                Some(len) =>{
                    debug_assert!(len < ci.len_in_o() as u32, "split at: {} len: {}",len,ci.len_in_o());
                    let (first,second) = ci.split_at(len);
                    self.add_to_window(first);
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
        let ind = self.win_sum.as_ref().expect("win_sum should be set here").win_indicator.clone();

        match (&ind,inst.inst_type().comp_indicator(&ind)){
            (_, None) => (),
            (WinIndicator::Neither, Some(set)) => {
                self.win_sum.as_mut().expect("win_sum should be set here").win_indicator = set;
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

// pub fn merge_patches<R: Read + Seek, W:Write>(sequential_patches: Vec<R>>,sink:W,max_u_size:Option<usize>,) -> std::io::Result<W> {
//     let max_u_size = max_u_size.unwrap_or(1<<28); //256MB
//     let header = Header::default();
//     let encoder = VCDEncoder::new(sink,header)?;
//     let mut state = MergeState{
//         cur_o_pos: 0,
//         cur_win: Vec::new(),
//         sink: encoder,
//         sequential_patches,
//         max_u_size,
//         win_sum: None,
//     };
//     let cntl_patch = state.sequential_patches.len()-1;
//     // we loop through the instructions of the last patch in our main loop
//     // when we find a copy, we start a copy resolution that will recursively resolve the Copy back though the patches
//     let mut input = [Inst::Run(RUN { len: 0,byte:0 }); 1];
//     let mut resolution_buffer = Vec::new();
//     while let Some(SparseInst { o_start, inst }) = state.next()? {
//         debug_assert!(inst.len() > 0, "{:?}",inst);
//         let expected_end = o_start + inst.len() as u64;
//         input[0] = inst;
//         resolve_list_of_inst(&mut state.sequential_patches,cntl_patch, &input, &mut resolution_buffer)?;
//         //dbg!(&resolution_buffer);
//         for inst in resolution_buffer.drain(..){
//             debug_assert!(inst.inst.len() > 0, "inst.len() == 0");
//             state.apply_instruction(inst)?;
//         }
//         debug_assert_eq!(state.cur_o_pos, expected_end, "cur_o_pos: {} expected_end: {}", state.cur_o_pos, expected_end);
//     }

//     state.flush_window()?;
//     state.sink.finish()
// }
// fn resolve_list_of_inst<R: Read + Seek>(
//     patches:&mut [R>],
//     patch_index:usize,
//     list:&[Inst],
//     output:&mut Vec<MergedInst>
// )->std::io::Result<()>{
//     //the list might already be resolved (no copy/seq)
//     //if it is, we just add it to the output
//     //else we resolve the copy/seq and add the resolved inst to the output
//     for inst in list{
//         let len = inst.len();
//         if inst.is_copy(){
//             let copy = inst.clone().to_copy();
//             match copy{
//                 CopyInst::Copy(copy) => {
//                     if patch_index == 0{ //we are emitting a copy from the first patch
//                         let ci = MergedInst { inst: RootInst::Copy(copy), patch_index };
//                         output.push(ci);
//                     }else{//this copy references some earlier patch output, we must resolve it.
//                         let src_o_pos = copy.src_o_pos();
//                         let next_patch_index = patch_index - 1;
//                         let next_slice = patches[next_patch_index].exact_slice(src_o_pos, len)?;
//                         //dbg!(&next_slice);
//                         resolve_list_of_inst(patches,next_patch_index,next_slice.as_slice(),output)?;
//                     }
//                 },
//                 CopyInst::Seq(CopyQ { inst, len, skip, trunc }) => {
//                     let mut inner_out = Vec::new();
//                     resolve_list_of_inst(patches,patch_index,inst.as_slice(),&mut inner_out)?;
//                     let effective_len = len - (skip + trunc);
//                     flatten_and_trim_sequence(&inner_out, skip, effective_len, output)
//                 },
//             }
//         }else{
//             let ci = MergedInst { inst: inst.clone().to_root(), patch_index };
//             output.push(ci);
//         }
//     }
//     Ok(())
// }
fn expand_sequence(seq:&[VcdExtract],len:u32,mut cur_o_pos:u64,output:&mut Vec<VcdExtract>) {
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
            output.push(DisInst { o_pos_start:cur_o_pos, inst: modified_instruction });
            current_len += inst_len;
            cur_o_pos += inst_len as u64;
            // If we've reached or exceeded the effective length, break out of the loop
            if current_len >= len {
                debug_assert_eq!(current_len, len, "current_len: {} len: {}", current_len, len);
                break;
            }
        }
    }
}
// fn flatten_and_trim_sequence(seq:&[MInst],skip:u32,len:u32,output:&mut Vec<VcdExtract>) {
//     let mut current_len = 0;
//     let mut skipped_bytes = 0;

//     // Calculate the effective length after considering truncation
//     let effective_len = len;
//     while current_len < effective_len {
//         for DisInst { o_pos_start, inst } in seq.iter().cloned() {
//             let mut modified_instruction = inst;
//             let end_pos = current_len + modified_instruction.len_in_o();
//             if skipped_bytes < skip {
//                 if skipped_bytes + modified_instruction.len_in_o() > skip {
//                     // We're in the middle of an instruction, need to apply skip
//                     modified_instruction.skip(skip - skipped_bytes);
//                     skipped_bytes += modified_instruction.len_in_o(); // Update skipped_bytes to reflect the adjusted instruction

//                 } else {
//                     // Entire instruction is skipped
//                     skipped_bytes += modified_instruction.len_in_o();
//                     continue;
//                 }
//             } else {
//                 // After skip has been accounted for, directly clone and potentially modify for truncation
//                 // If adding this instruction would exceed the effective length, apply truncation
//                 if end_pos > effective_len {
//                     let trunc_amt = end_pos - effective_len;
//                     modified_instruction.trunc(trunc_amt);
//                 }
//             }
//             current_len += modified_instruction.len_in_o();
//             skipped_bytes += modified_instruction.len_in_o(); // Update skipped_bytes to reflect the adjusted instruction
//             match modified_instruction{
//                 ExInstType::Copy(MergeCopy::PredecessorSeq(q)) => {
//                     flatten_and_trim_sequence(&q.inst, q.skip, q.len, output);
//                 },
//                 ExInstType::Copy(a)=> output.push(DisInst { o_pos_start, inst: ExInstType::Copy(a.take_discopy()) }),
//                 ExInstType::Add(a) => output.push(DisInst { o_pos_start, inst: ExInstType::Add(a) }),
//                 ExInstType::Run(a) => output.push(DisInst { o_pos_start, inst: ExInstType::Run(a) }),
//             }
//             // If we've reached or exceeded the effective length, break out of the loop
//             if current_len >= effective_len {
//                 break;
//             }
//         }
//     }
// }



// #[derive(Clone, Debug, PartialEq, Eq)]
// struct MergedInst{
//     inst: RootInst,
//     patch_index: usize,
// }
// #[derive(Clone, Debug, PartialEq, Eq)]
// enum RootInst{
//     Add(ADD),
//     Run(RUN),
//     Copy(DisCopy),
// }

// impl RootInst {
//     fn src_range(&self)->Option<Range<u64>>{
//         match self{
//             RootInst::Copy(copy) => Some(copy.min_src()),
//             _ => None,
//         }
//     }
//     fn len(&self)->u32{
//         match self{
//             RootInst::Add(add) => add.len(),
//             RootInst::Run(run) => run.len(),
//             RootInst::Copy(copy) => copy.copy.len_in_u(),
//         }
//     }
//     fn skip(&mut self,skip:u32){
//         match self{
//             RootInst::Copy(copy) => copy.copy.skip(skip),
//             RootInst::Add(add) => add.skip(skip),
//             RootInst::Run(run) => run.skip(skip),
//         }
//     }
//     fn trunc(&mut self,amt:u32){
//         match self{
//             RootInst::Copy(copy) => copy.copy.trunc(amt),
//             RootInst::Add(add) => add.trunc(amt),
//             RootInst::Run(run) => run.trunc(amt),
//         }

//     }
//     fn max_u_trunc_amt(&self,max_space_avail:u32)->u32{
//         if max_space_avail >= self.len() || max_space_avail == 0 {return 0}
//         match self{
//             RootInst::Copy(copy) => copy.max_u_trunc_amt(max_space_avail),
//             a => a.len() - max_space_avail,

//         }
//     }
// }


// impl Inst {
//     fn is_copy(&self) -> bool {
//         matches!(self, Inst::Copy{..} | Inst::Sequence(_) )
//     }
//     fn to_copy(self) ->CopyInst{
//         match self{
//             Inst::Copy(c) => CopyInst::Copy(c),
//             Inst::Sequence(seq) => CopyInst::Seq(seq),
//             _ => panic!("Expected Copy or Sequence, got {:?}", self),
//         }
//     }
//     fn to_root(self) -> RootInst{
//         match self{
//             Inst::Add(add) => RootInst::Add(add),
//             Inst::Run(run) => RootInst::Run(run),
//             Inst::Copy(c) => RootInst::Copy(c),
//             _ => panic!("Expected Add, Run, or Copy, got {:?}", self),
//         }
//     }
//     // fn skip(&mut self, amt: u32) {
//     //     if amt == 0 {return}
//     //     match self {

//     //         Self::Sequence(inst) => {inst.skip(amt);},
//     //         Self::Add(inst) => {inst.skip(amt);},
//     //         Self::Run(inst) => {inst.skip(amt);},
//     //         Self::Copy (copy) => {copy.copy.skip(amt);},
//     //     }
//     // }
//     // fn trunc(&mut self, amt: u32) {
//     //     if amt == 0 {return}
//     //     match self {
//     //         Self::Sequence(inst) => {inst.trunc += amt;},
//     //         Self::Add(inst) => {inst.trunc(amt);},
//     //         Self::Run(inst) => {inst.trunc(amt);},
//     //         Self::Copy (copy) => {copy.copy.trunc(amt);},
//     //     }
//     // }
// }




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
    use crate::{applicator::apply_patch, decoder::VCDDecoder, reader::{DeltaIndicator, VCDReader}, RUN};

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
        //let add_run = add_run_patch();
        let comp = complex_patch();
        let copy = copy_patch();
        let seq = seq_patch();
        let merger = Merger::new(seq).unwrap().unwrap();
        let merger = merger.merge(copy).unwrap().unwrap();
        let merger = merger.merge(comp).unwrap().unwrap();
        /*
        So each patch only considers the src to be '01234'
        So when we string patches together, we can only look at the first 5 bytes of the prev input
        My prev impl somehow worked per the notes:
        01234 Add-> A12XXXYZ34 Compl YZZXYZ12XX -> Copy YZZXYYZZXY -> Seq YZXYZXYZXY
        However the new impl won't work that way:
        01234 -> Compl YZZ4YZ1234 -> Copy YZZ4YYZZ4Y -> Seq YZ4YZ4YZX4
        We cannot merge the Add for some reason.
        There must be an error some where, since the '4' should be 'X'
        That 'X' should be there from the Add/Run patch.
        */
        //let merger = merger.merge(add_run).unwrap().unwrap();
        let merged_patch = merger.finish().write(Vec::new(), None).unwrap();
        let mut cursor = Cursor::new(merged_patch);
        let mut output = Vec::new();
        apply_patch(&mut cursor, Some(Cursor::new(SRC.to_vec())), &mut output).unwrap();
        //print output as a string
        let as_str = std::str::from_utf8(&output).unwrap();
        println!("{}",as_str);
        assert_eq!(output,answer);
    }
    // #[test]
    // fn test_kitchen_sink(){
    //     //"hello" -> "hello world!" -> "Hello! Hello! Hello. hello. hello..."
    //     //we need to use a series of VCD_TARGET windows and Sequences across multiple patches
    //     //we should use copy/seq excessively since add/run is simple in the code paths.
    //     let src = b"hello!";
    //     let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
    //     encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(5), source_segment_position: Some(0), size_of_the_target_window:5 , delta_indicator: DeltaIndicator(0) }).unwrap();
    //     // Instructions
    //     encoder.next_inst(EncInst::COPY(COPY { len: 5, u_pos: 0 })).unwrap();
    //     encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(1), source_segment_position: Some(4), size_of_the_target_window:6 , delta_indicator: DeltaIndicator(0) }).unwrap();
    //     encoder.next_inst(EncInst::ADD(" w".as_bytes().to_vec())).unwrap();
    //     encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 0 })).unwrap();
    //     encoder.next_inst(EncInst::ADD("rld".as_bytes().to_vec())).unwrap();
    //     encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(1), source_segment_position: Some(5), size_of_the_target_window:1 , delta_indicator: DeltaIndicator(0) }).unwrap();
    //     encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 0 })).unwrap();
    //     let p1 = encoder.finish().unwrap().into_inner();
    //     let p1_answer = b"hello world!";
    //     let mut cursor = Cursor::new(p1.clone());
    //     let mut output = Vec::new();
    //     apply_patch(&mut cursor, Some(Cursor::new(src.to_vec())), &mut output).unwrap();
    //     assert_eq!(output,p1_answer); //ensure our instructions do what we think they are.
    //     let patch_1 = make_patch_reader(p1);
    //     let mut encoder = VCDEncoder::new(Cursor::new(Vec::new()), HDR).unwrap();
    //     encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(11), source_segment_position: Some(1), size_of_the_target_window:7 , delta_indicator: DeltaIndicator(0) }).unwrap();
    //     encoder.next_inst(EncInst::ADD("H".as_bytes().to_vec())).unwrap();
    //     encoder.next_inst(EncInst::COPY(COPY { len: 4, u_pos: 0 })).unwrap(); //ello
    //     encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 10 })).unwrap(); //'!'
    //     encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 4 })).unwrap(); //' '
    //     encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(7), source_segment_position: Some(0), size_of_the_target_window:14 , delta_indicator: DeltaIndicator(0) }).unwrap();
    //     encoder.next_inst(EncInst::COPY(COPY { len: 19, u_pos: 0 })).unwrap(); //Hello! Hello! Hello
    //     encoder.next_inst(EncInst::ADD(".".as_bytes().to_vec())).unwrap();
    //     encoder.next_inst(EncInst::COPY(COPY { len: 1, u_pos: 13 })).unwrap(); // ' '
    //     encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(6), source_segment_position: Some(15), size_of_the_target_window:7, delta_indicator: DeltaIndicator(0) }).unwrap();
    //     encoder.next_inst(EncInst::ADD("h".as_bytes().to_vec())).unwrap();
    //     encoder.next_inst(EncInst::COPY(COPY { len: 6, u_pos: 0 })).unwrap(); //'ello. '
    //     encoder.start_new_win(WindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(6), source_segment_position: Some(21), size_of_the_target_window:8 , delta_indicator: DeltaIndicator(0) }).unwrap();
    //     encoder.next_inst(EncInst::COPY(COPY { len: 6, u_pos: 0 })).unwrap(); //'hello.'
    //     encoder.next_inst(EncInst::COPY(COPY { len: 3, u_pos: 11 })).unwrap(); //Seq '.' == Run(3) '.'
    //     let p2 = encoder.finish().unwrap().into_inner();
    //     let p2_answer = b"Hello! Hello! Hello. hello. hello...";
    //     let mut cursor = Cursor::new(p2.clone());
    //     let mut output = Vec::new();
    //     apply_patch(&mut cursor, Some(Cursor::new(p1_answer.to_vec())), &mut output).unwrap();
    //     assert_eq!(output,p2_answer);
    //     let patch_2 = make_patch_reader(p2);
    //     let merged_patch = merge_patches(vec![patch_1,patch_2],Vec::new(),None).unwrap();
    //     let mut cursor = Cursor::new(merged_patch);
    //     let mut output = Vec::new();
    //     let answer = b"Hello! Hello! Hello. hello. hello...";
    //     apply_patch(&mut cursor, Some(Cursor::new(src.to_vec())), &mut output).unwrap();
    //     //print output as a string
    //     let as_str = std::str::from_utf8(&output).unwrap();
    //     println!("{}",as_str);
    //     assert_eq!(output,answer);
    // }
    // #[test]
    // fn insert_single_element() {
    //     let target = vec![1, 2, 3];
    //     let inserts = vec![(1, vec![4])];
    //     let result = expand_elements(target, inserts);
    //     assert_eq!(result, vec![1, 4, 3]);
    // }

    // #[test]
    // fn insert_multiple_elements_at_different_positions() {
    //     let target = vec![1, 2, 3];
    //     let inserts = vec![(1, vec![4, 5]), (2, vec![6, 7])];
    //     let result = expand_elements(target, inserts);
    //     assert_eq!(result, vec![1, 4, 5, 6, 7]);
    // }

    // #[test]
    // fn insert_at_beginning_and_end() {
    //     let target = vec![2, 3, 4];
    //     let inserts = vec![(0, vec![1]), (3, vec![5])];
    //     let result = expand_elements(target, inserts);
    //     assert_eq!(result, vec![1, 3, 4, 5]);
    // }

    // #[test]
    // fn insert_with_no_elements() {
    //     let target = vec![1, 2, 3];
    //     let inserts = Vec::new();
    //     let result = expand_elements(target, inserts);
    //     assert_eq!(result, vec![1, 2, 3]);
    // }
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
        expand_sequence(&seq, 15, 0,&mut output);
        let result = vec![
            VcdExtract { o_pos_start: 0, inst: ExInstType::Copy(DisCopy { u_pos: 0, len_u: 2, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            VcdExtract { o_pos_start: 2, inst: ExInstType::Copy(DisCopy { u_pos: 4, len_u: 6, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            VcdExtract { o_pos_start: 8, inst: ExInstType::Copy(DisCopy { u_pos: 12, len_u: 4, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            VcdExtract { o_pos_start: 12, inst: ExInstType::Copy(DisCopy { u_pos: 0, len_u: 2, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
            VcdExtract { o_pos_start: 14, inst: ExInstType::Copy(DisCopy { u_pos: 4, len_u: 1, sss: 0, ssp: 0, vcd_trgt: false, copy_type: CopyType::CopyS }) },
        ];
        assert_eq!(output, result, "Output should contain a truncated instruction");
    }

    // #[test]
    // fn test_skip_sequence() {
    //     let inst = MergedInst { inst: RootInst::Add(ADD {len:5, p_pos:0 }), patch_index: 0 };
    //     let seq = vec![inst];
    //     let mut output = Vec::new();
    //     flatten_and_trim_sequence(&seq, 3, 7, &mut output);
    //     let result = vec![MergedInst { inst: RootInst::Add(ADD {len:2, p_pos:3 }), patch_index: 0 },inst];
    //     assert_eq!(output, result, "Output should contain two copies of the instruction");
    // }

    // #[test]
    // fn test_truncate_sequence() {
    //     let seq = vec![MergedInst { inst: RootInst::Add(ADD { len: 10, p_pos:0 }), patch_index: 0 }];
    //     let mut output = Vec::new();
    //     flatten_and_trim_sequence(&seq, 0, 5, &mut output);
    //     let result = vec![MergedInst { inst: RootInst::Add(ADD { len: 5, p_pos:0 }), patch_index: 0 }];
    //     assert_eq!(output, result, "Output should contain a truncated instruction");
    // }

    // #[test]
    // fn test_skip_and_trim() {
    //     let seq = vec![
    //         MergedInst { inst: RootInst::Add(ADD { len: 3, p_pos: 0 }), patch_index: 0 },
    //         MergedInst { inst: RootInst::Run(RUN {len:4, byte: 0 }), patch_index: 0 },
    //         MergedInst { inst: RootInst::Copy(DisCopy {len_u: 5, u_pos: 0, sss: 5, ssp: 0 }), patch_index: 0 },
    //     ];
    //     let mut output = Vec::new();
    //     flatten_and_trim_sequence(&seq, 3, 14, &mut output);
    //     let result = vec![
    //         MergedInst { inst: RootInst::Run(RUN {len:4, byte: 0 }), patch_index: 0 },
    //         MergedInst { inst: RootInst::Copy(DisCopy {copy:COPY{len: 5, u_pos: 0 }, sss: 5, ssp: 0 }), patch_index: 0 },
    //         MergedInst { inst: RootInst::Add(ADD { len: 3, p_pos: 0 }), patch_index: 0 },
    //         MergedInst { inst: RootInst::Run(RUN {len:2, byte: 0 }), patch_index: 0 },
    //     ];
    //     assert_eq!(output, result, "Output should contain a truncated instruction");
    // }
    // #[test]
    // fn test_trunc_and_trim() {
    //     let seq = vec![
    //         MergedInst { inst: RootInst::Add(ADD { len: 3, p_pos: 0 }), patch_index: 0 },
    //         MergedInst { inst: RootInst::Run(RUN {len:4, byte: 0 }), patch_index: 0 },
    //         MergedInst { inst: RootInst::Copy(DisCopy {copy:COPY{len: 5, u_pos: 0 }, sss: 5, ssp: 0 }), patch_index: 0 },
    //     ];
    //     let mut output = Vec::new();
    //     flatten_and_trim_sequence(&seq, 3, 12, &mut output);
    //     let result = vec![
    //         MergedInst { inst: RootInst::Run(RUN {len:4, byte: 0 }), patch_index: 0 },
    //         MergedInst { inst: RootInst::Copy(DisCopy {copy:COPY{len: 5, u_pos: 0 }, sss: 5, ssp: 0 }), patch_index: 0 },
    //         MergedInst { inst: RootInst::Add(ADD { len: 3, p_pos: 0 }), patch_index: 0 },
    //     ];
    //     assert_eq!(output, result, "Output should contain a truncated instruction");
    // }
}