use std::io::{Read, Seek};

use crate::{reader::{VCDReader, VCDiffReadMsg, WindowSummary}, Cache, CodeTableEntry, TableInst, ADD, COPY, RUN, VCD_C_TABLE};


///Decoded instruction
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecInst{
    Add(ADD),
    Copy(COPY),
    Run(RUN)
}
impl DecInst{
    pub fn len(&self)->usize{
        (match self{
            DecInst::Add(a) => a.len,
            DecInst::Copy(c) => c.len,
            DecInst::Run(r) => r.len
        }) as usize
    }
    pub fn is_copy(&self)->bool{
        matches!(self, DecInst::Copy(_))
    }
    pub fn take_copy(self) -> COPY{
        match self {
            DecInst::Copy(c) => c,
            DecInst::Add(_) |
            DecInst::Run(_) => panic!("Not a COPY instruction!"),
        }
    }
}



///This works as a decoder for the VCDIFF format.
///It can only advance forward in the output stream.
///You simply ask for the byte we are interrogating.
///It will return the DecodedInstruction that control that byte in the output stream.
#[derive(Debug)]
pub struct VCDDecoder<R> {
    caches: Cache,
    ///The monotonic Reader for scanning the VCDIFF file
    reader: VCDReader<R>,
    ///Where we are in the address section
    addr_pos: u64,
    ///Where we are in the data section
    data_pos: u64,
    cur_window: usize,
    ///Are we currently in a window?
    in_window: bool,
    ///This is global in the target output
    ///That is, it cuts across window boundaries
    ///This increments after we read any single DecodedInstruction
    cur_o_position: u64,
    ///This is our position in string U, it starts at SourceSegmentSize and increments as we read instructions
    ///Giving us our current position in U
    cur_u_position: u64,
}
impl<R: Read + Seek> VCDDecoder<R> {
    pub fn new(reader: VCDReader<R>) -> Self {
        VCDDecoder {
            caches: Cache::new(),
            reader,
            addr_pos: 0,
            data_pos: 0,
            cur_window: 0,
            in_window: false,
            cur_o_position: 0,
            cur_u_position: 0,
        }
    }
    pub fn reader(&mut self) -> &mut VCDReader<R> {
        &mut self.reader
    }
    pub fn position(&self) -> u64 {
        self.cur_o_position
    }
    pub fn u_position(&self) -> u64 {
        self.cur_u_position
    }
    pub fn cur_win_index(&self) -> usize {
        self.cur_window
    }
    fn handle_inst(&mut self, inst: TableInst) -> std::io::Result<Option<DecInst>> {
        Ok(match inst{
            TableInst::Run => {
                //read the length of the run
                let len = self.reader.read_as_inst_size_unchecked()?;
                let mut byte = [0u8];
                self.reader.read_from_src(self.data_pos, &mut byte)?;
                self.data_pos += 1;
                self.cur_o_position += len;
                self.cur_u_position += len;

                let inst = DecInst::Run(RUN { len: len as u32, byte: byte[0] });
                Some(inst)
            },
            TableInst::Add { size } => {
                let len = if size == 0 {
                    self.reader.read_as_inst_size_unchecked()?
                }else{size as u64};
                let pos = self.data_pos;
                self.data_pos += len;
                self.cur_o_position += len;
                self.cur_u_position += len;

                let inst = DecInst::Add(ADD { len: len as u32, p_pos: pos });
                Some(inst)
            },
            TableInst::Copy { size, mode } => {
                let len = if size == 0 {
                    self.reader.read_as_inst_size_unchecked()?
                }else{size as u64};
                //is 'here' u_pos or o_pos? Should be u, as a COPY cannot apply to a different window directly
                //so this should be changed?
                //TODO figure out a test to see what this should be.
                let (addr,read) = self.caches.addr_decode(self.reader.get_reader(self.addr_pos)?, self.cur_u_position, mode as usize)?;
                self.addr_pos += read as u64;
                self.cur_o_position += len;
                self.cur_u_position += len;

                let inst = DecInst::Copy(COPY { len: len as u32, u_pos: addr as u32 });
                Some(inst)
            },
            TableInst::NoOp => None,
        })
    }
    pub fn next(&mut self) -> std::io::Result<VCDiffDecodeMsg> {
        match self.reader.next()?{
            VCDiffReadMsg::WindowSummary(ws) => {
                assert!(!self.in_window, "We got a window summary while we are still in a window");
                assert_eq!(ws.delta_indicator.to_u8(), 0, "Secondary Compression is not currently supported");
                self.in_window = true;
                self.cur_window += 1;
                self.addr_pos = ws.addr_sec_start();
                self.data_pos = ws.data_sec_start();
                self.cur_u_position = ws.source_segment_size.unwrap_or(0);
                Ok(VCDiffDecodeMsg::WindowSummary(ws))
            },
            VCDiffReadMsg::InstSecByte(code) => {
                //this is the only message that can advance our position
                if self.in_window{
                    let o_start = self.cur_o_position;
                    let CodeTableEntry{ first, second } = VCD_C_TABLE[code as usize];
                    let f = if let Some(inst) = self.handle_inst(first)? {
                        inst
                    }else{
                        panic!("NoOp is not allowed in the first position of an opcode"); //? maybe in an app specific table? Seems illogical yet
                    };
                    let s = self.handle_inst(second)?;

                    Ok(VCDiffDecodeMsg::Inst{o_start,first: f, second: s})
                }else{
                    panic!("We got an opcode without a window summary");
                }

            },
            VCDiffReadMsg::EndOfWindow => {
                if !self.in_window {
                    panic!("We got an end of window without a window summary");
                }else{
                    self.in_window = false;
                    Ok(VCDiffDecodeMsg::EndOfWindow)
                }
            }
            VCDiffReadMsg::EndOfFile => Ok(VCDiffDecodeMsg::EndOfFile),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VCDiffDecodeMsg{
    WindowSummary(WindowSummary),
    Inst{o_start:u64, first: DecInst, second: Option<DecInst>},
    EndOfWindow,
    EndOfFile,
}


#[cfg(test)]
mod test_super {

    use crate::reader::{DeltaIndicator, WinIndicator};

    use super::*;

    #[test]
    fn test_decode() {
        //'hello' -> 'Hello! Hello!'
        let insts = [
            VCDiffDecodeMsg::WindowSummary(WindowSummary{

                source_segment_size: Some(4),
                source_segment_position: Some(1),
                length_of_the_delta_encoding: 12,
                size_of_the_target_window: 13,
                delta_indicator: DeltaIndicator::from_u8(0),
                length_of_data_for_adds_and_runs: 3,
                length_of_instructions_and_sizes: 2,
                length_of_addresses_for_copys: 2,
                win_start_pos: 5,
                win_indicator: WinIndicator::VCD_SOURCE,
            }),
            VCDiffDecodeMsg::Inst{o_start: 0, first: DecInst::Add(ADD{len: 1, p_pos: 14}), second: Some(DecInst::Copy(COPY { len:4, u_pos:1 }))},
            VCDiffDecodeMsg::Inst{o_start: 5, first: DecInst::Add(ADD{len: 2, p_pos: 15}), second: Some(DecInst::Copy(COPY { len:6, u_pos:4 }))},
            VCDiffDecodeMsg::EndOfWindow,
            VCDiffDecodeMsg::EndOfFile,
        ];
        let bytes = vec![
            214,195,196,0, //magic
            0, //hdr_indicator
            1, //win_indicator VCD_SOURCE
            4, //SSS
            1, //SSP
            12, //delta window size
            13, //target window size
            0, //delta indicator
            3, //length of data for ADDs and RUNs
            2, //length of instructions and sizes
            2, //length of addresses for COPYs
            72,33,32, //'H! ' data section
            163, //ADD1 COPY4_mode0
            189, //ADD2 COPY6_mode2
            1,
            3,
        ];
        let reader = std::io::Cursor::new(&bytes);
        let mut dec = VCDDecoder::new(VCDReader::new(reader.clone()).unwrap());
        for check in insts.into_iter(){
            let msg = dec.next().unwrap();
            assert_eq!(msg, check, "{:?} != {:?}", msg, check);
        }
    }

}