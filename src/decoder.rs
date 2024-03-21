use std::io::{Read, Seek};

use crate::{decode_integer, reader::{VCDReader, VCDiffReadMsg, WindowSummary}, CodeTableEntry, TableInst, VCD_C_TABLE};

///Decoded ADD instruction
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ADD{
    ///Length of the data in the data section
    pub len:u32,
    ///Absolute position in the Patch file where the data starts (in the data section for this window)
    pub p_pos:u64,
}
///Decoded COPY instruction
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct COPY{
    pub len:u32,
    ///Decoded start position in window string U
    pub u_pos:u32,
}
///Inlined of Decoded COPY instruction
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RUN{
    pub len: u32,
    pub byte: u8
}
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

struct Cache {
    near: [usize; Self::S_NEAR], // Fixed array size of 4
    next_slot: usize,
    same: [usize; Self::S_SAME * 256], // Fixed array size of 3 * 256
}

impl Cache {
    const S_NEAR: usize = 4;
    const S_SAME: usize = 3;
    fn new() -> Self {
        Cache {
            near: [0; Self::S_NEAR],
            next_slot: 0,
            same: [0; Self::S_SAME * 256],
        }
    }
    fn update(&mut self, address: usize) {
        // Update near cache
        self.near[self.next_slot] = address;
        self.next_slot = (self.next_slot + 1) % Self::S_NEAR; // Modulus for circular buffer within array

        // Update same cache
        let same_index = address % (Self::S_SAME * 256); // Modulus for same cache addressing
        self.same[same_index] = address;
    }

    #[allow(dead_code)] //placeholder until we try to encode after merging
    pub fn addr_encode(&mut self, addr: usize, here: usize) -> (u32, u8) { // Return encoded address and mode
        let mut best_distance = addr;
        let mut best_mode = 0; // VCD_SELF

        // VCD_HERE
        let distance = here - addr;
        if distance < best_distance {
            best_distance = distance;
            best_mode = 1;
        }

        // Near cache
        for (i, &near_addr) in self.near.iter().enumerate() {
            if addr >= near_addr && addr - near_addr < best_distance {
                best_distance = addr - near_addr;
                best_mode = i + 2;
            }
        }

        // Same cache
        let same_index = addr % (Self::S_SAME * 256);
        if self.same[same_index] == addr {
            let distance = same_index % 256;
            best_distance = distance;
            best_mode = self.near.len() + 2 + same_index / 256;
        }

        self.update(addr);
        (best_distance as u32, best_mode as u8)
    }

    /// **addr_section**, is a reader that is positioned at the start of the address section
    /// It is only advanced by this function, so it remembers what it has read
    /// **here** is the current position in the target output
    /// **mode** is the mode of the address
    pub fn addr_decode<R:Read>(&mut self, addr_section: &mut R, here: u64, mode: usize) ->  std::io::Result<u64> {
        let addr ;
        if mode == 0 { // VCD_SELF
            let (x,_) = decode_integer(addr_section)?;
            addr = x;
        } else if mode == 1 { // VCD_HERE
            let (x,_) = decode_integer(addr_section)?;
            addr = here - x;
        } else if mode >= 2 && mode - 2 < Cache::S_NEAR {  // Near cache
                let near_index = mode - 2;
                let (x,_) = decode_integer(addr_section)?;
                addr = self.near[near_index] as u64 + x;
        } else { // Same cache
            let m = mode - (2 + Cache::S_NEAR);
            let mut byte = [0u8];
            addr_section.read_exact(&mut byte)?;
            let same_index = m * 256 + byte[0] as usize;
            addr = self.same[same_index] as u64;
        }
        self.update(addr as usize);
        Ok(addr)
    }
}

///This works as a decoder for the VCDIFF format.
///It can only advance forward in the output stream.
///You simply ask for the byte we are interrogating.
///It will return the DecodedInstruction that control that byte in the output stream.
pub struct VCDDecoder<R> {
    caches: Cache,
    ///The monotonic Reader for scanning the VCDIFF file
    reader: VCDReader<R>,
    ///Where we are in the address section
    addr_pos: u64,
    ///Where we are in the data section
    data_pos: u64,
    ///Where we are in the instruction section
    inst_pos: u64,
    cur_window: usize,
    ///Are we currently in a window?
    in_window: bool,
    ///This is global in the target output
    ///That is, it cuts across window boundaries
    ///This increments after we read any single DecodedInstruction
    cur_o_position: u64,
    ///This is our position in string U, it starts at SourceSegmentSize and increments as we read instructions
    ///Giving us our current position in U
    cur_u_position: Option<u64>,
}
impl<R: Read + Seek> VCDDecoder<R> {
    pub fn new(reader: VCDReader<R>) -> Self {
        VCDDecoder {
            caches: Cache::new(),
            reader,
            addr_pos: 0,
            data_pos: 0,
            inst_pos: 0,
            cur_window: 0,
            in_window: false,
            cur_o_position: 0,
            cur_u_position: None,
        }
    }
    pub fn position(&self) -> u64 {
        self.cur_o_position
    }
    pub fn u_position(&self) -> Option<u64> {
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
                if let Some(u_pos) = self.cur_u_position{
                    self.cur_u_position = Some(u_pos + len);

                }
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
                if let Some(upos) = self.cur_u_position{
                    self.cur_u_position = Some(upos + len);

                }
                let inst = DecInst::Add(ADD { len: len as u32, p_pos: pos });
                Some(inst)
            },
            TableInst::Copy { size, mode } => {
                let len = if size == 0 {
                    self.reader.read_as_inst_size_unchecked()?
                }else{size as u64};
                let addr = self.caches.addr_decode(self.reader.get_reader(self.addr_pos)?, self.cur_o_position,mode as usize)?;
                self.cur_o_position += len;
                if let Some(upos) = self.cur_u_position{
                    self.cur_u_position = Some(upos + len);

                }else{
                    panic!("We are in a window with a copy instruction, but no source segment size was given.");
                }
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
                self.cur_u_position = ws.source_segment_size;
                Ok(VCDiffDecodeMsg::WindowSummary(ws))
            },
            VCDiffReadMsg::OpCode(code) => {
                //this is the only message that can advance our position
                if self.in_window{
                    let o_start = self.cur_o_position;
                    let CodeTableEntry{ first, second } = VCD_C_TABLE[code as usize];
                    let f = if let Some(inst) = self.handle_inst(first)? {
                        self.cur_o_position += inst.len() as u64;
                        inst
                    }else{
                        panic!("NoOp is not allowed in the first position of an opcode"); //? maybe in an app specific table? Seems illogical yet
                    };
                    let s = if let Some(inst) = self.handle_inst(second)? {
                        self.cur_o_position += inst.len() as u64;
                        Some(inst)
                    }else{None};

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

    use super::*;
    const TEST_PATCH: [u8;42] = [
        // header 0..5
        214,195,196,0,
        0,//hdr_indicator
        // /header (5)
        // window_header 5..14
        1,//win_indicator
        4,0,//source_segment_size, source_segment_position
        33,//length_of_the_delta_encoding
        28,//size_of_the_target_window
        0,//delta_indicator
        24,//length_of_data_for_adds_and_runs
        3,//length_of_instructions_and_sizes
        1,//length_of_addresses_for_copys
        // /window_header (9)
        119,120,121,122,101,102,
        103,104,101,102,103,104,
        101,102,103,104,101,102,//data_section_position 14..38 (24)
        103,104,122,122,122,122,
        20,1,24,//instructions_section_position 38..41(3)
        0,//Copy addresses 41..42(1)
    ];

    #[test]
    fn test_decode() {
        let reader = std::io::Cursor::new(&TEST_PATCH);
        let mut dec = VCDDecoder::new(VCDReader::new(reader.clone()).unwrap());
        while let Ok(msg) = dec.next() {
            println!("{:?}", msg);
            if let VCDiffDecodeMsg::EndOfFile = msg {
                break;
            }
        }
    }

}