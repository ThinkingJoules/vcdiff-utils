use std::io::Write;

use crate::{encode_integer, reader::{DeltaIndicator, WinIndicator}, Cache, CodeTableEntry, TableInst, COPY, RUN, VCD_C_TABLE};


pub struct VCDEncoder<W:Write> {
    sink:W,
    caches: Cache,
    //circular buffer of last 3 instructions
    buffer: [Option<EncInst>;3],
    //should keep increasing. Value points to the index to write to (mod 3)
    buffer_pos: usize,
    data_buffer: Vec<u8>,
    inst_buffer: Vec<u8>,
    addr_buffer: Vec<u8>,
    cur_win:Option<WindowHeader>,
    cur_t_size:u32,
}

impl<W: Write> VCDEncoder<W> {
    pub fn new(sink: W) -> Self {
        VCDEncoder {
            sink,
            caches: Cache::new(),
            buffer: [None,None,None],
            buffer_pos: 0,
            data_buffer: Vec::new(),
            inst_buffer: Vec::new(),
            addr_buffer: Vec::new(),
            cur_win: None,
            cur_t_size: 0,
        }
    }
    fn cur_u_pos(&self) -> u32 {
        self.cur_win.as_ref().unwrap().source_segment_position.unwrap_or(0) as u32 + self.cur_t_size
    }
    fn last_idx(&self) -> usize {
        // +2 = -1
        (self.buffer_pos + 2) % 3
    }
    fn next_idx(&self) -> usize {
        // attempt to write a new instruction here
        self.buffer_pos % 3
    }
    ///This will take in a new instruction and attempt to merge it with the last instruction
    ///If it cannot be merged, it will be added to the unwritten buffer.
    pub fn next_inst(&mut self,mut inst:EncInst) ->Result<(),&str> {
        if self.cur_win.is_none() {
            return Err("No window started");
        }
        if let Some(prev) = &mut self.buffer[self.last_idx()] {
            if prev.try_merge(&mut inst) {
                return Ok(());
            }
        }
        self.add_inst(inst);
        Ok(())
    }
    ///This will fire off the unwritten buffered instruction encodings if the current slot has an instruction
    ///Otherwise it will just add the instruction to the empty slot in the buffer
    fn add_inst(&mut self,inst:EncInst) {
        //if cur pos is some, then we need to encode it and maybe it's successor
        if self.buffer[self.next_idx()].is_some() {
            self.encode_insts(0);//try to write 1 or 2 instructions
        }
        self.buffer[self.next_idx()] = Some(inst);
        self.buffer_pos += 1;
    }
    ///This finds the proper opcode and fires off the encoding to the 3 buffers for one or two instructions
    fn encode_insts(&mut self, offset: usize) {
        //TODO: Figure out a cleaner control flow for calling 'encode first inst'
        let idx = (self.buffer_pos + offset) % 3;
        //if nothing exists at idx we do nothing
        if let Some(inst) = self.buffer[idx].as_ref() {
            // 1. Mode Retrieval (For COPY)
            let (f_addr,f_mode) = match &inst {
                EncInst::COPY(copy) => {
                    //we actually need to update the cache here or else a second copy would peek the wrong state
                    self.caches.addr_encode(copy.u_pos as usize, self.cur_u_pos() as usize)
                },
                EncInst::ADD(_) => (0,0), //ADD pass through
                a => {//RUN short circuit
                    let first = TableInst::from_enc_inst(Some(a),0);
                    self.encode_first_inst(offset, get_single_inst_opcode(first),first.size(),None);
                    return;
                },
            };
            //first inst is not Run, so we need to try the second inst
            let first = TableInst::from_enc_inst(Some(&inst),f_mode);
            if let Some(next) = self.buffer[(idx + 1) % 3].as_ref() {
                let s_mode = match &next {
                    EncInst::COPY(copy) => {
                        self.caches.peek_addr_encode(copy.u_pos as usize, (self.cur_u_pos()+inst.len()) as usize).1
                    },
                    EncInst::RUN(_) => {//run cannot be the second inst, send the first inst through
                        self.encode_first_inst(offset, get_single_inst_opcode(first),first.size(),Some(f_addr));
                        return;
                    },
                    _ => 0, //ADD passes through again
                };

                // 2. TableInst Creation
                let second = TableInst::from_enc_inst(Some(next),s_mode);

                let table_entry = CodeTableEntry { first, second };
                match VCD_C_TABLE.iter().skip(163).position(|entry| entry == &table_entry){
                    Some(double_opcode) => {
                        self.encode_first_inst(offset, double_opcode as u8, first.size(), Some(f_addr));
                        self.encode_second_to_buffers(offset + 1, second.size());
                    },
                    None => self.encode_first_inst(offset, get_single_inst_opcode(first),first.size(),Some(f_addr)),
                }
            }else{//second doesn't exist, just encode the first
                self.encode_first_inst(offset, get_single_inst_opcode(first),first.size(),Some(f_addr));
            }

        }
    }
    ///This function encodes the opcode and takes the buffer instruction indicated by offset
    ///It increments the cur_t_size and pushes the opcode to the inst_buffer
    ///Then splits the instruction to the 3 buffers
    fn encode_first_inst(&mut self,offset:usize,op_code:u8,size:u8,addr:Option<u32>) {
        //attempts to encode one instruction
        //if nothing exists at idx this will panic
        let idx = (self.buffer_pos + offset) % 3;
        let inst = self.buffer[idx].take().unwrap();
        self.cur_t_size += inst.len();
        self.inst_buffer.push(op_code);
        self.encode_inst_to_buffers(inst, size, addr);
    }
    ///This function encodes the second instruction to the buffers
    ///It updates the cache if this is a COPY instruction
    ///Then it increments the cur_t_size
    ///Then splits the instruction to the 3 buffers
    fn encode_second_to_buffers(&mut self,offset: usize,size:u8) {
        //attempts to encode the second instruction
        //if nothing exists at idx we do nothing
        let idx = (self.buffer_pos + offset) % 3;
        let inst = self.buffer[idx].take().unwrap();
        let addr = if let EncInst::COPY(COPY { u_pos, .. }) = &inst {
            Some(self.caches.addr_encode(*u_pos as usize, self.cur_u_pos() as usize).0)
        } else { None };
        //this must be after the addr calculation
        self.cur_t_size += inst.len();
        self.encode_inst_to_buffers(inst, size, addr)
    }
    ///This function takes an EncInst and splits it into the 3 buffers
    ///It DOES NOT increment the cur_t_size or change the unwritten buffer in any way.
    fn encode_inst_to_buffers(&mut self, inst:EncInst, size:u8, addr:Option<u32>) {
        if size == 0 {
            encode_integer(&mut self.inst_buffer, inst.len() as u64).unwrap();
        }
        match inst {
            EncInst::ADD(data) => {
                self.data_buffer.extend(data);
            },
            EncInst::COPY(_) => {
                encode_integer(&mut self.addr_buffer, addr.unwrap() as u64).unwrap();
            },
            EncInst::RUN(run) => {
                self.data_buffer.push(run.byte);
            },
        }
    }
    pub fn start_new_win(&mut self,win_hdr:WindowHeader)-> std::io::Result<()> {
        self.flush()?;
        //we use the win_hdr to verify that what they gave us matches what we have.
        //the size of the target window should be the same as the cur_t_size



        todo!()
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(wh) = self.cur_win.take() {
            //first clear our unwritten buffer and write the data to the 3 buffers
            for i in 0..3 {
                self.encode_insts(i);
            }
            //check that our local data matches the given header
            //calc the lengths and encode the win hdr
            //write the 3 arrays
            //reset the buffer/t_size/etc

        }

        Ok(())
    }
}

pub enum EncInst{
    ADD(Vec<u8>),
    RUN(RUN),
    //we take the COPY instruction verbatim
    //it is up to the caller to ensure they call end_cur_win at the right time
    //The address will only be right if they know what string U is already.
    COPY(COPY),
}

impl EncInst {
    fn len(&self) -> u32 {
        match self {
            EncInst::ADD(data) => data.len() as u32,
            EncInst::RUN(run) => run.len,
            EncInst::COPY(copy) => copy.len,
        }
    }
    fn try_merge(&mut self, other:&mut EncInst) -> bool {
        match (self,other) {
            (EncInst::ADD(data),EncInst::ADD(other_data)) => {
                data.append(other_data);
                true
            },
            (EncInst::RUN(run),EncInst::RUN(other_run)) if run.byte == other_run.byte => {
                run.len += other_run.len;
                true
            },
            _ => false,
        }
    }
}

impl TableInst{
    fn from_enc_inst(inst:Option<&EncInst>,mode:u8) -> Self {
        if inst.is_none() {
            return TableInst::NoOp;
        }
        let inst = inst.unwrap();
        let size = if inst.len() > u8::MAX as u32{
            0
        } else {
            inst.len() as u8
        };
        match inst {
            EncInst::ADD(_) => TableInst::Add { size },
            EncInst::RUN(_) => TableInst::Run,
            EncInst::COPY(_) => TableInst::Copy { size, mode },
        }
    }
    fn size(&self) -> u8 {
        match self {
            TableInst::Add { size } => *size,
            TableInst::Run => 0,
            TableInst::Copy { size, .. } => *size,
            TableInst::NoOp => 0,
        }
    }
}

pub struct WindowHeader {
    pub win_indicator: WinIndicator,
    pub source_segment_size: Option<u64>,
    pub source_segment_position: Option<u64>,
    pub size_of_the_target_window: u64,
    pub delta_indicator: DeltaIndicator,
}

fn get_single_inst_opcode(first:TableInst) -> u8 {
    let table_entry = CodeTableEntry { first, second: TableInst::NoOp };
    VCD_C_TABLE.iter().position(|entry| entry == &table_entry).unwrap() as u8
}