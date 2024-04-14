use std::io::Write;

use vcdiff_common::{encode_integer, integer_encoded_size, Cache, CodeTableEntry, CopyType, DeltaIndicator, Header, InstType, Instruction, TableInst, WinIndicator, COPY, MAGIC, RUN, VCD_C_TABLE};

///This is the main struct that will be used to write VCDIFF files.
#[derive(Debug)]
pub struct VCDWriter<W> {
    sink:W,
    caches: Cache,
    //circular buffer of last 3 instructions
    buffer: [Option<WriteInst>;3],
    //should keep increasing. Value points to the index to write to (mod 3)
    buffer_pos: usize,
    data_buffer: Vec<u8>,
    inst_buffer: Vec<u8>,
    addr_buffer: Vec<u8>,
    cur_win:Option<WriteWindowHeader>,
    cur_t_size:u32,
}

impl<W: Write> VCDWriter<W> {
    ///Creates a new VCDWriter with the given header and sink.
    pub fn new(mut sink: W, header:Header) -> std::io::Result<Self> {
        let Header { hdr_indicator, secondary_compressor_id, code_table_data } = header;
        if hdr_indicator != 0 || secondary_compressor_id.is_some() || code_table_data.is_some(){
            unimplemented!("Secondary compressor and code table data are not supported yet.")
        }
        sink.write_all(&MAGIC)?;
        sink.write_all(&[hdr_indicator])?;

        Ok(VCDWriter {
            sink,
            caches: Cache::new(),
            buffer: [None,None,None],
            buffer_pos: 0,
            data_buffer: Vec::new(),
            inst_buffer: Vec::new(),
            addr_buffer: Vec::new(),
            cur_win: None,
            cur_t_size: 0,
        })
    }
    fn cur_u_pos(&self) -> u32 {
        self.cur_win.as_ref().unwrap().source_segment_size.unwrap_or(0) as u32 + self.cur_t_size
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
    pub fn next_inst(&mut self,mut inst:WriteInst) ->Result<(),&str> {
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
    fn add_inst(&mut self,inst:WriteInst) {

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
            //let cur_inst_len = inst.len_in_t(self.cur_u_pos());
            // 1. Mode Retrieval (For COPY)
            let (f_addr,f_mode) = match &inst {
                WriteInst::COPY(copy) => {
                    debug_assert!({
                        match copy.copy_type {
                            CopyType::CopyQ { len_o } => {
                                copy.len_in_u() + copy.u_pos - self.cur_u_pos() == len_o
                            },
                            CopyType::CopyS => true,
                            CopyType::CopyT { inst_u_pos_start } => inst_u_pos_start == self.cur_u_pos(),
                        }
                    });
                    //we actually need to update the cache here or else a second copy would peek the wrong state
                    self.caches.addr_encode(copy.u_pos as usize, self.cur_u_pos() as usize)
                },
                WriteInst::ADD(_) => (0,0), //ADD pass through
                a => {//RUN short circuit
                    let first = from_enc_inst_first(a,0);
                    self.encode_first_inst(offset, 0,first.size(),None);
                    return;
                },
            };
            //first inst is not Run, so we need to try the second inst
            let first = from_enc_inst_first(&inst,f_mode);
            if let Some(next) = self.buffer[(self.buffer_pos + offset + 1) % 3].as_ref() {
                let sec_here = self.cur_u_pos()+inst.len_in_u();//+cur_inst_len;
                let s_mode = match &next {
                    WriteInst::COPY(copy) => {
                        debug_assert!({
                            match copy.copy_type {
                                CopyType::CopyQ { len_o } => {
                                    copy.len_in_u() + copy.u_pos - (self.cur_u_pos() + inst.len_in_o()) == len_o
                                }CopyType::CopyS => true,
                                CopyType::CopyT { inst_u_pos_start } => inst_u_pos_start == self.cur_u_pos() + inst.len_in_o(),
                            }
                        }, "Copy instruction does not match the expected length: {:?} cur_u_pos: {} inst_len: {}",copy,self.cur_u_pos() + inst.len_in_o(),next.len_in_o());

                        self.caches.peek_addr_encode(copy.u_pos as usize, sec_here as usize).1
                    },
                    WriteInst::RUN(_) => {//run cannot be the second inst, send the first inst through
                        self.encode_first_inst(offset, get_single_inst_opcode(first),first.size(),Some(f_addr));
                        return;
                    },
                    _ => 0, //ADD passes through again
                };

                // 2. TableInst Creation
                let second = from_enc_inst_sec(Some(next),s_mode);

                let table_entry = get_proper_table_entry(first,Some(second));
                let start_offset = if table_entry.sec_is_noop() {0}else{163};
                //this should always find a code in the standard table
                let code_pos = VCD_C_TABLE.iter().skip(start_offset).position(|entry| entry == &table_entry).unwrap();
                let op_code = (code_pos + start_offset) as u8;
                self.encode_first_inst(offset, op_code, first.size(), Some(f_addr));
                if !table_entry.sec_is_noop() {
                    self.encode_second_to_buffers(offset + 1, second.size());
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
        let inst_len = inst.len_in_o();
        self.cur_t_size += inst_len;
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
        let inst_len = inst.len_in_o();
        let addr = if let WriteInst::COPY(COPY { u_pos, .. }) = &inst {
            Some(self.caches.addr_encode(*u_pos as usize, self.cur_u_pos() as usize).0)
        } else { None };
        //this must be after the addr calculation
        self.cur_t_size += inst_len;
        self.encode_inst_to_buffers(inst, size, addr)
    }
    ///This function takes an EncInst and splits it into the 3 buffers
    ///It DOES NOT increment the cur_t_size or change the unwritten buffer in any way.
    fn encode_inst_to_buffers(&mut self, inst:WriteInst, size:u8, addr:Option<u32>) {
        if size == 0 {
            encode_integer(&mut self.inst_buffer, inst.len_in_u() as u64).unwrap();
        }
        match inst {
            WriteInst::ADD(data) => {
                self.data_buffer.extend(data);
            },
            WriteInst::COPY(_) => {
                encode_integer(&mut self.addr_buffer, addr.unwrap() as u64).unwrap();
            },
            WriteInst::RUN(run) => {
                self.data_buffer.push(run.byte);
            },
        }
    }
    ///This function will finish the current window and write it to the sink.
    /// # Arguments
    /// * `win_hdr` - The header for the next window window
    pub fn start_new_win(&mut self,win_hdr:WriteWindowHeader)-> std::io::Result<()> {
        self.write_cur_window_to_sink()?;
        self.cur_win = Some(win_hdr);
        self.caches = Cache::new();
        Ok(())
    }
    ///This will consume the writer and write the last window to the sink
    /// # Returns
    /// * The sink with the written data
    pub fn finish(mut self) -> std::io::Result<W> {
        self.write_cur_window_to_sink()?;
        Ok(self.sink)
    }
    fn write_cur_window_to_sink(&mut self) -> std::io::Result<()> {
        for i in 0..3 {
            self.encode_insts(i);
        }
        debug_assert!(self.buffer[0].is_none() && self.buffer[1].is_none() && self.buffer[2].is_none());
        if let Some(WriteWindowHeader { win_indicator, source_segment_size, source_segment_position, size_of_the_target_window, delta_indicator }) = self.cur_win.take() {
            //first clear our unwritten buffer and write the data to the 3 buffers


            if size_of_the_target_window != self.cur_t_size as u64{
                //format the error to include the size mismatch
                let s = format!("Size of the target window does not match the given instructions. Expected {} but instructions sum to {}",size_of_the_target_window,self.cur_t_size);
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, s));
            }
            if delta_indicator.to_u8() != 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Compression of the target window is not supported. Delta indicator must be 0."));
            }
            match win_indicator {
                WinIndicator::Neither => {
                    if source_segment_position.is_some() || source_segment_size.is_some() {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Source segment size and position must be None for Neither window type."));
                    }
                    self.sink.write_all(&[win_indicator.to_u8()])?;
                },
                WinIndicator::VCD_SOURCE | WinIndicator::VCD_TARGET => {
                    if source_segment_position.is_none() || source_segment_size.is_none() {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Source segment size and position must be provided for VCD_SOURCE or VCD_TARGET window type."));
                    }
                    self.sink.write_all(&[win_indicator.to_u8()])?;
                    encode_integer(&mut self.sink, source_segment_size.unwrap())?;
                    encode_integer(&mut self.sink, source_segment_position.unwrap())?;
                },
            }

            /*
            The delta encoding of the target window
              Length of the delta encoding         - integer
              The delta encoding
                  Size of the target window        - integer
                  Delta_Indicator                  - byte
                  Length of data for ADDs and RUNs - integer
                  Length of instructions and sizes - integer
                  Length of addresses for COPYs    - integer
                  Data section for ADDs and RUNs   - array of bytes
                  Instructions and sizes section   - array of bytes
                  Addresses section for COPYs      - array of bytes

            */
            //calc the lengths and encode the win hdr
            let ad_len = self.data_buffer.len();
            let is_len = self.inst_buffer.len();
            let a_len = self.addr_buffer.len();
            //calc the lengths of the integers
            let ad_int_len = integer_encoded_size(ad_len as u64);
            let is_int_len = integer_encoded_size(is_len as u64);
            let a_int_len = integer_encoded_size(a_len as u64);
            //size of target window int
            let tw_int_len = integer_encoded_size(size_of_the_target_window);
            //total length of the delta encoding
            let total_len = tw_int_len + 1 + ad_int_len + is_int_len + a_int_len + ad_len + is_len + a_len;
            //write the total length
            encode_integer(&mut self.sink, total_len as u64)?;
            //write the size of the target window
            encode_integer(&mut self.sink, size_of_the_target_window)?;
            //write the delta indicator
            self.sink.write_all(&[delta_indicator.to_u8()])?;
            //write the lengths of the 3 arrays
            encode_integer(&mut self.sink, ad_len as u64)?;
            encode_integer(&mut self.sink, is_len as u64)?;
            encode_integer(&mut self.sink, a_len as u64)?;
            //write the 3 arrays
            self.sink.write_all(&self.data_buffer)?;
            self.sink.write_all(&self.inst_buffer)?;
            self.sink.write_all(&self.addr_buffer)?;
            //reset the buffer/t_size/etc
            self.data_buffer.clear();
            self.inst_buffer.clear();
            self.addr_buffer.clear();
            self.cur_t_size = 0;
            self.buffer_pos = 0;
        }
        Ok(())
    }
}

///These are the Instructions that must be given to the VCDWriter.
#[derive(Clone, Debug,PartialEq, Eq)]
pub enum WriteInst{
    ADD(Vec<u8>),
    RUN(RUN),
    //we take the COPY instruction verbatim
    //it is up to the caller to ensure they call end_cur_win at the right time
    //The address will only be right if they know what string U is already.
    COPY(COPY),
}
impl WriteInst{
    fn try_merge(&mut self, other:&mut WriteInst) -> bool {
        match (self,other) {
            (WriteInst::ADD(data),WriteInst::ADD(other_data)) => {
                data.append(other_data);
                true
            },
            (WriteInst::RUN(run),WriteInst::RUN(other_run)) if run.byte == other_run.byte => {
                run.len += other_run.len;
                true
            },
            _ => false,
        }
    }
}

impl Instruction for WriteInst {
    fn len_in_u(&self) -> u32 {
        match self {
            WriteInst::ADD(data) => data.len() as u32,
            WriteInst::RUN(run) => run.len,
            WriteInst::COPY(copy) => copy.len,
        }
    }

    fn inst_type(&self)->InstType {
        match self {
            WriteInst::ADD(_) => InstType::Add,
            WriteInst::RUN(_) => InstType::Run,
            WriteInst::COPY(c) => c.inst_type(),
        }
    }


}


fn from_enc_inst_first(inst:&WriteInst,mode:u8) -> TableInst {
    let len = inst.len_in_u();
    let mut size = if len > u8::MAX as u32{
        0
    } else {
        len as u8
    };
    match inst {
        WriteInst::ADD(_) => {
            if !(1..=17).contains(&size) {
                size = 0;
            }
            TableInst::Add { size }
        },
        WriteInst::RUN(_) => TableInst::Run,
        WriteInst::COPY(_) => {
            if !(4..=18).contains(&size) {
                size = 0;
            }
            TableInst::Copy { size, mode }
        },
    }
}
fn from_enc_inst_sec(inst:Option<&WriteInst>,mode:u8) -> TableInst {
    if inst.is_none() {
        return TableInst::NoOp;
    }
    let inst = inst.unwrap();
    let len = inst.len_in_u();
    let size = if len > u8::MAX as u32{
        0
    } else {
        len as u8
    };
    match inst {
        WriteInst::COPY(_) => {
            match mode {
                0..=5 => {
                    if (4..=6).contains(&size) {
                        TableInst::Copy { size, mode }
                    }else{
                        TableInst::NoOp
                    }
                },
                6..=8 => {
                    if size == 4{
                        TableInst::Copy { size, mode }
                    }else{
                        TableInst::NoOp
                    }
                },
                _ => TableInst::NoOp,
            }
        },
        WriteInst::ADD(_) => {
            if size == 1 {
                TableInst::Add { size }
            }else{
                TableInst::NoOp
            }
        },
        WriteInst::RUN(_) => TableInst::NoOp,
    }
}


///The window header that must be given to the VCDWriter to start a new window.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WriteWindowHeader {
    pub win_indicator: WinIndicator,
    pub source_segment_size: Option<u64>,
    pub source_segment_position: Option<u64>,
    pub size_of_the_target_window: u64,
    pub delta_indicator: DeltaIndicator,
}
fn get_proper_table_entry(first:TableInst,second:Option<TableInst>) -> CodeTableEntry{
    //we need to filter additionally for the odd ball double ops
    match (first,second) {
        (TableInst::Copy { size:copy_size, .. },Some(TableInst::Add { size:add_size })) if add_size == 1 && copy_size == 4 => CodeTableEntry { first, second:second.unwrap() },
        (TableInst::Add { size:add_size },Some(TableInst::Copy { .. })) if (1..=4).contains(&add_size) => CodeTableEntry { first, second:second.unwrap()  },
        (first,Some(_)) |
        (first,None) => CodeTableEntry { first, second: TableInst::NoOp },
    }
}
fn get_single_inst_opcode(first:TableInst) -> u8 {
    let table_entry = CodeTableEntry { first, second: TableInst::NoOp };
    VCD_C_TABLE.iter().position(|entry| entry == &table_entry).unwrap() as u8
}

#[cfg(test)]
mod test_super {

    use vcdiff_common::CopyType;

    use super::*;
    use std::io::Cursor;
    #[test]
    fn test_get_single_inst_opcode() {
        let first = TableInst::Add { size: 1 };
        let res = get_single_inst_opcode(first);
        assert_eq!(res,2);
    }
    #[test]
    fn test_get_double_inst_opcode() {
        let first = TableInst::Add { size: 2 };
        let second = TableInst::Copy { size: 6, mode: 2 };
        let table_entry = CodeTableEntry { first, second };
        let mut res = VCD_C_TABLE.iter().skip(163).position(|entry| entry == &table_entry).unwrap() as u8;
        res+=163;
        assert_eq!(res,189);
    }
    #[test]
    fn test_basic_add_run() {
        // Setup
        let mock_sink = Cursor::new(Vec::new());
        let header = Header { hdr_indicator: 0, secondary_compressor_id: None, code_table_data: None };
        let mut writer = VCDWriter::new(mock_sink, header).unwrap();

        // Create a new window
        let win_hdr = WriteWindowHeader {
            win_indicator: WinIndicator::Neither,
            source_segment_size: None,
            source_segment_position: None,
            size_of_the_target_window: 5,
            delta_indicator: DeltaIndicator::default(),
        };
        writer.start_new_win(win_hdr).unwrap();

        // Instructions
        writer.next_inst(WriteInst::ADD(vec![b'h',b'e'])).unwrap();
        writer.next_inst(WriteInst::RUN(RUN { len: 2, byte: b'l' })).unwrap();
        writer.next_inst(WriteInst::ADD(vec![b'o'])).unwrap();

        // Force encoding
        let w = writer.finish().unwrap().into_inner();
        let answer = vec![
            214, 195, 196, 0,  //magic
            0,  //hdr_indicator
            0, //win_indicator
            13, //size_of delta window
            5, //size of target window
            0, //delta indicator
            4, //length of data for ADDs and RUNs
            4, //length of instructions and sizes
            0, //length of addresses for COPYs
            104, 101, 108, 111, //data section b"helo"
            3, //ADD size 2
            0, //RUN
            2, //... of len 2
            2, //ADD size 1
        ];
        assert_eq!(w, answer);

    }
    #[test]
    fn test_basic_add_seq() {
        // Setup
        let mock_sink = Cursor::new(Vec::new());
        let header = Header { hdr_indicator: 0, secondary_compressor_id: None, code_table_data: None };
        let mut writer = VCDWriter::new(mock_sink, header).unwrap();

        // Create a new window
        let win_hdr = WriteWindowHeader {
            win_indicator: WinIndicator::Neither,
            source_segment_size: None,
            source_segment_position: None,
            size_of_the_target_window: 8,
            delta_indicator: DeltaIndicator::default(),
        };
        writer.start_new_win(win_hdr).unwrap();

        // Instructions -> "" -> "tererest'
        writer.next_inst(WriteInst::ADD(vec![b't',b'e',b'r'])).unwrap();
        writer.next_inst(WriteInst::COPY(COPY { len: 5, u_pos: 1,copy_type:CopyType::CopyQ { len_o:3 } })).unwrap();
        writer.next_inst(WriteInst::ADD(vec![b's',b't'])).unwrap();

        // Force encoding
        let w = writer.finish().unwrap().into_inner();
        let answer = vec![
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
        assert_eq!(w, answer);

    }
    #[test]
    fn test_hello_transformation() {
        // Setup
        let mock_sink = Cursor::new(Vec::new());
        let header = Header { hdr_indicator: 0, secondary_compressor_id: None, code_table_data: None };
        let mut writer = VCDWriter::new(mock_sink, header).unwrap();

        // New window
        let win_hdr = WriteWindowHeader {
            win_indicator: WinIndicator::VCD_SOURCE,
            source_segment_size: Some(4),
            source_segment_position: Some(1),
            size_of_the_target_window: 13,
            delta_indicator: DeltaIndicator::default(),
        };
        writer.start_new_win(win_hdr).unwrap();

        // Instructions
        // "hello" -> "Hello! Hello!"
        writer.next_inst(WriteInst::ADD(vec![b'H'])).unwrap(); // Add 'H'
        writer.next_inst(WriteInst::COPY(COPY{ len: 4, u_pos: 0,copy_type:CopyType::CopyS })).unwrap(); // Copy 'ello'
        writer.next_inst(WriteInst::ADD(vec![b'!', b' '])).unwrap(); // Add '! '
        writer.next_inst(WriteInst::COPY(COPY{ len: 6, u_pos: 4,copy_type:CopyType::CopyT { inst_u_pos_start: 11 }  })).unwrap(); // Copy "Hello!"

        // Force encoding (remains the same)
        let w = writer.finish().unwrap().into_inner();
        //dbg!(&w);
        let answer = vec![
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
            163, //ADD1 COPY4_mode6
            183, //ADD2 COPY6_mode0
            0,
            4,
        ];

        assert_eq!(w, answer);
    }
    #[test]
    pub fn neither_win(){
        let mock_sink = Cursor::new(Vec::new());
        let header = Header { hdr_indicator: 0, secondary_compressor_id: None, code_table_data: None };
        let mut writer = VCDWriter::new(mock_sink, header).unwrap();
        let n_window = WriteWindowHeader {
            win_indicator: WinIndicator::Neither,
            source_segment_size: None,
            source_segment_position: None,
            size_of_the_target_window: 1,
            delta_indicator: DeltaIndicator::default(),
        };
        writer.start_new_win(n_window).unwrap();
        writer.next_inst(WriteInst::ADD(vec![b'H'])).unwrap();
        let w = writer.finish().unwrap().into_inner();
        //dbg!(&w);
        let answer = vec![
            214,195,196,0, //magic
            0, //hdr_indicator
            0, //win_indicator Neither
            7, //delta window size
            1, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            1, //length of instructions and size
            0, //length of addr
            72, //data section
            2, //ADD1
        ];
        assert_eq!(w, answer);
    }
    #[test]
    pub fn complex_transform(){
        //we need 3 windows, Neither, Src, and Target, in that order.
        //src will be 'hello' and output will be 'Hello! Hello!'
        //we encode just the Add(H) in the Neither window
        //then we encode the COPY(ello) in the Src window
        //then we encode the Copy(Hello!) in the Target window
        let mock_sink = Cursor::new(Vec::new());
        let header = Header { hdr_indicator: 0, secondary_compressor_id: None, code_table_data: None };
        let mut wrtier = VCDWriter::new(mock_sink, header).unwrap();
        let n_window = WriteWindowHeader {
            win_indicator: WinIndicator::Neither,
            source_segment_size: None,
            source_segment_position: None,
            size_of_the_target_window: 1,
            delta_indicator: DeltaIndicator::default(),
        };
        wrtier.start_new_win(n_window).unwrap();
        wrtier.next_inst(WriteInst::ADD(vec![b'H'])).unwrap();
        let s_window = WriteWindowHeader {
            win_indicator: WinIndicator::VCD_SOURCE,
            source_segment_size: Some(4),
            source_segment_position: Some(1),
            size_of_the_target_window: 5,
            delta_indicator: DeltaIndicator::default(),
        };
        wrtier.start_new_win(s_window).unwrap();
        wrtier.next_inst(WriteInst::COPY(COPY{ len: 4, u_pos: 0,copy_type:CopyType::CopyS })).unwrap();
        wrtier.next_inst(WriteInst::ADD(vec![b'!'])).unwrap();
        let t_window = WriteWindowHeader {
            win_indicator: WinIndicator::VCD_TARGET,
            source_segment_size: Some(6),
            source_segment_position: Some(0),
            size_of_the_target_window: 7,
            delta_indicator: DeltaIndicator::default(),
        };
        wrtier.start_new_win(t_window).unwrap();
        wrtier.next_inst(WriteInst::ADD(vec![b' '])).unwrap();
        wrtier.next_inst(WriteInst::COPY(COPY{ len: 6, u_pos: 0 ,copy_type:CopyType::CopyS})).unwrap();

        let w = wrtier.finish().unwrap().into_inner();
        //dbg!(&w);
        let answer = vec![
            214,195,196,0, //magic
            0, //hdr_indicator
            0, //win_indicator Neither
            7, //delta window size
            1, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            1, //length of instructions and size
            0, //length of addr
            72, //data section 'H
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
            33, //data section '!'
            247, //COPY4_mode0 ADD1
            0, //addr 0
            2, //win_indicator VCD_TARGET
            6, //SSS
            0, //SSP
            8, //delta window size
            7, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            1, //length of instructions and size
            1, //length of addr
            32, //data section ' '
            165, //ADD1 COPY6_mode0
            0, //addr 0
        ];

        assert_eq!(w, answer);



    }

    #[test]
    pub fn kitchen_sink_transform(){
        //we need 3 windows, Neither, Src, and Target, in that order.
        //src will be 'hello' and output will be 'Hello! Hello! Hell...'
        //we encode just the Add(H) in the Neither window
        //then we encode the COPY(ello) in the Src window
        //then we encode the Copy(Hello!) in the Target window
        //then we encode the Copy(Hell) in the Target window, referencing the last window
        //then we encode the Add('.') in the Target window
        //then we encode an implicit Copy For the last '..' chars.
        let mock_sink = Cursor::new(Vec::new());
        let header = Header { hdr_indicator: 0, secondary_compressor_id: None, code_table_data: None };
        let mut wrtier = VCDWriter::new(mock_sink, header).unwrap();
        let n_window = WriteWindowHeader {
            win_indicator: WinIndicator::Neither,
            source_segment_size: None,
            source_segment_position: None,
            size_of_the_target_window: 1,
            delta_indicator: DeltaIndicator::default(),
        };
        wrtier.start_new_win(n_window).unwrap();
        wrtier.next_inst(WriteInst::ADD(vec![b'H'])).unwrap();
        let s_window = WriteWindowHeader {
            win_indicator: WinIndicator::VCD_SOURCE,
            source_segment_size: Some(4),
            source_segment_position: Some(1),
            size_of_the_target_window: 5,
            delta_indicator: DeltaIndicator::default(),
        };
        wrtier.start_new_win(s_window).unwrap();
        wrtier.next_inst(WriteInst::COPY(COPY{ len: 4, u_pos: 0, copy_type:CopyType::CopyS })).unwrap();
        wrtier.next_inst(WriteInst::ADD(vec![b'!'])).unwrap();
        let t_window = WriteWindowHeader {
            win_indicator: WinIndicator::VCD_TARGET,
            source_segment_size: Some(6),
            source_segment_position: Some(0),
            size_of_the_target_window: 7,
            delta_indicator: DeltaIndicator::default(),
        };
        wrtier.start_new_win(t_window).unwrap();
        wrtier.next_inst(WriteInst::ADD(vec![b' '])).unwrap();
        wrtier.next_inst(WriteInst::COPY(COPY{ len: 6, u_pos: 0,copy_type:CopyType::CopyS })).unwrap();
        let t_window = WriteWindowHeader {
            win_indicator: WinIndicator::VCD_TARGET,
            source_segment_size: Some(5),
            source_segment_position: Some(6),
            size_of_the_target_window: 8,
            delta_indicator: DeltaIndicator::default(),
        };
        wrtier.start_new_win(t_window).unwrap();
        wrtier.next_inst(WriteInst::COPY(COPY{ len: 5, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap();
        wrtier.next_inst(WriteInst::ADD(vec![b'.'])).unwrap();
        wrtier.next_inst(WriteInst::COPY(COPY{ len: 3, u_pos: 10,copy_type:CopyType::CopyQ { len_o: 2 }  })).unwrap();


        let w = wrtier.finish().unwrap().into_inner();
        //dbg!(&w);
        let answer = vec![
            214,195,196,0, //magic
            0, //hdr_indicator
            0, //win_indicator Neither
            7, //delta window size
            1, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            1, //length of instructions and size
            0, //length of addr
            72, //data section 'H
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
            33, //data section '!'
            247, //COPY4_mode0 ADD1
            0, //addr 0
            2, //win_indicator VCD_TARGET
            6, //SSS
            0, //SSP
            8, //delta window size
            7, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            1, //length of instructions and size
            1, //length of addr
            32, //data section ' '
            165, //ADD1 COPY6_mode0
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
            46, //data section '.'
            21, //ADD1 COPY5_mode0
            2, //Add1 NOOP
            19, //COPY0_mode0
            3, //...size
            0, //addr 0
            10, //addr 1
        ];

        assert_eq!(w, answer);



    }
    #[test]
    pub fn kitchen_sink_transform2(){

        let mock_sink = Cursor::new(Vec::new());
        let header = Header { hdr_indicator: 0, secondary_compressor_id: None, code_table_data: None };
        let mut writer = VCDWriter::new(mock_sink, header).unwrap();
        writer.start_new_win(WriteWindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(11), source_segment_position: Some(1), size_of_the_target_window:7 , delta_indicator: DeltaIndicator(0) }).unwrap();
        writer.next_inst(WriteInst::ADD("H".as_bytes().to_vec())).unwrap();
        writer.next_inst(WriteInst::COPY(COPY { len: 4, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap(); //ello
        writer.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 10,copy_type:CopyType::CopyS  })).unwrap(); //'!'
        writer.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 4,copy_type:CopyType::CopyS  })).unwrap(); //' '
        writer.start_new_win(WriteWindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(7), source_segment_position: Some(0), size_of_the_target_window:14 , delta_indicator: DeltaIndicator(0) }).unwrap();
        writer.next_inst(WriteInst::COPY(COPY { len: 7, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap(); //'Hello! '
        writer.next_inst(WriteInst::COPY(COPY { len: 12, u_pos: 7,copy_type:CopyType::CopyQ { len_o:5 }  })).unwrap(); //'Hello! Hello'
        writer.next_inst(WriteInst::ADD(".".as_bytes().to_vec())).unwrap();
        writer.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 13,copy_type:CopyType::CopyT { inst_u_pos_start: 20 }})).unwrap(); // ' '

        let w = writer.finish().unwrap().into_inner();
        //dbg!(&w);
        let answer = vec![
            214,195,196,0, //magic
            0, //hdr_indicator
            1, //win_indicator Src
            11, //SSS
            1, //SSP
            14, //delta window size
            7, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            5, //length of instructions and size
            3, //length of addr
            72, //data section 'H'
            163, //ADD1 COPY4_mode0
            19, //COPY0_mode0
            1, //..size
            19, //COPY0_mode0
            1, //..size
            0, //addr 0
            10, //addr 1
            4, //addr 2
            2, //win_indicator VCD_TARGET
            7, //SSS
            0, //SSP
            14, //delta window size
            14, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            5, //length of instructions and size
            3, //length of addr
            46, //data section '.'
            23, //COPY0_mode0 noop
            28, //..size
            2, //Add1 NOOP
            19, //COPY0_mode0
            1, //..size
            0, //addr 0
            7, //addr 1
            13, //addr 2
        ];

        assert_eq!(w, answer);



    }
}