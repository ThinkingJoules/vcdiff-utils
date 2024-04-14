

use std::io::{self, Read, Seek};

use vcdiff_common::{decode_integer, Cache, CodeTableData, CodeTableEntry, CopyType, DeltaIndicator, Header, Inst, Instruction, TableInst, WinIndicator, WindowSummary, ADD, COPY, MAGIC, RUN, VCD_C_TABLE};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum VCDiffState {
    Window{inst_sec_start:u64, addr_sec_start:u32,end_of_window:u32,sss:u64},
    Instructions{addr_sec_start:u64,end_of_window:u32,sss:u64},
    EoW,
    EoF,
}

/// A reader for VCDIFF patches.
#[derive(Debug)]
pub struct VCDReader<R> {
    source: R,
    pub header: Header,
    caches: Cache,
    cur_state: VCDiffState,
    cur_pos: u64,
    moved: bool,
    addr_pos: u64,
    ///Where we are in the data section
    data_pos: u64,
    ///This is our position in string U, it starts at SourceSegmentSize and increments as we read instructions
    ///Giving us our current position in U
    cur_u_position: u32,
}

impl<R: Read + Seek> VCDReader<R> {
    /// Creates a new VCDReader from a source that implements the `Read` trait.
    /// This function attempts to read and parse the VCDIFF header to ensure the stream is valid.
    /// If the stream does not start with the expected VCDIFF magic number or the header is invalid,
    /// an error is returned.
    pub fn new(mut source: R) -> io::Result<Self> {
        // Attempt to read and validate the VCDIFF header.
        source.seek(io::SeekFrom::Start(0))?;
        let header = read_header(&mut source)?;
        if let Some(_) = header.code_table_data.as_ref(){
            unimplemented!("Application-Defined Code Tables are not supported.")
        }
        if let Some(_) = header.secondary_compressor_id{
            unimplemented!("Secondary compressors are not supported.")
        }
        Ok(Self {
            source,
            cur_pos:header.encoded_size() as u64,
            header,
            cur_state: VCDiffState::EoW,
            moved: false,
            caches: Cache::new(),
            addr_pos: 0,
            data_pos: 0,
            cur_u_position: 0,

        })
    }
    /// Consumes the reader and returns the inner reader.
    pub fn into_inner(self)->R{
        self.source
    }
    /// Seeks to the start of the window at the given position.
    /// This is useful when you want to seek to a specific window in the patch. It allows you to 'rewind'. If needed.
    /// This does not verify that the position is a valid window start.
    /// # Arguments
    /// * `win_start_pos` - The byte offset from the start of the patch file where the window starts.
    pub fn seek_to_window(&mut self,win_start_pos:u64){
        self.cur_pos = win_start_pos;
        self.cur_state = VCDiffState::EoW;
        self.moved = true; //let the next call seek
    }
    /// Allows inner access to the reader in a controlled manner that will not mess up the reader's state.
    /// This is useful when you need to read from the patch file directly.
    /// # Arguments
    /// * `from_start` - The byte offset from the start of the patch file where to start reading.
    /// * `buf` - The buffer to read_exact into.
    pub fn read_from_src(&mut self,from_start:u64, buf:&mut [u8])->io::Result<()>{
        self.get_reader(from_start)?.read_exact(buf)
    }
    /// Allows inner access to the reader in a controlled manner that will not mess up the reader's state.
    /// This is useful when you need to read from the patch file directly.
    /// # Arguments
    /// * `at_from_start` - The byte offset from the start of the patch file for where to seek to.
    pub fn get_reader(&mut self,at_from_start:u64)->io::Result<&mut R>{
        self.moved = true;
        self.source.seek(io::SeekFrom::Start(at_from_start))?;
        Ok(&mut self.source)
    }
    ///since other layers might need to access R, and we have Seek trait, we need to make sure we
    ///are reading from the right spot. Will probably be a noop often.
    fn resume(&mut self)->std::io::Result<()>{
        if self.moved {self.source.seek(io::SeekFrom::Start(self.cur_pos))?;}
        Ok(())
    }

    fn read_as_inst(&mut self)->io::Result<Option<VCDiffReadMsg>>{
        self.resume()?;
        if let VCDiffState::Instructions { addr_sec_start, end_of_window,.. } = &self.cur_state {
            if self.cur_pos >= *addr_sec_start{
                //we skip over the copy address section
                let pos = addr_sec_start + *end_of_window as u64;
                self.cur_pos = self.source.seek(io::SeekFrom::Start(pos))?;
                self.cur_state = VCDiffState::EoW;
                return Ok(Some(VCDiffReadMsg::EndOfWindow));
            }
            let mut buffer = [0; 1];
            self.source.read_exact(&mut buffer)?;
            self.cur_pos += 1;
            Ok(Some(self.decode_inst(buffer[0])?))
        }else{
            Ok(None)
        }
    }
    fn decode_inst(&mut self,byte:u8)->io::Result<VCDiffReadMsg>{
        debug_assert!(matches!(self.cur_state,VCDiffState::Instructions{..}));
        let CodeTableEntry{ first, second } = VCD_C_TABLE[byte as usize];
        let f = if let Some(inst) = self.handle_inst(first)? {
            inst
        }else{
            panic!("NoOp is not allowed in the first position of an opcode"); //? maybe in an app specific table? Seems illogical yet
        };
        let s = self.handle_inst(second)?;

        Ok(VCDiffReadMsg::Inst{first: f, second: s})

    }
    fn handle_inst(&mut self, inst: TableInst) -> std::io::Result<Option<Inst>> {
        let inst = match inst{
            TableInst::Run => {
                //read the length of the run
                let len = self.read_as_inst_size_unchecked()? as u32;
                let mut byte = [0u8];
                self.read_from_src(self.data_pos, &mut byte)?;
                self.data_pos += 1;
                Inst::Run(RUN { len: len as u32, byte: byte[0] })
            },
            TableInst::Add { size } => {
                let len = if size == 0 {
                    self.read_as_inst_size_unchecked()? as u32
                }else{size as u32};
                let pos = self.data_pos;
                self.data_pos += len as u64;
                Inst::Add(ADD { len: len as u32, p_pos: pos })
            },
            TableInst::Copy { size, mode } => {
                let len = if size == 0 {
                    self.read_as_inst_size_unchecked()? as u32
                }else{size as u32};
                let (value,read) = if mode < Cache::SAME_START as u8 {
                    //read int
                    decode_integer(self.get_reader(self.addr_pos)?)?
                }else{
                    //read byte
                    let mut byte = [0u8];
                    self.read_from_src(self.addr_pos, &mut byte)?;
                    (byte[0] as u64, 1)
                };
                let addr = self.caches.addr_decode(value, self.cur_u_position as u64, mode as usize) as u32;
                if addr >= self.cur_u_position{
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Address is out of bounds: {:?} >= {:?} (mode: {} len: {}) encoded {} ",
                        addr, self.cur_u_position,mode,len,value))
                    );
                }
                self.addr_pos += read as u64;
                let sss = match self.cur_state{
                    VCDiffState::Instructions{sss,..} => sss,
                    _ => panic!("Invalid State!"),
                };
                let end_pos = addr + len;
                //dbg!(end_pos,cur_u, len_u, sss, ssp);
                let copy_type = if end_pos > self.cur_u_position{//seQ
                    if (addr as u64) < sss {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("CopyT (Sequence) must be entirely in T! Position in U: {} Copy Address: {} Copy Len: {}, End Position in U: {}, Source Segment Size: {}",
                                self.cur_u_position, addr,len,end_pos, sss
                            )
                        ));
                    }
                    let len_o = (end_pos - self.cur_u_position) as u32;
                    CopyType::CopyQ{len_o}
                }else if end_pos as u64 <= sss{//inS
                    CopyType::CopyS
                }else{//inT
                    if (addr as u64) < sss {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("CopyT (Non-Sequence) must be entirely in T! Position in U: {} Copy Address: {} Copy Len: {}, End Position in U: {}, Source Segment Size: {}",
                                self.cur_u_position, addr,len,end_pos, sss
                            )
                        ));
                    }
                    CopyType::CopyT{inst_u_pos_start:self.cur_u_position}
                };
                Inst::Copy(COPY { len: len as u32, u_pos: addr,copy_type })
            },
            TableInst::NoOp => return Ok(None),
        };
        self.cur_u_position += inst.len_in_o();
        Ok(Some(inst))
    }
    ///This reads the next bytes as an integer.
    ///It does not verify that the reader is in the correct state.
    ///If used incorrectly, this will screw up the reader's state.
    fn read_as_inst_size_unchecked(&mut self)->io::Result<u64>{
        self.resume()?;
        let (integer,len) = decode_integer(&mut self.source)?;
        self.cur_pos += len as u64;
        Ok(integer)
    }
    /// Reads the next segment from the VCDIFF patch, returning it as a `VCDiffMessage`.
    pub fn next(&mut self) -> io::Result<VCDiffReadMsg> {
        match self.cur_state {
            VCDiffState::Window { inst_sec_start, addr_sec_start, end_of_window,sss } => {
                //we need to skip over the data_section
                //we could get rid of the seek trait
                //but we would need to read over a bunch of bytes to get to the instructions
                //I assume seek is faster
                self.cur_pos = self.source.seek(io::SeekFrom::Start(inst_sec_start))?;
                //we assume the patch file contains at least one opcode.
                assert!(addr_sec_start> 0);
                let a = inst_sec_start + addr_sec_start as u64;
                self.cur_state = VCDiffState::Instructions { addr_sec_start:a, end_of_window,sss };
                match self.read_as_inst()? {
                    Some(v) => Ok(v),
                    None => {
                        //this is an implementation error
                        panic!("Instructions state should always have a next opcode");
                    }
                }
            },
            VCDiffState::Instructions { .. } => {
                match self.read_as_inst() {
                    Ok(Some(v)) => Ok(v),
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                        self.cur_state = VCDiffState::EoF;
                        Ok(VCDiffReadMsg::EndOfFile)
                    }
                    Err(e) => Err(e),
                    Ok(None) => {
                        //this is an implementation error
                        panic!("Instructions state should always have a next opcode");
                    }
                }
            },
            VCDiffState::EoW => {
                self.resume()?;
                match read_window_header(&mut self.source,self.cur_pos) {
                    Ok(ws) => {
                        self.cur_pos += ws.win_hdr_len() as u64;
                        self.cur_state = VCDiffState::Window {
                            inst_sec_start: ws.inst_sec_start(),
                            addr_sec_start: (ws.addr_sec_start() - ws.inst_sec_start()) as u32,
                            end_of_window: (ws.end_of_window() - ws.addr_sec_start()) as u32,
                            sss: ws.source_segment_size.unwrap_or(0),
                        };
                        self.addr_pos = ws.addr_sec_start();
                        self.data_pos = ws.data_sec_start();
                        self.cur_u_position = ws.source_segment_size.unwrap_or(0) as u32;
                        self.caches = Cache::new();
                        Ok(VCDiffReadMsg::WindowSummary(ws))
                    }
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                        self.cur_state = VCDiffState::EoF;
                        Ok(VCDiffReadMsg::EndOfFile)
                    }
                    Err(e) => return Err(e),
                }
            },
            VCDiffState::EoF => Ok(VCDiffReadMsg::EndOfFile),
        }
    }
}

/// Reads the fixed part of the VCDIFF patch file header.
pub fn read_header<R: Read>(source: &mut R) -> io::Result<Header> {
    let mut buffer = [0; 4]; // Buffer to read the fixed header part.
    source.read_exact(&mut buffer)?;
    // Validate the fixed part of the header.
    if buffer != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid VCDiff header"));
    }

    // Read the Hdr_Indicator.
    let mut byte_buffer = [0; 1];
    source.read_exact(&mut byte_buffer)?;
    let hdr_indicator = byte_buffer[0];

    // hdr_indicator has been read already
    let has_secondary_compressor = hdr_indicator & 0x01 != 0;
    let has_code_table = hdr_indicator & 0x02 != 0;

    let secondary_compressor_id = if has_secondary_compressor {
        source.read_exact(&mut byte_buffer)?;
        Some(byte_buffer[0]) // Assuming existence of read_u8 for brevity
    } else {
        None
    };

    let code_table_data = if has_code_table {
        source.read_exact(&mut byte_buffer)?;
        let size_of_near_cache = byte_buffer[0];
        source.read_exact(&mut byte_buffer)?;
        let size_of_same_cache = byte_buffer[0];
        let (length_of_compressed_code_table,_) = decode_integer(source)?;
        let mut compressed_code_table_data = vec![0u8; length_of_compressed_code_table as usize];
        source.read_exact(&mut compressed_code_table_data)?;

        Some(CodeTableData {
            size_of_near_cache,
            size_of_same_cache,
            compressed_code_table_data,
        })
    } else {
        None
    };

    Ok(Header {
        hdr_indicator,
        secondary_compressor_id,
        code_table_data,
    })
}
/// Reads the window header from the VCDIFF patch file.
/// To avoid the Seek trait we require the caller to tell us the current position of the reader.
/// This way the caller can know that the reader can only read exactly the window header.
/// # Arguments
/// * `source` - The reader to read the window header from. Must be already positioned at the start of the window header.
/// * `win_start_pos` - The byte offset from the start of the patch file where the window starts.
pub fn read_window_header<R: Read>(source: &mut R, win_start_pos: u64) -> io::Result<WindowSummary>{
    let mut buffer = [0; 1];
    source.read_exact(&mut buffer)?;
    let win_indicator = WinIndicator::from_u8(buffer[0]);
    let (source_segment_size,source_segment_position) = match win_indicator {
        WinIndicator::VCD_SOURCE |
        WinIndicator::VCD_TARGET => {
            let (sss,_) = decode_integer(source)?;
            let (ssp,_) = decode_integer(source)?;
            (Some(sss), Some(ssp))
        },
        WinIndicator::Neither => (None,None),
    };

    let (length_of_the_delta_encoding,_) = decode_integer(source)?;
    let (size_of_the_target_window,_) = decode_integer(source)?;
    source.read_exact(&mut buffer)?;
    let delta_indicator = DeltaIndicator::from_u8(buffer[0]);
    let (length_of_data_for_adds_and_runs,_) = decode_integer(source)?;
    let (length_of_instructions_and_sizes,_) = decode_integer(source)?;
    let (length_of_addresses_for_copys,_) = decode_integer(source)?;
    Ok(WindowSummary{
        win_start_pos,
        win_indicator,
        source_segment_size,
        source_segment_position,
        length_of_the_delta_encoding,
        size_of_the_target_window,
        delta_indicator,
        length_of_data_for_adds_and_runs,
        length_of_instructions_and_sizes,
        length_of_addresses_for_copys,
    })

}
/// Represents the main messages that can be emitted by the Reader while processing a VCDIFF patch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VCDiffReadMsg {
    WindowSummary(WindowSummary),
    Inst{first: Inst, second: Option<Inst>},
    EndOfWindow,
    EndOfFile,
}

#[cfg(test)]
mod test_super {
    use vcdiff_common::{DeltaIndicator, WinIndicator, WindowSummary};

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
    const TEST_WINDOW: WindowSummary = WindowSummary{
        win_start_pos: 5,
        win_indicator: WinIndicator::VCD_SOURCE,
        source_segment_size: Some(4),
        source_segment_position: Some(0),
        length_of_the_delta_encoding: 33,
        size_of_the_target_window: 28,
        delta_indicator: DeltaIndicator(0),
        length_of_data_for_adds_and_runs: 24,
        length_of_instructions_and_sizes: 3,
        length_of_addresses_for_copys: 1,
    };
    #[test]
    fn test_header() {
        let mut reader = std::io::Cursor::new(&TEST_PATCH);
        let header = read_header(&mut reader).unwrap();
        assert_eq!(header.hdr_indicator, 0);
        assert_eq!(header.secondary_compressor_id, None);
        assert_eq!(header.code_table_data, None);
        assert_eq!(header.encoded_size(), 5);
    }
    #[test]
    fn test_win_header(){
        let mut reader = std::io::Cursor::new(&TEST_PATCH);
        let header = read_header(&mut reader).unwrap();
        let pos = header.encoded_size() as u64;
        let ws = read_window_header(&mut reader,pos).unwrap();
        assert_eq!(ws, TEST_WINDOW);
        assert_eq!(ws.win_hdr_len(), 9);
        assert_eq!(ws.data_sec_start(), 14);
        assert_eq!(ws.inst_sec_start(), 38);
        assert_eq!(ws.addr_sec_start(), 41);
        assert_eq!(ws.end_of_window(), 42);
    }
    #[test]
    fn test_decode() {
        //'hello' -> 'Hello! Hello!'
        let insts = [
            VCDiffReadMsg::WindowSummary(WindowSummary{

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
            VCDiffReadMsg::Inst{first: Inst::Add(ADD{len: 1, p_pos: 14}), second: Some(Inst::Copy(COPY { len:4, u_pos:0,copy_type:CopyType::CopyS }))},
            VCDiffReadMsg::Inst{first: Inst::Add(ADD{len: 2, p_pos: 15}), second: Some(Inst::Copy(COPY { len:6, u_pos:4,copy_type:CopyType::CopyT { inst_u_pos_start: 11 }  }))},
            VCDiffReadMsg::EndOfWindow,
            VCDiffReadMsg::EndOfFile,
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
            163, //ADD1 COPY4_mode6
            183, //ADD2 COPY6_mode0
            0,
            4,
        ];
        let reader = std::io::Cursor::new(&bytes);
        let mut dec = VCDReader::new(reader.clone()).unwrap();
        for check in insts.into_iter(){
            let msg = dec.next().unwrap();
            assert_eq!(msg, check, "{:?} != {:?}", msg, check);
        }
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
            116, 101, 114, 115, 116, //data section b"terst" 12..17
            200, //ADD size3 & COPY5_mode0
            3, //ADD size 2
            1, //addr for copy
        ];
        let patch = std::io::Cursor::new(patch);
        let mut dec = VCDReader::new(patch).unwrap();
        let insts = [
            VCDiffReadMsg::WindowSummary(WindowSummary{
                source_segment_size: None,
                source_segment_position: None,
                length_of_the_delta_encoding: 13,
                size_of_the_target_window: 8,
                delta_indicator: DeltaIndicator::from_u8(0),
                length_of_data_for_adds_and_runs: 5,
                length_of_instructions_and_sizes: 2,
                length_of_addresses_for_copys: 1,
                win_start_pos: 5,
                win_indicator: WinIndicator::Neither,
            }),
            VCDiffReadMsg::Inst{first: Inst::Add(ADD{len: 3, p_pos: 12}), second: Some(Inst::Copy(COPY { len:5, u_pos:1,copy_type:CopyType::CopyQ {len_o: 3 } }))},
            VCDiffReadMsg::Inst{first: Inst::Add(ADD{len: 2, p_pos: 15}), second: None},
            VCDiffReadMsg::EndOfWindow,
            VCDiffReadMsg::EndOfFile,
        ];
        for check in insts.into_iter(){
            let msg = dec.next().unwrap();
            assert_eq!(msg, check, "{:?} != {:?}", msg, check);
        }
    }

    #[test]
    fn kitchen_sink_transform2(){
        let patch = vec![
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
        use CopyType::*;
        let insts = [
            VCDiffReadMsg::WindowSummary(WindowSummary{
                win_indicator: WinIndicator::VCD_SOURCE,
                source_segment_size: Some(11),
                source_segment_position: Some(1),
                size_of_the_target_window:7 ,
                delta_indicator: DeltaIndicator(0),
                length_of_the_delta_encoding: 14,
                length_of_data_for_adds_and_runs: 1,
                length_of_instructions_and_sizes: 5,
                length_of_addresses_for_copys: 3,
                win_start_pos: 5,
            }),
            VCDiffReadMsg::Inst{first: Inst::Add(ADD{len: 1, p_pos: 14}), second: Some(Inst::Copy(COPY { len:4, u_pos:0,copy_type:CopyType::CopyS }))},
            VCDiffReadMsg::Inst { first: Inst::Copy(COPY { len: 1, u_pos: 10, copy_type: CopyS }), second: None },
            VCDiffReadMsg::Inst { first: Inst::Copy(COPY { len: 1, u_pos: 4, copy_type: CopyS }), second: None },
            VCDiffReadMsg::EndOfWindow,
            VCDiffReadMsg::WindowSummary(WindowSummary{
                win_start_pos: 23,
                win_indicator: WinIndicator::VCD_TARGET,
                source_segment_size: Some(7),
                source_segment_position: Some(0),
                length_of_the_delta_encoding: 14,
                size_of_the_target_window: 14,
                delta_indicator: DeltaIndicator(0),
                length_of_data_for_adds_and_runs: 1,
                length_of_instructions_and_sizes: 5,
                length_of_addresses_for_copys: 3
            }),
            VCDiffReadMsg::Inst { first: Inst::Copy(COPY { len: 7, u_pos: 0, copy_type: CopyS }), second: None },
            VCDiffReadMsg::Inst { first: Inst::Copy(COPY { len: 12, u_pos: 7, copy_type: CopyQ { len_o: 5 } }), second: None },
            VCDiffReadMsg::Inst{first: Inst::Add(ADD{len: 1, p_pos: 32}), second: None},
            VCDiffReadMsg::Inst { first: Inst::Copy(COPY { len: 1, u_pos: 13, copy_type: CopyT { inst_u_pos_start: 20 } }), second: None },

            VCDiffReadMsg::EndOfWindow,
            VCDiffReadMsg::EndOfFile
        ];
        let patch = std::io::Cursor::new(patch);
        let mut dec = VCDReader::new(patch).unwrap();

        // writer.start_new_win(WriteWindowHeader { win_indicator: WinIndicator::VCD_SOURCE, source_segment_size: Some(11), source_segment_position: Some(1), size_of_the_target_window:7 , delta_indicator: DeltaIndicator(0) }).unwrap();
        // writer.next_inst(WriteInst::ADD("H".as_bytes().to_vec())).unwrap();
        // writer.next_inst(WriteInst::COPY(COPY { len: 4, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap(); //ello
        // writer.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 10,copy_type:CopyType::CopyS  })).unwrap(); //'!'
        // writer.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 4,copy_type:CopyType::CopyS  })).unwrap(); //' '
        // writer.start_new_win(WriteWindowHeader { win_indicator: WinIndicator::VCD_TARGET, source_segment_size: Some(7), source_segment_position: Some(0), size_of_the_target_window:14 , delta_indicator: DeltaIndicator(0) }).unwrap();
        // writer.next_inst(WriteInst::COPY(COPY { len: 7, u_pos: 0,copy_type:CopyType::CopyS  })).unwrap(); //'Hello! '
        // writer.next_inst(WriteInst::COPY(COPY { len: 12, u_pos: 7,copy_type:CopyType::CopyQ { len_o:5 }  })).unwrap(); //'Hello! Hello'
        // writer.next_inst(WriteInst::ADD(".".as_bytes().to_vec())).unwrap();
        // writer.next_inst(WriteInst::COPY(COPY { len: 1, u_pos: 13,copy_type:CopyType::CopyT { inst_u_pos_start: 20 }})).unwrap(); // ' '

        // let w = writer.finish().unwrap().into_inner();
        // //dbg!(&w);

        for check in insts.into_iter(){
            let msg = dec.next().unwrap();
            assert_eq!(msg, check, "{:?} != {:?}", msg, check);
        }



    }
}