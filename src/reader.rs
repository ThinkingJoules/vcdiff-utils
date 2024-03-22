
use std::io::{self, Read, Seek};

use crate::{decode_integer, integer_encoded_size, MAGIC};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum VCDiffState {
    Window{inst_sec_start:u64, addr_sec_start:u32,end_of_window:u32},
    Instructions{addr_sec_start:u64,end_of_window:u32},
    EoW,
    EoF,
}
pub struct VCDReader<R> {
    source: R,
    pub header: Header,
    cur_state: VCDiffState,
    cur_pos: u64,
    moved: bool,
}

impl<R: Read + Seek> VCDReader<R> {
    /// Creates a new VCDReader from a source that implements the `Read` trait.
    /// This function attempts to read and parse the VCDIFF header to ensure the stream is valid.
    /// If the stream does not start with the expected VCDIFF magic number or the header is invalid,
    /// an error is returned.
    pub fn new(mut source: R) -> io::Result<Self> {
        // Attempt to read and validate the VCDIFF header.
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
        })
    }
    pub fn read_from_src(&mut self,from_start:u64, buf:&mut [u8])->io::Result<()>{
        self.source.seek(io::SeekFrom::Start(from_start))?;
        self.moved = true;
        self.source.read_exact(buf)
    }
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
        if let VCDiffState::Instructions { addr_sec_start, end_of_window } = &self.cur_state {
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
            Ok(Some(VCDiffReadMsg::InstSecByte(buffer[0])))
        }else{
            Ok(None)
        }
    }
    ///This reads the next bytes as an integer.
    ///It does not verify that the reader is in the correct state.
    ///If used incorrectly, this will screw up the reader's state.
    pub fn read_as_inst_size_unchecked(&mut self)->io::Result<u64>{
        self.resume()?;
        let (integer,len) = decode_integer(&mut self.source)?;
        self.cur_pos += len as u64;
        Ok(integer)
    }
    /// Reads the next segment from the VCDIFF patch, returning it as a `VCDiffMessage`.
    /// If the end of the file is reached, `None` is returned.
    pub fn next(&mut self) -> io::Result<VCDiffReadMsg> {
        match self.cur_state {
            VCDiffState::Window { inst_sec_start, addr_sec_start, end_of_window } => {
                //we need to skip over the data_section
                //we could get rid of the seek trait
                //but we would need to read over a bunch of bytes to get to the instructions
                //I assume seek is faster
                self.cur_pos = self.source.seek(io::SeekFrom::Start(inst_sec_start))?;
                //we assume the patch file contains at least one opcode.
                assert!(addr_sec_start> 0);
                let a = inst_sec_start + addr_sec_start as u64;
                self.cur_state = VCDiffState::Instructions { addr_sec_start:a, end_of_window };
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
                        };
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
    /// A single byte from the Instructions section.
    /// This might actually be a size of an instruction, but we can't know that without the code table.
    InstSecByte(u8),
    EndOfWindow,
    EndOfFile,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Header{
    hdr_indicator: u8,
    secondary_compressor_id: Option<u8>,
    code_table_data: Option<CodeTableData>,
}
impl Header{
    pub fn encoded_size(&self)->usize{
        let mut size = 4 + 1; // Fixed part of the header
        if self.secondary_compressor_id.is_some(){
            size += 1;
        }
        if let Some(code_table_data) = &self.code_table_data{
            let integer = code_table_data.compressed_code_table_data.len();
            let int_size = integer_encoded_size(integer as u64);
            size += 1 + 1 + integer + int_size;
        }
        size
    }
}

/// Encapsulates the code table data found in the VCDIFF patch header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CodeTableData {
    size_of_near_cache: u8,
    size_of_same_cache: u8,
    compressed_code_table_data: Vec<u8>,
}



/// Represents a summary of a window in a VCDIFF patch, including the positions of different sections within the window.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowSummary {
    win_start_pos:u64,
    pub win_indicator: WinIndicator,
    pub source_segment_size: Option<u64>,
    pub source_segment_position: Option<u64>,
    length_of_the_delta_encoding: u64,
    pub size_of_the_target_window: u64,
    pub delta_indicator: DeltaIndicator,
    length_of_data_for_adds_and_runs: u64,
    length_of_instructions_and_sizes: u64,
    length_of_addresses_for_copys: u64,
}
impl WindowSummary{
    pub fn win_hdr_len(&self)->usize{
        let mut size = 1;
        if let Some(s) = self.source_segment_size{
            size += integer_encoded_size(s);
        }
        if let Some(s) = self.source_segment_position{
            size += integer_encoded_size(s);
        }
        size += integer_encoded_size(self.length_of_the_delta_encoding);
        size += integer_encoded_size(self.size_of_the_target_window);
        size += 1; //delta_indicator
        size += integer_encoded_size(self.length_of_data_for_adds_and_runs);
        size += integer_encoded_size(self.length_of_instructions_and_sizes);
        size += integer_encoded_size(self.length_of_addresses_for_copys);
        size
    }
    pub fn data_sec_start(&self)->u64{
        self.win_start_pos + self.win_hdr_len() as u64
    }
    pub fn inst_sec_start(&self)->u64{
        self.data_sec_start() + self.length_of_data_for_adds_and_runs
    }
    pub fn addr_sec_start(&self)->u64{
        self.inst_sec_start() + self.length_of_instructions_and_sizes
    }
    pub fn end_of_window(&self)->u64{
        self.addr_sec_start() + self.length_of_addresses_for_copys
    }
    pub fn is_vcd_target(&self)->bool{
        self.win_indicator == WinIndicator::VCD_TARGET
    }
    pub fn has_reference_data(&self)->bool{
        self.win_indicator != WinIndicator::Neither
    }
}

#[repr(u8)]
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum WinIndicator {
    Neither = 0,
    VCD_SOURCE = 1 << 0,
    VCD_TARGET = 1 << 1,
}
impl Default for WinIndicator {
    fn default() -> Self {
        Self::Neither
    }
}
impl WinIndicator {
    pub fn from_u8(byte: u8) -> Self {
        match byte {
            0 => Self::Neither,
            1 => Self::VCD_SOURCE,
            2 => Self::VCD_TARGET,
            _ => panic!("Invalid WinIndicator byte: {}", byte),
        }
    }

    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeltaIndicator(u8);

impl DeltaIndicator {
    pub fn from_u8(byte: u8) -> Self {
        Self(byte)
    }

    pub fn to_u8(&self) -> u8 {
        self.0
    }

    pub fn is_datacomp(&self) -> bool {
        self.0 & 0x01 != 0
    }

    pub fn is_instcomp(&self) -> bool {
        self.0 & 0x02 != 0
    }

    pub fn is_addrcomp(&self) -> bool {
        self.0 & 0x04 != 0
    }
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
    fn test_reader(){
        let mut reader = std::io::Cursor::new(&TEST_PATCH);
        let mut vcd_reader = VCDReader::new(&mut reader).unwrap();
        let msg = vcd_reader.next().unwrap();
        assert_eq!(msg, VCDiffReadMsg::WindowSummary(TEST_WINDOW));
        //next should be the 3 opcodes
        let msg = vcd_reader.next().unwrap();
        assert_eq!(msg, VCDiffReadMsg::InstSecByte(20));
        //this is technically not an opcode, but a length
        //however, without the reader knowing the table, it can't know
        //Our reader only knows this bytes is in the instructions section
        //So this really isn't an opcode, but we can only return it as if it is
        //this is fine.
        let msg = vcd_reader.next().unwrap();
        assert_eq!(msg, VCDiffReadMsg::InstSecByte(1));
        let msg = vcd_reader.next().unwrap();
        assert_eq!(msg, VCDiffReadMsg::InstSecByte(24));
        //next should be EoW
        let msg = vcd_reader.next().unwrap();
        assert_eq!(msg, VCDiffReadMsg::EndOfWindow);
        //next should be end of file
        let msg = vcd_reader.next().unwrap();
        assert_eq!(msg, VCDiffReadMsg::EndOfFile);
    }
}