

pub const MAGIC:[u8;4] = [b'V'|0x80, b'C'|0x80, b'D'|0x80, 0];


#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Header{
    pub hdr_indicator: u8,
    pub secondary_compressor_id: Option<u8>,
    pub code_table_data: Option<CodeTableData>,
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
    pub size_of_near_cache: u8,
    pub size_of_same_cache: u8,
    pub compressed_code_table_data: Vec<u8>,
}



/// Represents a summary of a window in a VCDIFF patch, including the positions of different sections within the window.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowSummary {
    pub win_start_pos:u64,
    pub win_indicator: WinIndicator,
    pub source_segment_size: Option<u64>,
    pub source_segment_position: Option<u64>,
    pub length_of_the_delta_encoding: u64,
    pub size_of_the_target_window: u64,
    pub delta_indicator: DeltaIndicator,
    pub length_of_data_for_adds_and_runs: u64,
    pub length_of_instructions_and_sizes: u64,
    pub length_of_addresses_for_copys: u64,
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
#[derive(Copy,Clone, Debug, PartialEq, Eq)]
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
pub struct DeltaIndicator(pub u8);

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


///Basic instruction
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Inst{
    Add(ADD),
    Copy(COPY),
    Run(RUN)
}
impl Instruction for Inst{
    fn len_in_u(&self)->u32{
        match self{
            Inst::Add(a) => a.len,
            Inst::Copy(c) => c.len,
            Inst::Run(r) => r.len
        }
    }

    fn inst_type(&self)->InstType {
        match self{
            Inst::Add(_) => InstType::Add,
            Inst::Run(_) => InstType::Run,
            Inst::Copy(COPY { copy_type, .. }) => InstType::Copy (*copy_type),
        }
    }
}
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
    pub copy_type:CopyType,
}
impl Instruction for COPY{
    fn len_in_u(&self)->u32{
        self.len
    }
    fn inst_type(&self)->InstType{
        InstType::Copy(self.copy_type)
    }
}
///Inlined of Decoded RUN instruction
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RUN{
    pub len: u32,
    pub byte: u8
}
impl Instruction for RUN{
    fn len_in_u(&self)->u32{
        self.len
    }
    fn inst_type(&self)->InstType{
        InstType::Run
    }
}

pub trait Instruction:Clone{
    fn len_in_u(&self)->u32;
    fn inst_type(&self)->InstType;
    fn len_in_o(&self)->u32{
        match self.inst_type(){
            InstType::Add => self.len_in_u(),
            InstType::Run => self.len_in_u(),
            InstType::Copy(copy_type) => match copy_type{
                CopyType::CopyS => self.len_in_u(),
                CopyType::CopyT{..} => self.len_in_u(),
                CopyType::CopyQ{len_o} => len_o,
            }
        }
    }
    fn is_implicit_seq(&self)->bool{
        matches!(self.inst_type(), InstType::Copy(CopyType::CopyQ{..}))
    }
    fn copy_in_s(&self)->bool{
        matches!(self.inst_type(), InstType::Copy(CopyType::CopyS))
    }
    fn copy_in_t(&self)->bool{
        matches!(self.inst_type(), InstType::Copy(CopyType::CopyT{..}))
    }
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InstType{
    Add,
    Run,
    Copy(CopyType),
}

impl InstType {
    pub fn is_copy(&self)->bool{
        matches!(self, InstType::Copy{..})
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
///Currently this is only a static implementation of the cache. We cannot handle different S_NEAR or S_SAME parameters.
#[derive(Clone,Debug)]
pub struct Cache {
    near: [usize; Self::S_NEAR], // Fixed array size of 4
    next_slot: usize,
    same: [usize; Self::S_SAME * 256], // Fixed array size of 3 * 256
}
impl Cache {
    pub const S_NEAR: usize = 4;
    pub const S_SAME: usize = 3;
    pub const SAME_START: usize = Self::S_NEAR + 2;
    pub fn new() -> Self {
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
    /// **read_value** is the value read from the address section
    /// **here** is the current position in the target output
    /// **mode** is the mode of the address
    /// returns (address, bytes_read_from_addr_section)
    pub fn addr_decode(&mut self, read_value:u64, here: u64, mode: usize) -> u64 {
        let addr;
        if mode < Self::SAME_START{
            match mode {
                0 => {
                    addr = read_value
                },
                1 => {
                    addr = here - read_value;
                },
                m => {
                    let near_index = m - 2;
                    addr = self.near[near_index] as u64 + read_value;
                }
            }
        }else{// Same cache
            let m = mode - Self::SAME_START;
            assert!(read_value <= u8::MAX as u64,"read value is too large");
            let same_index = m * 256 + read_value as usize;
            addr = self.same[same_index] as u64;
        }
        self.update(addr as usize);
        addr
    }
    pub fn addr_encode(&mut self, addr: usize, here: usize) -> (u32, u8) { // Return encoded address and mode
        let res = self.peek_addr_encode(addr, here);
        self.update(addr);
        res
    }
    pub fn peek_addr_encode(&self, addr: usize, here: usize) -> (u32, u8){
        assert!(addr < here,"addr can not be ahead of cur pos");
        return (addr as u32,0);
        //There is an error either here in this code, or somewhere in the writer/encoder code.
        // let mut best_distance = addr;
        // let mut best_mode = 0; // VCD_SELF
        // // VCD_HERE
        // let distance = here - addr;
        // if distance < best_distance {
        //     best_distance = distance;
        //     best_mode = 1;
        // }

        // // Near cache
        // for (i, &near_addr) in self.near.iter().enumerate() {
        //     if addr >= near_addr && addr - near_addr < best_distance {
        //         best_distance = addr - near_addr;
        //         best_mode = i + 2;
        //     }
        // }

        // // Same cache
        // let distance = addr % (Self::S_SAME * 256);
        // if self.same[distance] == addr {
        //     best_distance = distance % 256;
        //     best_mode = Self::SAME_START + distance / 256;
        // }
        // (best_distance as u32, best_mode as u8)
    }
}

// Define the Instruction enum with associated data for size and mode.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TableInst {
    NoOp,
    Add { size: u8 },
    Run ,
    Copy { size: u8, mode: u8 },
}

impl TableInst {
    pub fn size(&self) -> u8 {
        match self {
            TableInst::Add { size } => *size,
            TableInst::Run => 0,
            TableInst::Copy { size, .. } => *size,
            TableInst::NoOp => 0,//ambiguous
        }
    }
}

// Define a struct for a code table entry that can represent up to two instructions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CodeTableEntry {
    pub first: TableInst,
    pub second: TableInst,
}
impl CodeTableEntry{
    pub fn sec_is_noop(&self) -> bool {
        self.second == TableInst::NoOp
    }

}

// Implement the function to generate the default VCDIFF code table.
pub fn generate_default_code_table() -> [CodeTableEntry; 256] {
    let mut table: [CodeTableEntry; 256] = [CodeTableEntry { first: TableInst::NoOp, second: TableInst::NoOp }; 256];

    // Entry 0: RUN 0 NOOP 0
    table[0] = CodeTableEntry { first: TableInst::Run , second: TableInst::NoOp };

    // Entries for ADD instructions with sizes 0 and [1-17]
    for i in 1..=18 {
        table[i] = CodeTableEntry { first: TableInst::Add { size: i as u8 - 1 }, second: TableInst::NoOp };
    }

    // Entries for COPY instructions with modes [0-8] and sizes 0 and [4-18]
    for mode in 0..=8u8 {
        for size in 0..=15 {
            let index = 19 + mode * 16 + size;
            table[index as usize] = CodeTableEntry { first: TableInst::Copy { size: if size == 0 { 0 } else { size + 3 }, mode }, second: TableInst::NoOp };
        }
    }

    // Combined ADD and COPY instructions
    let mut index = 163;
    for add_size in 1..=4 {
        for copy_mode in 0..=5 {
            for copy_size in 4..=6 {
                table[index] = CodeTableEntry {
                    first: TableInst::Add { size: add_size },
                    second: TableInst::Copy { size: copy_size, mode: copy_mode },
                };
                index += 1;
            }
        }
    }

    // Combined ADD and COPY instructions with specific sizes for COPY
    for add_size in 1..=4 {
        for copy_mode in 6..=8 {
            table[index] = CodeTableEntry {
                first: TableInst::Add { size: add_size },
                second: TableInst::Copy { size: 4, mode: copy_mode },
            };
            index += 1;
        }
    }

    // The last line for COPY ADD combinations
    for mode in 0..=8 {
        table[index] = CodeTableEntry {
            first: TableInst::Copy { size: 4, mode },
            second: TableInst::Add { size: 1 },
        };
        index += 1;
    }

    table
}

pub fn decode_integer<R: std::io::Read>(source: &mut R) -> std::io::Result<(u64,usize)> {
    let mut value = 0u64;
    let mut byte_count = 0; // To keep track of the number of bytes read

    loop {
        let mut byte = [0u8; 1];
        source.read_exact(&mut byte)?;

        // Calculate the shift for this byte. Initial bytes are maximally shifted
        // and the adjustment is made later.
        byte_count += 1;
        let shift = 63 - (byte_count * 7); // Adjust shift based on byte count
        let bits = (byte[0] & 0x7F) as u64;
        value = value.checked_add(bits << shift).expect("Overflow");

        if byte[0] & 0x80 == 0 { // If this is the last byte
            break;
        }
    }
    //each byte is 7 bits
    //we stored the first bit in position 63
    //once we know how many bytes we read, we can shift the value to the right
    //we need the first bit to be byte_count * 7 position
    //so if it is 4 bytes, we need to shift right 63 - 4 * 7 = 35 for the first bit to be in position 28
    value >>= 63 - byte_count * 7;

    Ok((value,byte_count))
}


pub fn encode_integer<W: std::io::Write>(mut sink: W, mut value: u64) -> std::io::Result<()> {
    let mut bytes = [0u8; 10]; // Maximum size needed for a u64 value in base 128 encoding
    let mut index = bytes.len(); // Start from the end of the array

    if value == 0 {
        return sink.write_all(&[0]);
    }

    while value > 0 {
        index -= 1; // Move towards the front of the array
        let digit = if index == bytes.len() - 1 {
            // If it's the first byte, do not set the MSB
            value % 128
        } else {
            // Otherwise, set the MSB
            (value % 128) | 128
        };
        bytes[index] = digit as u8;
        value /= 128;
    }

    // Write all the bytes at once, starting from the first used byte in the array
    sink.write_all(&bytes[index..])
}

pub fn integer_encoded_size(value: u64) -> usize {
    if value == 0 {
        return 1;
    }

    let mut byte_count = 0;
    let mut value = value;
    while value > 0 {
        byte_count += 1;
        value /= 128;
    }
    byte_count
}



#[cfg(test)]
mod test_super {
    use super::*;
    // As given in RFC 3284, pg 4.
    const CORRECT_ENCODING: [u8; 4] = [58 | 0x80, 111 | 0x80, 26 | 0x80, 21];
    const TEST_VALUE: u64 = 123456789;


    #[test]
    fn test_encode_specific_value() -> std::io::Result<()> {
        // Encode the value
        let mut buffer = Vec::new();
        encode_integer(&mut buffer, TEST_VALUE)?;

        // Check that the encoded bytes match the expected sequence
        assert_eq!(buffer, CORRECT_ENCODING, "Encoding mismatch");

        Ok(())
    }

    #[test]
    fn test_decode_specific_value() -> std::io::Result<()> {
        // Decode the bytes back to a u64
        let mut cursor = std::io::Cursor::new(&CORRECT_ENCODING);
        let (decoded,_) = decode_integer(&mut cursor)?;

        // Verify the decoded value matches the original
        assert_eq!(decoded, TEST_VALUE, "Decoding mismatch");

        Ok(())
    }
    #[test]
    fn test_len() -> () {
        assert_eq!(integer_encoded_size(TEST_VALUE), CORRECT_ENCODING.len(), "Length mismatch");
    }

}

use TableInst::*;
pub const VCD_C_TABLE: [CodeTableEntry;256] = [
    CodeTableEntry { first: Run , second: NoOp },
    CodeTableEntry { first: Add { size: 0 }, second: NoOp },
    CodeTableEntry { first: Add { size: 1 }, second: NoOp },
    CodeTableEntry { first: Add { size: 2 }, second: NoOp },
    CodeTableEntry { first: Add { size: 3 }, second: NoOp },
    CodeTableEntry { first: Add { size: 4 }, second: NoOp },
    CodeTableEntry { first: Add { size: 5 }, second: NoOp },
    CodeTableEntry { first: Add { size: 6 }, second: NoOp },
    CodeTableEntry { first: Add { size: 7 }, second: NoOp },
    CodeTableEntry { first: Add { size: 8 }, second: NoOp },
    CodeTableEntry { first: Add { size: 9 }, second: NoOp },
    CodeTableEntry { first: Add { size: 10 }, second: NoOp },
    CodeTableEntry { first: Add { size: 11 }, second: NoOp },
    CodeTableEntry { first: Add { size: 12 }, second: NoOp },
    CodeTableEntry { first: Add { size: 13 }, second: NoOp },
    CodeTableEntry { first: Add { size: 14 }, second: NoOp },
    CodeTableEntry { first: Add { size: 15 }, second: NoOp },
    CodeTableEntry { first: Add { size: 16 }, second: NoOp },
    CodeTableEntry { first: Add { size: 17 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 0, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 4, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 5, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 6, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 7, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 8, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 9, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 10, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 11, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 12, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 13, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 14, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 15, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 16, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 17, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 18, mode: 0 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 0, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 4, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 5, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 6, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 7, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 8, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 9, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 10, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 11, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 12, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 13, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 14, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 15, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 16, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 17, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 18, mode: 1 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 0, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 4, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 5, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 6, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 7, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 8, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 9, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 10, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 11, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 12, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 13, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 14, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 15, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 16, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 17, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 18, mode: 2 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 0, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 4, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 5, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 6, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 7, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 8, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 9, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 10, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 11, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 12, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 13, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 14, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 15, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 16, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 17, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 18, mode: 3 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 0, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 4, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 5, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 6, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 7, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 8, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 9, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 10, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 11, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 12, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 13, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 14, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 15, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 16, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 17, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 18, mode: 4 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 0, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 4, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 5, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 6, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 7, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 8, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 9, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 10, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 11, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 12, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 13, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 14, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 15, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 16, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 17, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 18, mode: 5 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 0, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 4, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 5, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 6, mode: 6 }, second: NoOp }, //118
    CodeTableEntry { first: Copy { size: 7, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 8, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 9, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 10, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 11, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 12, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 13, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 14, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 15, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 16, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 17, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 18, mode: 6 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 0, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 4, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 5, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 6, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 7, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 8, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 9, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 10, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 11, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 12, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 13, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 14, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 15, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 16, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 17, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 18, mode: 7 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 0, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 4, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 5, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 6, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 7, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 8, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 9, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 10, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 11, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 12, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 13, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 14, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 15, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 16, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 17, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Copy { size: 18, mode: 8 }, second: NoOp },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 0 } }, //163
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 5, mode: 0 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 6, mode: 0 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 1 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 5, mode: 1 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 6, mode: 1 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 2 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 5, mode: 2 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 6, mode: 2 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 3 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 5, mode: 3 } }, //173
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 6, mode: 3 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 4 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 5, mode: 4 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 6, mode: 4 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 5 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 5, mode: 5 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 6, mode: 5 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 0 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 5, mode: 0 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 6, mode: 0 } }, //183
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 1 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 5, mode: 1 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 6, mode: 1 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 2 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 5, mode: 2 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 6, mode: 2 } }, //189
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 3 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 5, mode: 3 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 6, mode: 3 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 4 } }, //193
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 5, mode: 4 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 6, mode: 4 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 5 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 5, mode: 5 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 6, mode: 5 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 0 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 0 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 0 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 1 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 1 } }, //203
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 1 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 2 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 2 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 2 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 3 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 3 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 3 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 4 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 4 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 4 } }, //213
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 5 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 5 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 5 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 0 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 0 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 0 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 1 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 1 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 1 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 2 } }, //223
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 2 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 2 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 3 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 3 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 3 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 4 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 4 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 4 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 5 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 5 } }, //233
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 5 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 6 } }, //235
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 7 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 8 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 6 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 7 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 8 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 6 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 7 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 8 } }, //243
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 6 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 7 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 8 } },
    CodeTableEntry { first: Copy { size: 4, mode: 0 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 1 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 2 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 3 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 4 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 5 }, second: Add { size: 1 } }, //253
    CodeTableEntry { first: Copy { size: 4, mode: 6 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 7 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 8 }, second: Add { size: 1 } }
];