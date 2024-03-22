
pub mod reader;
pub mod decoder;
pub mod translator;
pub mod merger;
pub mod encoder;
pub mod applicator;
//We only allow windows of less than 2GB, so we can use u32 for all sizes.
//The patch file itself can be larger, but we can't have a window that large.
pub const MAGIC:[u8;4] = [b'V'|0x80, b'C'|0x80, b'D'|0x80, 0];

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
///Inlined of Decoded RUN instruction
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RUN{
    pub len: u32,
    pub byte: u8
}

#[derive(Debug)]
pub struct Cache {
    near: [usize; Self::S_NEAR], // Fixed array size of 4
    next_slot: usize,
    same: [usize; Self::S_SAME * 256], // Fixed array size of 3 * 256
}

impl Cache {
    const S_NEAR: usize = 4;
    const S_SAME: usize = 3;
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

    pub fn addr_encode(&mut self, addr: usize, here: usize) -> (u32, u8) { // Return encoded address and mode
        let res = self.peek_addr_encode(addr, here);
        self.update(addr);
        res
    }
    pub fn peek_addr_encode(&self, addr: usize, here: usize) -> (u32, u8){
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
        let distance = addr % (Self::S_SAME * 256);
        if self.same[distance] == addr {
            best_distance = distance % 256;
            best_mode = Self::S_NEAR + 2 + distance / 256;
        }
        (best_distance as u32, best_mode as u8)
    }

    /// **addr_section**, is a reader that is positioned at the start of the address section
    /// It is only advanced by this function, so it remembers what it has read
    /// **here** is the current position in the target output
    /// **mode** is the mode of the address
    /// returns (address, bytes_read_from_addr_section)
    pub fn addr_decode<R:Read>(&mut self, addr_section: &mut R, here: u64, mode: usize) ->  std::io::Result<(u64,usize)> {
        let addr;
        let read;
        if mode == 0 { // VCD_SELF
            let (x,a) = decode_integer(addr_section)?;
            addr = x;
            read = a;
        } else if mode == 1 { // VCD_HERE
            let (x,a) = decode_integer(addr_section)?;
            addr = here - x;
            read = a;
        } else if mode >= 2 && mode - 2 < Cache::S_NEAR {  // Near cache
                let near_index = mode - 2;
                let (x,a) = decode_integer(addr_section)?;
                addr = self.near[near_index] as u64 + x;
                read = a;
        } else { // Same cache
            let m = mode - (2 + Cache::S_NEAR);
            let mut byte = [0u8];
            addr_section.read_exact(&mut byte)?;
            let same_index = m * 256 + byte[0] as usize;
            addr = self.same[same_index] as u64;
            read = 1;
        }
        self.update(addr as usize);
        Ok((addr,read))
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

// Define a struct for a code table entry that can represent up to two instructions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CodeTableEntry {
    first: TableInst,
    second: TableInst,
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
        value += bits << shift;

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
    use std::io::Cursor;
    #[test]
    fn test_vcd_self_mode() {
        // Test VCD_SELF mode (address encoded by itself)
        let mut cache = Cache::new();
        let address = 100;
        let here = 200;

        // Encode
        let (encoded_value, mode) = cache.addr_encode(address, here);

        // Expected values based on RFC 3284 section 5.3
        let expected_value = address;
        let expected_mode = 0; // VCD_SELF

        assert_eq!(encoded_value, expected_value as u32);
        assert_eq!(mode, expected_mode);

        // Mock reading the encoded value from address section
        let mut reader = Cursor::new(vec![100]); // Encoded value 100

        // Decode
        let (decoded_address, _) = cache.addr_decode(&mut reader, here as u64, mode as usize).unwrap();

        // Expected decoded address based on the encoding
        assert_eq!(decoded_address, expected_value as u64);
    }
    #[test]
    fn test_vcd_here_mode() {
        // Test VCD_HERE mode (address encoded as here - addr)
        let mut cache = Cache::new();
        let address = 100;
        let here = 150;

        // Encode
        let (encoded_value, mode) = cache.addr_encode(address, here);

        // Expected values based on RFC 3284 section 5.3
        let expected_value = here - address;
        let expected_mode = 1; // VCD_HERE

        assert_eq!(encoded_value, expected_value as u32); // Cast to u32 for size comparison
        assert_eq!(mode, expected_mode);

        // Mock reading the encoded value from address section
        let mut reader = Cursor::new(vec![50]); // Encoded value 100

        // Decode
        let (decoded_address,_) = cache.addr_decode(&mut reader, here as u64, mode as usize).unwrap();

        // Expected decoded address based on the encoding
        assert_eq!(decoded_address, here as u64 - expected_value as u64);
    }
    #[test]
    fn test_near_cache_mode() {
        // Test encoding and decoding using the near cache
        let mut cache = Cache::new();
        let address = 110;
        let here = 210;

        // Update near cache to simulate a previously encoded address
        cache.update(100);

        // Encode
        let (encoded_value, mode) = cache.addr_encode(address, here);

        // Expected values based on RFC 3284 section 5.3
        let expected_value = address - 100; // Distance from address to element in near cache
        let expected_mode = 2; // Near cache mode (offset 2, since fixed size is 4)

        assert_eq!(encoded_value, expected_value as u32);
        assert_eq!(mode, expected_mode);

        // Mock reading the encoded value from address section
        let mut reader = Cursor::new(vec![10]); // Encoded value 10
        // Decode
        let (decoded_address,_) = cache.addr_decode(&mut reader, here as u64, mode as usize).unwrap();

        // Expected decoded address based on the encoding
        assert_eq!(decoded_address, (100 + expected_value) as u64);
    }
    #[test]
    fn test_same_cache_mode() {
        let mut cache = Cache::new();
        let base_addr = 5 * 256;  // Example base address

        // Update cache with the base address to populate the same cache
        cache.update(base_addr);

        // Target address with the same byte value as base_addr,
        // but at a different offset
        let target_addr = base_addr; // Offset by a value that's not a multiple of 256

        // Encode
        let (encoded_value, mode) = cache.addr_encode(target_addr, 2100); // 'here' is irrelevant

        // Expected Values
        let offset = target_addr % (Cache::S_SAME * 256); // Relative offset
        let byte_value = target_addr % 256;
        let expected_mode = Cache::S_NEAR + 2 + offset / 256;

        assert_eq!(encoded_value as usize, byte_value); // Encoded value is the byte itself
        assert_eq!(mode, expected_mode as u8);

        // Mock reading the encoded value (offset and byte value combined)
        let mut reader = Cursor::new(vec![(offset % 256) as u8, byte_value as u8]);

        // Decode
        let (decoded_address, _) = cache.addr_decode(&mut reader, 0, mode as usize).unwrap();

        // Expected decoded address based on the encoding
        assert_eq!(decoded_address, target_addr as u64);
    }

}
use std::io::Read;

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
    CodeTableEntry { first: Copy { size: 6, mode: 6 }, second: NoOp },
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
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 6, mode: 3 } }, //193
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 4 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 5, mode: 4 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 6, mode: 4 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 5 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 5, mode: 5 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 6, mode: 5 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 0 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 0 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 0 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 1 } }, //203
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 1 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 1 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 2 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 2 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 2 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 3 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 3 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 3 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 4 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 4 } }, //213
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 4 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 5 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 5, mode: 5 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 6, mode: 5 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 0 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 0 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 0 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 1 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 1 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 1 } }, //223
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 2 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 2 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 2 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 3 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 3 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 3 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 4 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 4 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 4 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 5 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 5, mode: 5 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 6, mode: 5 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 6 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 7 } },
    CodeTableEntry { first: Add { size: 1 }, second: Copy { size: 4, mode: 8 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 6 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 7 } },
    CodeTableEntry { first: Add { size: 2 }, second: Copy { size: 4, mode: 8 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 6 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 7 } },
    CodeTableEntry { first: Add { size: 3 }, second: Copy { size: 4, mode: 8 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 6 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 7 } },
    CodeTableEntry { first: Add { size: 4 }, second: Copy { size: 4, mode: 8 } },
    CodeTableEntry { first: Copy { size: 4, mode: 0 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 1 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 2 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 3 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 4 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 5 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 6 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 7 }, second: Add { size: 1 } },
    CodeTableEntry { first: Copy { size: 4, mode: 8 }, second: Add { size: 1 } }
];