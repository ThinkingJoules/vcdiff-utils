use std::{fmt::Debug, io::{Read,Seek,Write}, ops::Range};

use vcdiff_common::{CopyType, Inst, Instruction, WinIndicator, WindowSummary, ADD, COPY, RUN};
use vcdiff_reader::{read_header, read_window_header, VCDReader, VCDiffReadMsg};



fn find_dep_ranges(summaries: &[WindowSummary])->Vec<Range<u64>>{
    let mut ranges = Vec::new();
    for ws in summaries.iter().rev() {
        if let WinIndicator::VCD_TARGET = ws.win_indicator {
            let ssp = ws.source_segment_position.unwrap() as u64;
            let sss = ws.source_segment_size.unwrap() as u64;
            ranges.push(ssp..ssp+sss);
        }
    }
    let mut ranges = merge_ranges(ranges);
    //sort with the smallest last
    ranges.sort_by(|a,b|b.start.cmp(&a.start));
    ranges
}
pub fn gather_summaries<R: Read + Seek>(patch_data:&mut R)-> std::io::Result<Vec<WindowSummary>>{
    let header = read_header(patch_data)?;
    let mut summaries = Vec::new();
    let mut win_start_pos = header.encoded_size() as u64;
    while let Ok(ws) = read_window_header(patch_data, win_start_pos) {
        win_start_pos = ws.end_of_window();
        summaries.push(ws);
        patch_data.seek(std::io::SeekFrom::Start(win_start_pos))?;
    }
    Ok(summaries)
}
fn merge_ranges<T: Ord + Copy>(ranges: Vec<Range<T>>) -> Vec<Range<T>> {
    let mut result: Vec<Range<T>> = Vec::new();
    let mut sorted_ranges = ranges;

    // 1. Sort the ranges by their start values
    sorted_ranges.sort_by(|a, b| a.start.cmp(&b.start));

    // 2. Iterate through the sorted ranges
    for range in sorted_ranges {
        // 3. If the result list is empty or the current range doesn't overlap
        //    with the last range in the result, simply add it to the result.
        if result.is_empty() || range.start > result.last().unwrap().end {
            result.push(range);
        } else {
            // 4. Overlap exists: Extend the end of the last range in the result
            //    to the maximum end of the overlapping ranges.
            let last_index = result.len() - 1;
            result[last_index].end = std::cmp::max(result[last_index].end, range.end);
        }
    }

    result
}
fn range_overlap<T: Ord + Copy>(range1: &Range<T>, range2: &Range<T>) -> Option<Range<T>> {
    let start = std::cmp::max(range1.start, range2.start);
    let end = std::cmp::min(range1.end, range2.end);

    if start < end {
        Some(Range { start, end })
    } else {
        None
    }
}

pub fn apply_patch<R:Read+Seek+Debug,W:Write>(patch:&mut R,mut src:Option<&mut R>,sink:&mut W) -> std::io::Result<()> {
    //to avoid the Read+Seek bound on sink,
    //we need to scan the whole patch file so we can cache the TargetSourced windows
    let windows = gather_summaries(patch)?;
    let dependencies = find_dep_ranges(&windows);
    let mut reader = VCDReader::new(patch)?;
    let mut ws = None;
    let mut cur_u = Vec::new();
    let mut sink_cache = SparseCache::new();
    let mut o_pos = 0;
    let mut t_pos = 0;

    //let mut stats = Stats::default();

    loop{
        let msg = reader.next()?;
        match msg{
            VCDiffReadMsg::WindowSummary(w) => {
                let WindowSummary{
                    win_indicator,
                    source_segment_size,
                    source_segment_position,
                    size_of_the_target_window, ..
                } = &w;
                let needed_capacity = match source_segment_size{
                    Some(sss) => {
                        sss + *size_of_the_target_window
                    },
                    _ => *size_of_the_target_window,
                } as usize;
                //stats = Stats::default();
                debug_assert!(cur_u.is_empty());
                if cur_u.capacity() < needed_capacity {
                    cur_u.reserve(needed_capacity - cur_u.capacity());
                }
                match (win_indicator,source_segment_position,source_segment_size,src.as_mut()) {
                    (WinIndicator::Neither,_,_,_) => (),
                    (WinIndicator::VCD_SOURCE, Some(ssp), Some(sss),Some(s)) => {
                        s.seek(std::io::SeekFrom::Start(*ssp))?;
                        cur_u.resize(*sss as usize, 0);
                        s.read_exact(&mut cur_u[..*sss as usize])?;
                    },
                    (WinIndicator::VCD_TARGET, Some(ssp), Some(sss),_) => {
                        //pull from our sparse cache
                        let slice = sink_cache.get_src_subslice(*ssp..*ssp+*sss);
                        cur_u.extend_from_slice(slice); //copy Trgt to S in U
                    },
                    _ => panic!("Invalid window configuration"),
                }
                ws = Some(w)
            },
            VCDiffReadMsg::Inst { first, second, .. } => {
                for inst in [Some(first),second]{
                    if inst.is_none() {break;}
                    let inst = inst.unwrap();
                    let len_in_o = inst.len_in_o() as usize;
                    match inst {
                        Inst::Add(ADD{ p_pos,.. }) => {
                            let patch_r = reader.get_reader(p_pos)?;
                            let mut slice = vec![0u8;len_in_o];
                            patch_r.read_exact(&mut slice)?;
                            cur_u.append(&mut slice);
                            //stats.add();
                        },
                        Inst::Copy(COPY{  u_pos, len:copy_in_u,copy_type }) => {
                            let u_pos = u_pos as usize;
                            //first figure out if this is an implicit sequence
                            let cur_end = cur_u.len();
                            //let copy_end = u_pos + copy_in_u as usize;
                            debug_assert!(u_pos < cur_end,"{:?} >= {:?}",u_pos,cur_end);
                            // if copy_end > cur_end {
                            //     stats.seq();
                            // }else{
                            //     stats.copy();
                            // }
                            unsafe{
                                // Get raw pointers
                                let cur_u_ptr = cur_u.as_mut_ptr();
                                let slice_ptr = cur_u.as_ptr();
                                // Extend 'cur_u' with uninitialized memory, making it long enough
                                cur_u.set_len(cur_u.len() + len_in_o);

                                // Copy data in a loop
                                let mut amt_copied = 0;
                                while amt_copied < len_in_o {
                                    let (copy_len,source_offset) = if matches!(copy_type,CopyType::CopyQ { .. }) {
                                        let seq_len = cur_end-u_pos;
                                        let copy_len = std::cmp::min(len_in_o - amt_copied, seq_len);
                                        let seq_offset = amt_copied % seq_len;
                                        (copy_len,seq_offset+u_pos)
                                    } else{//regular copy
                                        //this should run the loop a single time
                                        debug_assert!(amt_copied == 0);
                                        (copy_in_u as usize, u_pos as usize)
                                    };
                                    // Calculate offsets for copying
                                    let dest_offset = cur_end + amt_copied;
                                    // Use ptr::copy_nonoverlapping for the memory copy
                                    std::ptr::copy_nonoverlapping(
                                        slice_ptr.add(source_offset),
                                        cur_u_ptr.add(dest_offset),
                                        copy_len
                                    );
                                    amt_copied += copy_len;
                                }
                            }
                        },
                        Inst::Run(RUN{  byte, .. }) => {
                            //stats.run();
                            cur_u.extend(std::iter::repeat(byte).take(len_in_o));
                        },
                    }
                    t_pos += len_in_o;
                }
            },
            VCDiffReadMsg::EndOfWindow => {
                //check dependencies for ranges
                //copy value in U (in T) to the SparseCache
                //our dependencies can be across window boundaries
                //so we need to cache what we can

                //currently o_pos is at the start of our cur_window
                let summary = ws.take().unwrap();
                let t_len = summary.size_of_the_target_window;
                //println!("Stats: {:?}",stats);
                assert_eq!(t_pos,t_len as usize,"{:?}",summary);
                //t_start should logically line up with the current value in o_pos
                //however they will be a different actual value
                let t_start = summary.source_segment_size.unwrap_or(0) as usize;
                let to_cache = merge_ranges(find_intersections(&(o_pos..o_pos+t_len), &dependencies));
                for Range { start, end } in to_cache {
                    let len = end-start;
                    sink_cache.add(start, len as usize, &mut std::io::Cursor::new(&cur_u),t_start,o_pos).unwrap();
                }
                o_pos += t_len; //now we move the o_pos
                t_pos = 0;
                sink.write_all(&cur_u[t_start..t_start+t_len as usize])?;
                cur_u.clear();
            },
            VCDiffReadMsg::EndOfFile => break,
        }
        //dbg!(&cur_u);
    }
    Ok(())
}


#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct Segment{
    src_start: u64,
    buf_start: usize,
    len: usize,
}

impl Segment {
    fn src_end(&self) -> u64 {
        self.src_start + self.len as u64
    }
}
///Sparse or Overlapping cache struct
///Used to spare parts of a src buffer that are needed, sparsely
pub struct SparseCache
{
    buffer: Vec<u8>,
    segment_map: Vec<Segment>,   // Maps (start, end) ranges of inner buffer to entries of K
}

impl SparseCache
{
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            segment_map: Vec::new(),
        }
    }

    pub fn add<R:Read+Seek>(&mut self, o_pos: u64, len: usize, src: &mut R, t_start:usize, o_start_at_t_start:u64) -> Result<(), std::io::Error> {

        let read_segments = self.prepare_to_add(o_pos, len,t_start,o_start_at_t_start);
        for ((seg_start, seg_len),buf_pos) in read_segments {
            src.seek(std::io::SeekFrom::Start(seg_start as u64))?;
            src.read_exact(&mut self.buffer[buf_pos..buf_pos+seg_len])?;
        }
        Ok(())
    }
    ///Returns ((src_start, src_len), buffer_start); the len is the same, obviously
    fn prepare_to_add(&mut self,o_pos: u64, len: usize, t_start:usize, o_start_at_t_start:u64) -> Vec<((u64, usize), usize)> {
        let mut missing_byte_sections = Vec::new();
        let end = o_pos + len as u64; // Define the end of the query range
        let mut current_start = o_pos; // Current start to look for in the query range
        //this function creates the proper new map, but doesn't touch the buffer
        //it returns the static insertion points (that is, each insertion point is independent)
        //to properly fix the buffer, we would need to progressively shift later segments more.
        let mut new_map = Vec::with_capacity(self.segment_map.len() + 1);
        let mut read_pos = (current_start - o_start_at_t_start) + t_start as u64;
        // Iterate over the segments
        let mut cur_shift = 0;
        for mut seg in self.segment_map.drain(..) {
            let segment_end = seg.src_end();
            let Segment { src_start, buf_start, .. } = &mut seg;
            if segment_end <= current_start {
                new_map.push(seg);
                // Skip segments entirely before the query range
                continue;
            }
            if current_start < *src_start && cur_shift == 0{//this is where we insert our new segment
                new_map.push(Segment{
                    src_start: o_pos,
                    buf_start:*buf_start,
                    len,
                });
            }
            if *src_start > current_start{
                // Found a gap before the current segment starts
                // min required for a segment wholly before the current segment found.
                let end_pos = std::cmp::min(*src_start, end);
                let read_len = (end_pos - current_start) as usize;
                cur_shift += read_len;
                //we start inserting at the buf_start position for the overlapping segment
                //this will require us to shift the
                missing_byte_sections.push(((read_pos, read_len), *buf_start));
            }
            //if the segment is not before, all following must be shifted
            *buf_start += cur_shift;
            new_map.push(seg);

            // Update current_start to the end of the current or overlapping segment
            current_start = std::cmp::max(current_start, segment_end);
            read_pos = (current_start - o_start_at_t_start) + t_start as u64;
            if current_start >= end {
                // If we've covered the query range, stop checking
                break;
            }
        }

        // Check for a gap at the end of the range
        if current_start < end {
            //append to the end of the buffer
            if missing_byte_sections.is_empty(){
                new_map.push(Segment{
                    src_start: o_pos,
                    buf_start:self.buffer.len(),
                    len,
                });
            }
            missing_byte_sections.push(((read_pos, (end - current_start) as usize), self.buffer.len()));
        }
        self.segment_map = new_map;
        self.prepare_buffer(&missing_byte_sections);

        missing_byte_sections
    }
    ///Slice of ((_src_start, len), buf_start)
    fn prepare_buffer(&mut self, splice_segments: &[((u64, usize),usize)]) {
        if splice_segments.is_empty() {
            return;
        }
        // Update buffer with single allocation
        let total_new_elements: usize = splice_segments.iter().map(|((_, v),_)| v).sum();
        let cur_size = self.buffer.len();
        let final_size = cur_size + total_new_elements;

        // Reserving capacity
        let mut new_vec = vec![0; final_size];

        // Rebuilding the vector
        let mut old_buf_pos = 0;
        let mut new_vec_pos = 0;
        for ((_, insert_len),orig_buf_start) in splice_segments.iter() {
            if old_buf_pos < cur_size{ //try to copy from the old buffer first
                let copy_len = orig_buf_start - old_buf_pos;
                new_vec[new_vec_pos..new_vec_pos+copy_len].copy_from_slice(&self.buffer[old_buf_pos..old_buf_pos+copy_len]);
                old_buf_pos += copy_len;
                new_vec_pos += copy_len
            }
            new_vec_pos += insert_len; //we initialized to 0, so we just add the insert_len

        }
        //copy the remaining elements
        if old_buf_pos < cur_size {
            new_vec[new_vec_pos..new_vec_pos+cur_size-old_buf_pos].copy_from_slice(&self.buffer[old_buf_pos..cur_size]);
        }
        self.buffer = new_vec;

    }

    pub fn get_src_subslice(&self, src_range: Range<u64>) -> &[u8] {
        //panic if the slice is not already fully within some existing keys range.
        // Extract bounds from the provided range
        let mut slice_start = None;
        let mut last_end = 0;
        for Segment { src_start, buf_start, len } in &self.segment_map {
            let len = *len as u64;
            if let Some(in_range) = range_overlap(&(*src_start..*src_start + len), &src_range) {
                let slice_end = src_start + len;
                if slice_start.is_some() {
                    if last_end != *src_start {
                        panic!("Non-contiguous segments in the cache");
                    }
                }else if in_range.start <= src_range.start{
                    slice_start = Some(*buf_start + (src_range.start - src_start) as usize);
                }
                if src_range.end <= slice_end {
                    let slice_end = *buf_start + len as usize - (slice_end - src_range.end) as usize;
                    return &self.buffer[slice_start.unwrap()..slice_end];
                }
            }
            last_end = *src_start + len;
        }

        panic!("The slice is not fully within some existing keys range: {:?}",self.segment_map);
    }
}

pub(crate) fn find_intersections<T: Ord + Copy>(reference_set: &Range<T>, test_sets: &[Range<T>]) -> Vec<Range<T>> {
    let mut intersections = Vec::new();

    for test_set in test_sets {
        if let Some(overlap) = range_overlap(&reference_set, test_set) {
            intersections.push(overlap);
        }
    }

    intersections
}
// ///Used for debugging
// #[derive(Debug,Default)]
// struct Stats{
//     add: usize,
//     copy: usize, //non-implicit sequence copys
//     run: usize,
//     seq: usize, //implicit sequence copys
// }
// impl Stats{
//     fn add(&mut self){
//         self.add += 1;
//     }
//     fn copy(&mut self){
//         self.copy += 1;
//     }
//     fn run(&mut self){
//         self.run += 1;
//     }
//     fn seq(&mut self){
//         self.seq += 1;
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    #[test]
    fn test_src_apply(){
        // "hello" -> "Hello! Hello!"
        let mut src = Cursor::new("hello".as_bytes().to_vec());

        //from encoder tests
        let patch = vec![
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
            235, //ADD1 COPY4_mode6
            183, //ADD2 COPY6_mode0
            0,
            4,
        ];
        let mut patch = Cursor::new(patch);
        let mut sink = Vec::new();
        apply_patch(&mut patch,Some(&mut src),&mut sink).unwrap();
        assert_eq!(sink, "Hello! Hello!".as_bytes());
    }
    #[test]
    fn test_complex_apply(){
        // "hello" -> "Hello! Hello!"
        let mut src = Cursor::new("hello".as_bytes().to_vec());

        //from encoder tests
        let patch = vec![
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
            2, //ADD1 (i = 13)
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
            253, //COPY4_mode5 ADD1
            0, //addr 0
            2, //win_indicator VCD_TARGET
            6, //SSS
            0, //SSP
            9, //delta window size
            7, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            2, //length of instructions and size
            1, //length of addr
            32, //data section ' '
            2, //ADD1 NOOP
            118, //COPY6_mode6 NOOP
            0, //addr 0
        ];
        let mut patch = Cursor::new(patch);
        let mut sink = Vec::new();
        apply_patch(&mut patch,Some(&mut src),&mut sink).unwrap();
        assert_eq!(sink, "Hello! Hello!".as_bytes());
    }

    #[test]
    fn test_kitchen_sink(){
        // "hello" -> "Hello! Hello! Hell..."
        let mut src = Cursor::new("hello".as_bytes().to_vec());

        //from encoder tests
        let patch = vec![
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
            253, //COPY4_mode5 ADD1
            0, //addr 0
            2, //win_indicator VCD_TARGET
            6, //SSS
            0, //SSP
            9, //delta window size
            7, //target window size
            0, //delta indicator
            1, //length of data for ADDs and RUN/
            2, //length of instructions and size
            1, //length of addr
            32, //data section ' '
            2, //ADD1 NOOP
            118, //COPY6_mode6 NOOP
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
            117, //ADD1 COPY5_mode6
            2, //Add1 NOOP
            35, //COPY0_mode1
            3, //...size
            0, //addr 0
            1, //addr 1
        ];
        let mut patch = Cursor::new(patch);
        let mut sink = Vec::new();
        apply_patch(&mut patch,Some(&mut src),&mut sink).unwrap();
        assert_eq!(sink, "Hello! Hello! Hell...".as_bytes());

    }

#[test]
    fn test_kitchen_sink2(){
        // "hello world!" -> "Hello! Hello! Hello. "
        let mut src = Cursor::new("hello world!".as_bytes().to_vec());

        //from encoder tests
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
        let mut patch = Cursor::new(patch);
        let mut sink = Vec::new();
        apply_patch(&mut patch,Some(&mut src),&mut sink).unwrap();
        let str = std::str::from_utf8(&sink).unwrap();
        assert_eq!(str, "Hello! Hello! Hello. ");

    }
    #[test]
    fn test_add_and_retrieve() {
        let mut cache = SparseCache::new();
        let mut data = Cursor::new(vec![1, 2, 3, 4]);

        // Add a segment
        cache.add( 3, 1, &mut data,0,0).unwrap();
        cache.add( 0, 2, &mut data,0,0).unwrap();
        cache.add( 2, 2, &mut data,0,0).unwrap();

        // Retrieve the segment
        let result = cache.get_src_subslice(3..4);
        assert_eq!(result, &[4]);

        let result = cache.get_src_subslice(0..2);
        assert_eq!(result, &[1, 2]);

        let result = cache.get_src_subslice(2..4);
        assert_eq!(result, &[3, 4]);
    }
    #[test]
    fn test_overlapping_segments() {
        let mut cache = SparseCache::new();
        let mut data = Cursor::new("hello world".as_bytes());

        cache.add(0, 3, &mut data,0,0).unwrap();
        cache.add( 6, 5, &mut data,0,0).unwrap();

        assert_eq!(cache.get_src_subslice(0..3), "hel".as_bytes());
        assert_eq!(cache.get_src_subslice(6..11), "world".as_bytes());
    }
    #[test]
    fn test_get_subslice() {
        let mut cache = SparseCache::new();
        let first = "ABCDEFGHIJ".as_bytes();
        let f_len = first.len(); //10
        let mut data = Cursor::new(first);

        cache.add(0, 3, &mut data,0,0).unwrap();
        let mut data = Cursor::new("KLMNOP".as_bytes());

        cache.add(13, 3, &mut data,0,f_len as u64).unwrap();

        let subslice1 = cache.get_src_subslice( 0..2);

        assert_eq!(subslice1, "AB".as_bytes());
        let subslice2 = cache.get_src_subslice( 13..15);
        assert_eq!(subslice2, "NO".as_bytes());

    }

    #[test]
    fn test_full_overlap() {
        let reference_set = Range { start: 5, end: 15 };
        let test_sets = vec![Range { start: 10, end: 12 }];
        let expected = vec![Range { start: 10, end: 12 }];
        assert_eq!(find_intersections(&reference_set, &test_sets), expected);
    }

    #[test]
    fn test_partial_overlaps() {
        let reference_set = Range { start: 0, end: 10 };
        let test_sets = vec![
            Range { start: 5, end: 15 },
            Range { start: -2, end: 5 },
            Range { start: 8, end: 12 },
        ];
        let expected = vec![
            Range { start: 5, end: 10 },
            Range { start: 0, end: 5 },
            Range { start: 8, end: 10 },
        ];
        assert_eq!(find_intersections(&reference_set, &test_sets), expected);
    }

    #[test]
    fn test_no_overlap() {
        let reference_set = Range { start: 5, end: 10 };
        let test_sets = vec![
            Range { start: 0, end: 4 },
            Range { start: 11, end: 15 },
        ];
        let expected: Vec<Range<u64>> = vec![]; // Empty result
        assert_eq!(find_intersections(&reference_set, &test_sets), expected);
    }

    #[test]
    fn test_empty_test_sets() {
        let reference_set = Range { start: 1, end: 10 };
        let test_sets: Vec<Range<u64>> = vec![];
        let expected: Vec<Range<u64>> = vec![];
        assert_eq!(find_intersections(&reference_set, &test_sets), expected);
    }
}