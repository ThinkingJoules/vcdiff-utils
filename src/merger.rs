use std::io::{Read, Seek};

use crate::translator::{ProcessInst, VCDTranslator};


pub fn merge<R: Read + Seek>(sequential_patches:&[VCDTranslator<R>]) -> std::io::Result<Vec<ProcessInst>> {
    let mut cur_o_pos = 0;

    todo!()
}
