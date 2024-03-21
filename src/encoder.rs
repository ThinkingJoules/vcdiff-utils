use std::io::Write;

use crate::translator::Inst;

pub fn encode<W: Write>(inst:&[Inst],sink:W) -> std::io::Result<()> {
    //first we write file header

    //then we determine the windows one at a time. Encode the win hdr and then all the data



    Ok(())
}