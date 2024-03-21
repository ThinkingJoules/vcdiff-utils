use std::io::{Read,Seek,Write};

use crate::{decoder::{VCDDecoder, COPY}, reader::VCDReader};

pub fn apply_patch<R:Read+Seek,W:Write>(patch:R,original:R,sink:W) -> std::io::Result<()> {
    let mut decoder = VCDDecoder::new(VCDReader::new(patch)?);

    todo!()
}

