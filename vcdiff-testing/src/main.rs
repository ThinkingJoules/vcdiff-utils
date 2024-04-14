use std::fs;
use std::io::{copy, Cursor, Read, Write};
use std::path::Path;
use reqwest::blocking::Client;
use xdelta3::stream::Xd3Config;
use std::time::Instant;
use open_vcdiff_rs_bindings::{FORMAT_STANDARD,encode};

/*
Xdelta3 seems to not produce valid patches.
Alternatively both open-vcdiff and my impl made the same error..
*/
fn main() {
    test_baseline_xdelta().unwrap();
    let res = test_baseline_ovcd_encode_xdelta_decode();
    if res.is_err() {
        println!("Error: {:?}", res.err().unwrap());
    }
    let res = test_baseline_xdelta_encode_vcdiff_decode();
    if res.is_err() {
        println!("Error: {:?}", res.err().unwrap());
    }
    let res = test_baseline_ovcd_encode_vcdiff_decode();
    if res.is_err() {
        println!("Error: {:?}", res.err().unwrap());
    }
    let res = test_merge();
    if res.is_err() {
        println!("Error: {:?}", res.err().unwrap());
    }

}
const DIR_PATH: &str = "../target/downloads";
fn test_baseline_xdelta() -> Result<(), Box<dyn std::error::Error>> {
    prepare_test(DIR_PATH)?;
    let (_,patch) = open_and_hash_file(&Path::new(DIR_PATH).join("patch_a.xdelta3.vcdiff")).unwrap();
    let (_,patch_b) = open_and_hash_file(&Path::new(DIR_PATH).join("patch_b.xdelta3.vcdiff")).unwrap();
    let (_,one) = open_and_hash_file(&Path::new(DIR_PATH).join("317.iso")).unwrap();
    let (h_two,two) = open_and_hash_file(&Path::new(DIR_PATH).join("318.iso")).unwrap();
    let (h_three,_) = open_and_hash_file(&Path::new(DIR_PATH).join("319.iso")).unwrap();
    //make sure xdelta decodes back to the original file
    let mut two_output = Vec::new();
    let start = Instant::now();
    xdelta3::stream::process(
        Xd3Config::new().no_compress(true),
        xdelta3::stream::ProcessMode::Decode,
            patch, one, &mut two_output
    ).unwrap();
    let duration = start.elapsed();
    println!("Time elapsed for decode patch_a is: {:?}", duration);
    let decode_hash= blake3::hash(&two_output);
    assert_eq!(decode_hash, h_two);

    //make sure xdelta decodes back to the original file
    let mut three_output = Vec::new();
    let start = Instant::now();
    xdelta3::stream::process(
        Xd3Config::new().no_compress(true),
        xdelta3::stream::ProcessMode::Decode,
            patch_b, two, &mut three_output
    ).unwrap();
    let duration = start.elapsed();
    println!("Time elapsed for decode patch_b is: {:?}", duration);
    let decode_hash= blake3::hash(&three_output);
    assert_eq!(decode_hash, h_three);

    Ok(())
}
fn test_baseline_ovcd_encode_xdelta_decode() -> Result<(), Box<dyn std::error::Error>> {
    prepare_test(DIR_PATH)?;
    let (_,patch) = open_and_hash_file(&Path::new(DIR_PATH).join("patch_a.ovcd.vcdiff")).unwrap();
    let (_,patch_b) = open_and_hash_file(&Path::new(DIR_PATH).join("patch_b.ovcd.vcdiff")).unwrap();
    let (_,one) = open_and_hash_file(&Path::new(DIR_PATH).join("317.iso")).unwrap();
    let (h_two,two) = open_and_hash_file(&Path::new(DIR_PATH).join("318.iso")).unwrap();
    let (h_three,_) = open_and_hash_file(&Path::new(DIR_PATH).join("319.iso")).unwrap();
    //make sure xdelta decodes back to the original file
    let mut two_output = Vec::new();
    let start = Instant::now();
    xdelta3::stream::process(
        Xd3Config::new().no_compress(true),
        xdelta3::stream::ProcessMode::Decode,
            patch, one, &mut two_output
    )?;
    let duration = start.elapsed();
    println!("Time elapsed for decode patch_a is: {:?}", duration);
    let decode_hash= blake3::hash(&two_output);
    assert_eq!(decode_hash, h_two);

    //make sure xdelta decodes back to the original file
    let mut three_output = Vec::new();
    let start = Instant::now();
    xdelta3::stream::process(
        Xd3Config::new().no_compress(true),
        xdelta3::stream::ProcessMode::Decode,
            patch_b, two, &mut three_output
    )?;
    let duration = start.elapsed();
    println!("Time elapsed for decode patch_b is: {:?}", duration);
    let decode_hash= blake3::hash(&three_output);
    assert_eq!(decode_hash, h_three);

    Ok(())
}

//I think xdelta doesn't follow the spec properly. It doesn't seem to work with my applicator.
fn test_baseline_xdelta_encode_vcdiff_decode() -> Result<(), Box<dyn std::error::Error>> {
    prepare_test(DIR_PATH)?;
    let (_,mut patch) = open_and_hash_file(&Path::new(DIR_PATH).join("patch_a.xdelta3.vcdiff")).unwrap();
    let (_,mut patch_b) = open_and_hash_file(&Path::new(DIR_PATH).join("patch_b.xdelta3.vcdiff")).unwrap();
    let (_,mut one) = open_and_hash_file(&Path::new(DIR_PATH).join("317.iso")).unwrap();
    let (h_two,mut two) = open_and_hash_file(&Path::new(DIR_PATH).join("318.iso")).unwrap();
    let (h_three,_) = open_and_hash_file(&Path::new(DIR_PATH).join("319.iso")).unwrap();
    //make sure xdelta decodes back to the original file
    let mut two_output = Vec::new();
    let start = Instant::now();
    vcdiff_decoder::apply_patch(&mut patch, Some(&mut one), &mut two_output)?;
    let duration = start.elapsed();
    println!("Time elapsed for decode patch_a is: {:?}", duration);
    let decode_hash= blake3::hash(&two_output);
    assert_eq!(decode_hash, h_two);

    //make sure xdelta decodes back to the original file
    let mut three_output = Vec::new();
    let start = Instant::now();
    vcdiff_decoder::apply_patch(&mut patch_b, Some(&mut two), &mut three_output)?;
    let duration = start.elapsed();
    println!("Time elapsed for decode patch_b is: {:?}", duration);
    let decode_hash= blake3::hash(&three_output);
    assert_eq!(decode_hash, h_three);

    Ok(())
}
fn test_baseline_ovcd_encode_vcdiff_decode() -> Result<(), Box<dyn std::error::Error>> {
    prepare_test(DIR_PATH)?;
    let (_,mut patch) = open_and_hash_file(&Path::new(DIR_PATH).join("patch_a.ovcd.vcdiff")).unwrap();
    let (_,mut patch_b) = open_and_hash_file(&Path::new(DIR_PATH).join("patch_b.ovcd.vcdiff")).unwrap();
    let (_,mut one) = open_and_hash_file(&Path::new(DIR_PATH).join("317.iso")).unwrap();
    let (h_two,mut two) = open_and_hash_file(&Path::new(DIR_PATH).join("318.iso")).unwrap();
    let (h_three,_) = open_and_hash_file(&Path::new(DIR_PATH).join("319.iso")).unwrap();
    //make sure xdelta decodes back to the original file
    let mut two_output = Vec::new();
    let start = Instant::now();
    vcdiff_decoder::apply_patch(&mut patch, Some(&mut one), &mut two_output)?;
    let duration = start.elapsed();
    println!("Time elapsed for decode patch_a is: {:?}", duration);
    let decode_hash= blake3::hash(&two_output);
    assert_eq!(decode_hash, h_two);

    //make sure xdelta decodes back to the original file
    let mut three_output = Vec::new();
    let start = Instant::now();
    vcdiff_decoder::apply_patch(&mut patch_b, Some(&mut two), &mut three_output)?;
    let duration = start.elapsed();
    println!("Time elapsed for decode patch_b is: {:?}", duration);
    let decode_hash= blake3::hash(&three_output);
    assert_eq!(decode_hash, h_three);

    Ok(())
}

fn test_merge() -> Result<(), Box<dyn std::error::Error>> {
    prepare_test(DIR_PATH)?;
    let (_,patch) = open_and_hash_file(&Path::new(DIR_PATH).join("patch_a.ovcd.vcdiff")).unwrap();
    let (_,patch_b) = open_and_hash_file(&Path::new(DIR_PATH).join("patch_b.ovcd.vcdiff")).unwrap();
    let (_,mut one) = open_and_hash_file(&Path::new(DIR_PATH).join("317.iso")).unwrap();
    let (h_three,_) = open_and_hash_file(&Path::new(DIR_PATH).join("319.iso")).unwrap();
    let start = Instant::now();
    let merger = vcdiff_merger::Merger::new(patch_b).unwrap().unwrap();
    let merger = merger.merge(patch).unwrap().unwrap();
    let mut summary_patch = Cursor::new(merger.finish().write(Vec::new(), None).unwrap());
    println!("Summary patch done: {} bytes", summary_patch.get_ref().len());
    let duration = start.elapsed();
    println!("Time elapsed in merging patches is: {:?}", duration);
    //apply the merged patch file to the first file

    let mut three_output = Vec::new();
    let start = Instant::now();
    vcdiff_decoder::apply_patch(&mut summary_patch, Some(&mut one), &mut three_output)?;
    let duration = start.elapsed();
    println!("Time elapsed for decode summary_patch is: {:?}", duration);
    let decode_hash= blake3::hash(&three_output);
    assert_eq!(decode_hash, h_three);
    Ok(())
}

fn download_file(url: &str, save_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Check if the file already exists
    if save_path.exists() {
        //println!("File {} already exists. Skipping download.", filename);
        return Ok(());
    }
    let response = Client::new().get(url).send()?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(save_path)?;
    let mut bytes = Cursor::new(response.bytes()?.to_vec());
    copy(&mut bytes, &mut file)?;

    println!("Downloaded {:?}", save_path);
    Ok(())
}
fn open_and_hash_file(path: &Path) -> Result<(blake3::Hash,Cursor<Vec<u8>>), Box<dyn std::error::Error>> {
    let mut file = fs::File::open(path)?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;
    let hash = blake3::hash(&contents);
    Ok((hash,Cursor::new(contents)))
}
fn prepare_test(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(path);
    let urls = vec![
        "https://dl-cdn.alpinelinux.org/alpine/v3.17/releases/x86_64/alpine-standard-3.17.0-x86_64.iso",
        "https://dl-cdn.alpinelinux.org/alpine/v3.18/releases/x86_64/alpine-standard-3.18.0-x86_64.iso",
        "https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/x86_64/alpine-standard-3.19.0-x86_64.iso",
    ];

    // Create the directory if it doesn't exist
    fs::create_dir_all(path)?;
    // Download files sequentially
    for (i,url) in urls.iter().enumerate() {
        let save_path = path.join(format!("{}.iso", 317+i));
        download_file(url, save_path.as_path())?;
    }
    // We want to generate two patch files each, one with xdelta3 and one with open-vcdiff
    // Then we want to try decoding them with xdelta3 and our applicator

    //open or create patch_a.xdelta3.vcdiff
    let patch_file = Path::new(path).join("patch_a.xdelta3.vcdiff");
    if !patch_file.exists() {
        let (_,first) = open_and_hash_file(Path::new(path).join("317.iso").as_path())?;
        let (_,second) = open_and_hash_file(Path::new(path).join("318.iso").as_path())?;
        let mut patch = Vec::new();
        let start = Instant::now();
        xdelta3::stream::process(
            Xd3Config::new().no_compress(true),
            xdelta3::stream::ProcessMode::Encode,
             second, first, &mut patch
        ).unwrap();
        println!("{:?} done: {} bytes",patch_file, patch.len());
        let duration = start.elapsed();
        println!("Time elapsed for encode is: {:?}", duration);
        //save to file
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(patch_file)?;
        file.write_all(&patch)?;
    }

    //open or create patch_b.xdelta3.vcdiff
    let patch_file = Path::new(path).join("patch_b.xdelta3.vcdiff");
    if !patch_file.exists() {
        let (_,second) = open_and_hash_file(Path::new(path).join("318.iso").as_path())?;
        let (_,third) = open_and_hash_file(Path::new(path).join("319.iso").as_path())?;
        let mut patch = Vec::new();
        let start = Instant::now();
        xdelta3::stream::process(
            Xd3Config::new().no_compress(true),
            xdelta3::stream::ProcessMode::Encode,
             third, second, &mut patch
        ).unwrap();
        println!("{:?} done: {} bytes",patch_file, patch.len());
        let duration = start.elapsed();
        println!("Time elapsed for encode is: {:?}", duration);
        //save to file
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(patch_file)?;
        file.write_all(&patch)?;
    }

    //open or create patch_a.ovcd.vcdiff
    let patch_file = Path::new(path).join("patch_a.ovcd.vcdiff");
    if !patch_file.exists() {
        let (_,first) = open_and_hash_file(Path::new(path).join("317.iso").as_path())?;
        let (_,second) = open_and_hash_file(Path::new(path).join("318.iso").as_path())?;
        let first = first.into_inner();
        let second = second.into_inner();
        let start = Instant::now();
        let patch = encode(&first, &second, FORMAT_STANDARD, false);
        println!("{:?} done: {} bytes",patch_file, patch.len());
        let duration = start.elapsed();
        println!("Time elapsed for encode is: {:?}", duration);
        //save to file
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(patch_file)?;
        file.write_all(&patch)?;
    }

    //open or create patch_b.ovcd.vcdiff
    let patch_file = Path::new(path).join("patch_b.ovcd.vcdiff");
    if !patch_file.exists() {
        let (_,second) = open_and_hash_file(Path::new(path).join("318.iso").as_path())?;
        let (_,third) = open_and_hash_file(Path::new(path).join("319.iso").as_path())?;
        let second = second.into_inner();
        let third = third.into_inner();
        let start = Instant::now();
        let patch = encode(&second, &third, FORMAT_STANDARD, false);
        println!("{:?} done: {} bytes",patch_file, patch.len());
        let duration = start.elapsed();
        println!("Time elapsed for encode is: {:?}", duration);
        //save to file
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(patch_file)?;
        file.write_all(&patch)?;
    }


    Ok(())
}


