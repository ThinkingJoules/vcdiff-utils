use std::fs;
use std::io::{copy, Cursor, Read, Seek, Write};
use std::path::Path;
use reqwest::blocking::Client;
use vcdiff_merge::applicator::{apply_patch, gather_summaries};
use vcdiff_merge::decoder::VCDDecoder;
use vcdiff_merge::merger::Merger;
use vcdiff_merge::reader::VCDReader;
use std::time::Instant;
use open_vcdiff_rs_bindings::{FORMAT_STANDARD,encode,decode};



fn download_file(url: &str, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let filename = url.split('/').last().unwrap_or("download.file");
    let filepath = path.join(filename);
    // Check if the file already exists
    if filepath.exists() {
        //println!("File {} already exists. Skipping download.", filename);
        return Ok(());
    }
    let response = Client::new().get(url).send()?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(filepath)?;
    let mut bytes = Cursor::new(response.bytes()?.to_vec());
    copy(&mut bytes, &mut file)?;

    println!("Downloaded {}", filename);
    Ok(())
}
fn open_and_hash_file(path: &Path) -> Result<(blake3::Hash,Cursor<Vec<u8>>), Box<dyn std::error::Error>> {
    let mut file = fs::File::open(path)?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;
    let hash = blake3::hash(&contents);
    Ok((hash,Cursor::new(contents)))
}
/*
Xdelta3 seems to not produce valid patches. Or my applicator isn't handling an edge case.
Google's open-vcdiff seems to work fine with my applicator, so hard to say who's code is wrong.
I'm created patches by cli using open-vcdiff and then applying them with my applicator.
*/
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let urls = vec![
        "https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/x86_64/alpine-standard-3.19.0-x86_64.iso",
        "https://dl-cdn.alpinelinux.org/alpine/v3.18/releases/x86_64/alpine-standard-3.18.0-x86_64.iso",
        "https://dl-cdn.alpinelinux.org/alpine/v3.17/releases/x86_64/alpine-standard-3.17.0-x86_64.iso"
    ];
    let path = "./downloads";

    // Create the directory if it doesn't exist
    fs::create_dir_all(path)?;
    // Download files sequentially
    for url in urls.iter() {
        download_file(url, Path::new(path))?;
    }

    // open each file and use xdelta3 to create a patch file between them
    let mut files = Vec::new();
    for url in urls {
        let filename = url.split('/').last().unwrap_or("download.file");
        let filepath = Path::new(path).join(filename);
        files.push(open_and_hash_file(&filepath)?);
    }
    // Create a patch file between the first two files
    let (_,mut one) = files.pop().unwrap();
    let (hash_2, mut two) = files.pop().unwrap();
    let (hash_3, mut three) = files.pop().unwrap();
    //let (ovcd_hash,_) = open_and_hash_file(&Path::new(path).join("3.18-openvcd.iso")).unwrap();
    //assert_eq!(hash_2, ovcd_hash);
    //see if we saved the patch file already
    let patch_file = Path::new(path).join("patch.ovcd");
    let patch = if patch_file.exists() {
        let mut patch = Vec::new();
        let mut file = fs::File::open(patch_file)?;
        file.read_to_end(&mut patch)?;
        patch
    } else {
        let start = Instant::now();
        let first = one.into_inner();
        let second = two.into_inner();
        let patch = encode(&first, &second, FORMAT_STANDARD, false);
        println!("First patch done: {} bytes", patch.len());
        let duration = start.elapsed();
        println!("Time elapsed for open-vcdiff encode is: {:?}", duration);
        //save to file
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(patch_file)?;
        file.write_all(&patch)?;
        two = Cursor::new(second);
        one = Cursor::new(first);
        patch
    };
    let mut patch = Cursor::new(patch);
    //let (_,mut patch) = open_and_hash_file(&Path::new(path).join("patch.openvcd")).unwrap();
    let sums = gather_summaries(&mut patch).unwrap();
    let mut expected_size_of_two = 0;
    for sum in sums {
        expected_size_of_two += sum.size_of_the_target_window;
        println!("{:?}", sum);
    }
    assert_eq!(expected_size_of_two, two.get_ref().len() as u64);
    //make sure xdelta decodes back to the original file
    let patch = patch.into_inner();
    let one = one.into_inner();
    let two = two.into_inner();
    // patch.rewind().unwrap();
    let start = Instant::now();
    let decoded = decode(&one, &patch);
    println!("First patch decoded");
    let duration = start.elapsed();
    println!("Time elapsed in open-vcdiff decode is: {:?}", duration);
    let decode_hash= blake3::hash(&decoded);
    if decode_hash != hash_2 {
        println!("open-vcdiff Hashes do not match");
        println!("Expected: {:?}", hash_2);
        println!("Actual: {:?}", decode_hash);
        println!("lengths: {} {}", two.len(), decoded.len());
    }else{
        println!("Xdelta properly decoded open-vcdiff patch :/");
    }
    //now we try the new applicator
    let mut decoded = Vec::new();
    // one.rewind().unwrap();
    let mut patch = Cursor::new(patch);
    let mut one = Cursor::new(one);
    let src = one.clone();
    let start = Instant::now();
    apply_patch(&mut patch, Some(src), &mut decoded).unwrap();
    println!("First patch applied new applicator");
    let duration = start.elapsed();
    println!("Time elapsed in new applicator is: {:?}", duration);
    let decode_hash= blake3::hash(&decoded);
    if decode_hash != hash_2 {
        println!("Applicator Hashes do not match");
        println!("Expected: {:?}", hash_2);
        println!("Actual: {:?}", decode_hash);
        println!("lengths: {} {}", two.len(), decoded.len());
    }else{
        println!("New applicator properly decoded open-vcdiff patch!");
    }
    // create patch file between the second and third files
    let mut two = Cursor::new(two);
    // let mut patch2 = Vec::new();
    // //see if we saved the patch file already
    // let patch_file = Path::new(path).join("patch2.xdelta");
    // if patch_file.exists() {
    //     let mut file = fs::File::open(patch_file)?;
    //     file.read_to_end(&mut patch2)?;
    // } else {
    //     let start = Instant::now();
    //     process(Xd3Config::new().no_compress(true),ProcessMode::Encode,&mut three, &mut two,&mut patch2).unwrap();
    //     println!("Second patch done: {} bytes", patch2.len());
    //     let duration = start.elapsed();
    //     println!("Time elapsed in xdelta3 encode patch 2 is: {:?}", duration);
    //     //save to file
    //     let mut file = fs::OpenOptions::new()
    //         .create(true)
    //         .write(true)
    //         .open(Path::new(path).join("patch2.xdelta"))?;
    //     file.write_all(&patch2)?;
    //     two.rewind().unwrap();
    // }
    // let mut patch2 = Cursor::new(patch2);
    let (_,mut patch2) = open_and_hash_file(&Path::new(path).join("patch2.openvcd")).unwrap();
    // let p2_summaries = gather_summaries(&mut patch2).unwrap();
    // let mut expected_size_of_three = 0;
    // println!("Number of windows: {}", p2_summaries.len());
    // for sum in p2_summaries {
    //     expected_size_of_three += sum.size_of_the_target_window;
    //     println!("{:?}", sum);
    // }
    // assert_eq!(expected_size_of_three, three.get_ref().len() as u64);
    patch2.rewind().unwrap();
    two.rewind().unwrap();
    let three = three.into_inner();
    // //make sure xdelta decodes back to the original file
    // let mut decoded = Vec::new();
    // let start = Instant::now();
    // process(Xd3Config::new(),ProcessMode::Decode,&mut patch2,&mut two ,&mut decoded).unwrap();
    // println!("Second patch decoded");
    // let duration = start.elapsed();
    // println!("Time elapsed in xdelta3 decode patch 2 is: {:?}", duration);
    // let decode_hash= blake3::hash(&decoded);
    // if decode_hash != hash_3 {
    //     println!("Xdelta Hashes do not match");
    //     println!("Expected: {:?}", hash_3);
    //     println!("Actual: {:?}", decode_hash);
    //     println!("lengths: {} {}", three.len(), decoded.len());
    // }else{
    //     println!("Xdelta properly decoded open-vcdiff patch 2 :/");
    // }
    //now we try the new applicator
    let mut decoded = Vec::new();
    two.rewind().unwrap();
    patch2.rewind().unwrap();
    let start = Instant::now();
    apply_patch(&mut patch2, Some(two), &mut decoded).unwrap();
    println!("Second patch applied new applicator");
    let duration = start.elapsed();
    println!("Time elapsed in new applicator patch 2 is: {:?}", duration);
    let decode_hash= blake3::hash(&decoded);
    if decode_hash != hash_3 {
        println!("Applicator Hashes do not match");
        println!("Expected: {:?}", hash_3);
        println!("Actual: {:?}", decode_hash);
        println!("lengths: {} {}", three.len(), decoded.len());
    }else{
        println!("New applicator properly decoded open-vcdiff patch 2!");
    }
    //merge the two patch files
    patch.rewind().unwrap();
    patch2.rewind().unwrap();
    let start = Instant::now();
    let merger = Merger::new(patch2).unwrap().unwrap();
    let merger = merger.merge(patch).unwrap().unwrap();
    let summary_patch = merger.finish().write(Vec::new(), None).unwrap();
    //apply the merged patch file to the first file
    println!("Summary patch done: {} bytes", summary_patch.len());
    let duration = start.elapsed();
    println!("Time elapsed in merging patches is: {:?}", duration);
    //write summary patch to disk
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(Path::new(path).join("summary_patch.vcdiff"))?;
    file.write_all(&summary_patch)?;
    let mut decoded = Vec::new();
    let mut summary_patch = Cursor::new(summary_patch);
    one.rewind().unwrap();
    let sum_summaries = gather_summaries(&mut summary_patch).unwrap();
    let mut expected_size_of_three = 0;
    println!("Number of windows: {}", sum_summaries.len());
    for sum in sum_summaries {
        expected_size_of_three += sum.size_of_the_target_window;
        println!("{:?}", sum);
    }
    assert_eq!(expected_size_of_three, three.len() as u64);
    summary_patch.rewind().unwrap();
    let start = Instant::now();
    //process(Xd3Config::new(),ProcessMode::Decode,&mut Cursor::new(summary_patch.clone()),&mut one ,&mut decoded).unwrap();
    apply_patch(&mut summary_patch, Some(one.clone()), &mut decoded).unwrap();
    let duration = start.elapsed();
    println!("Time elapsed in applying merged patch is: {:?}", duration);
    let decode_hash= blake3::hash(&decoded);

    assert_eq!(decode_hash, hash_3);

    Ok(())
}
// fn main() {
//     println!("Hello, world!");
// }
