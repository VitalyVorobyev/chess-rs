// tools/gen_golden.rs
use std::{fs::File, io::Write, path::Path};
use image::ImageReader;
use chess_core::{ChessParams};
use chess_core::response::chess_response_u8;

fn write_golden(path_out: &Path, w: usize, h: usize, data: &[f32]) -> std::io::Result<()> {
    let mut f = File::create(path_out)?;
    f.write_all(&(w as u32).to_le_bytes())?;
    f.write_all(&(h as u32).to_le_bytes())?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let imgs = std::fs::read_dir("testdata/images")?;
    std::fs::create_dir_all("testdata/golden")?;

    let params = ChessParams::default();

    for e in imgs {
        let p = e?.path();
        if p.extension().and_then(|s| s.to_str()) != Some("png") { continue; }
        let img = ImageReader::open(&p)?.decode()?.to_luma8();

        let w = img.width() as usize;
        let h = img.height() as usize;
        let resp = chess_response_u8(img.as_raw(), w, h, &params);

        let name = p.file_stem().unwrap().to_string_lossy();
        let out = Path::new("testdata/golden").join(format!("{name}.bin"));
        write_golden(&out, resp.w, resp.h, &resp.data)?;
        println!("golden: {:?} -> {:?}", p, out);
    }
    Ok(())
}
