use std::{fs::File, io::Read, path::Path};
use image::ImageReader;
use chess_core::{ChessParams};
use chess_core::response::chess_response_u8;

fn read_golden(path: &Path) -> (usize, usize, Vec<f32>) {
    let mut buf = Vec::new();
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();

    let w = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;

    let mut data = Vec::with_capacity(w*h);
    let mut i = 8;
    while i < buf.len() {
        let v = f32::from_le_bytes(buf[i..i+4].try_into().unwrap());
        data.push(v);
        i += 4;
    }
    assert_eq!(data.len(), w*h);
    (w, h, data)
}

#[test]
fn response_matches_golden_set() {
    let params = ChessParams::default();
    let imgs = std::fs::read_dir("testdata/images").unwrap();

    for e in imgs {
        let p = e.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) != Some("png") { continue; }

        let img = ImageReader::open(&p).unwrap().decode().unwrap().to_luma8();
        let w = img.width() as usize;
        let h = img.height() as usize;

        let resp = chess_response_u8(img.as_raw(), w, h, &params);

        let name = p.file_stem().unwrap().to_string_lossy();
        let gold_path = Path::new("testdata/golden").join(format!("{name}.bin"));
        let (gw, gh, gdata) = read_golden(&gold_path);

        assert_eq!((gw, gh), (resp.w, resp.h));

        // epsilon: tight enough to catch logic changes
        let eps = 1e-4_f32;
        for (i, (a, b)) in resp.data.iter().zip(gdata.iter()).enumerate() {
            let d = (a - b).abs();
            assert!(
                d <= eps,
                "diff too high at idx {i} ({name}): {a} vs {b} (d={d})"
            );
        }
    }
}
