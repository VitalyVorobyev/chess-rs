use anyhow::Context;
use chess::{find_corners_coarse_to_fine_image_trace, ChessParams, CoarseToFineParams};
use image::{ImageBuffer, ImageReader, Luma};
use serde::Serialize;
use std::{fs::File, io::Write, path::PathBuf};

#[derive(Serialize)]
struct CornerOut {
    x: f32,
    y: f32,
    strength: f32,
}

#[derive(Serialize)]
struct CornerDump {
    image: String,
    width: u32,
    height: u32,
    pyramid_levels: u8,
    min_size: u32,
    roi_radius: u32,
    merge_radius: f32,
    build_ms: f64,
    coarse_ms: f64,
    refine_ms: f64,
    merge_ms: f64,
    corners: Vec<CornerOut>,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input: PathBuf = args
        .next()
        .expect("usage: dump_corners_multiscale <image> [--levels N] [--min-size PX] [--roi PX] [--merge R]")
        .into();

    let mut cf = CoarseToFineParams::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--levels" => {
                let v = args.next().context("expected an integer after --levels")?;
                cf.pyramid.num_levels = v.parse().context("could not parse levels as u8")?;
                if cf.pyramid.num_levels == 0 {
                    anyhow::bail!("levels must be >= 1");
                }
            }
            "--min-size" => {
                let v = args
                    .next()
                    .context("expected an integer after --min-size")?;
                cf.pyramid.min_size = v
                    .parse()
                    .context("could not parse min-size (use integer >= 1)")?;
                if cf.pyramid.min_size == 0 {
                    anyhow::bail!("min-size must be >= 1");
                }
            }
            "--roi" => {
                let v = args.next().context("expected an integer after --roi")?;
                cf.roi_radius = v
                    .parse()
                    .context("could not parse roi radius (use integer >= 1)")?;
                if cf.roi_radius == 0 {
                    anyhow::bail!("roi radius must be >= 1");
                }
            }
            "--merge" => {
                let v = args.next().context("expected a float after --merge")?;
                cf.merge_radius = v
                    .parse()
                    .context("could not parse merge radius (use float > 0)")?;
                if cf.merge_radius <= 0.0 {
                    anyhow::bail!("merge radius must be > 0");
                }
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let img = ImageReader::open(&input)?.decode()?.to_luma8();
    let params = ChessParams::default();

    let res = find_corners_coarse_to_fine_image_trace(&img, &params, &cf);

    println!(
        "Detected {} corners across {} pyramid levels",
        res.corners.len(),
        cf.pyramid.num_levels
    );
    println!("pyramid: {:5.2} ms", res.build_ms);
    println!(" coarse: {:5.2} ms", res.coarse_ms);
    println!(" refine: {:5.2} ms", res.refine_ms);
    println!("  merge: {:5.2} ms", res.merge_ms);

    let json_out = input.with_extension("multiscale.corners.json");
    let dump = CornerDump {
        image: input.to_string_lossy().into_owned(),
        width: img.width(),
        height: img.height(),
        pyramid_levels: cf.pyramid.num_levels,
        min_size: cf.pyramid.min_size,
        roi_radius: cf.roi_radius,
        merge_radius: cf.merge_radius,
        build_ms: res.build_ms,
        coarse_ms: res.coarse_ms,
        refine_ms: res.refine_ms,
        merge_ms: res.merge_ms,
        corners: res
            .corners
            .iter()
            .map(|c| CornerOut {
                x: c.xy[0],
                y: c.xy[1],
                strength: c.strength,
            })
            .collect(),
    };
    let mut json_file = File::create(&json_out)?;
    serde_json::to_writer_pretty(&mut json_file, &dump)?;
    json_file.write_all(b"\n")?;
    println!("Saved JSON dump to {}", json_out.display());

    // simple visualization: draw small 3x3 white squares around corners
    let mut vis: ImageBuffer<Luma<u8>, _> = img.clone();
    for c in &res.corners {
        let x = c.xy[0].round() as i32;
        let y = c.xy[1].round() as i32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let xx = x + dx;
                let yy = y + dy;
                if xx >= 0 && yy >= 0 && xx < vis.width() as i32 && yy < vis.height() as i32 {
                    vis.put_pixel(xx as u32, yy as u32, Luma([255u8]));
                }
            }
        }
    }

    let out = input.with_extension("multiscale.corners.png");
    vis.save(&out)?;
    println!("Saved visualization to {}", out.display());

    Ok(())
}
