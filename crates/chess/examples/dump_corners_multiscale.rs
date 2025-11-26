use anyhow::Context;
use chess::{find_corners_multiscale_image, ChessParams, PyramidParams};
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
    scale_factor: f32,
    min_size: u32,
    corners: Vec<CornerOut>,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input: PathBuf = args
        .next()
        .expect(
            "usage: dump_corners_multiscale <image> [--levels N] [--scale FACTOR] [--min-size PX]",
        )
        .into();

    let mut pyr = PyramidParams::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--levels" => {
                let v = args.next().context("expected an integer after --levels")?;
                pyr.num_levels = v.parse().context("could not parse levels as u8")?;
                if pyr.num_levels == 0 {
                    anyhow::bail!("levels must be >= 1");
                }
            }
            "--scale" => {
                let v = args.next().context("expected a number after --scale")?;
                pyr.scale_factor = v
                    .parse()
                    .context("could not parse scale factor (use a float between 0 and 1)")?;
                if !(0.0..1.0).contains(&pyr.scale_factor) {
                    anyhow::bail!("scale factor must be in (0, 1)");
                }
            }
            "--min-size" => {
                let v = args
                    .next()
                    .context("expected an integer after --min-size")?;
                pyr.min_size = v
                    .parse()
                    .context("could not parse min-size (use integer >= 1)")?;
                if pyr.min_size == 0 {
                    anyhow::bail!("min-size must be >= 1");
                }
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let img = ImageReader::open(&input)?.decode()?.to_luma8();
    let params = ChessParams::default();

    let corners = find_corners_multiscale_image(&img, &params, &pyr);
    println!(
        "Detected {} corners across {} pyramid levels",
        corners.len(),
        pyr.num_levels
    );

    let json_out = input.with_extension("multiscale.corners.json");
    let dump = CornerDump {
        image: input.to_string_lossy().into_owned(),
        width: img.width(),
        height: img.height(),
        pyramid_levels: pyr.num_levels,
        scale_factor: pyr.scale_factor,
        min_size: pyr.min_size,
        corners: corners
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
    for c in &corners {
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
