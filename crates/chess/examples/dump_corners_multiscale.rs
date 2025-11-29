use anyhow::{anyhow, Context};
use chess::{find_corners_coarse_to_fine_image_trace, ChessParams, CoarseToFineParams};
use image::{ImageBuffer, ImageReader, Luma};
use log::{info, LevelFilter};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::{fs::File, io::Write, path::Path, path::PathBuf};

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

#[derive(Debug, Deserialize)]
struct DumpConfig {
    /// Path to the input image.
    image: PathBuf,
    /// Pyramid levels (optional override of default).
    pyramid_levels: Option<u8>,
    /// Smallest pyramid dimension (optional override of default).
    min_size: Option<u32>,
    /// ROI radius in pixels (optional override of default).
    roi_radius: Option<u32>,
    /// Merge radius in pixels (optional override of default).
    merge_radius: Option<f32>,
    /// Minimum log level to emit (trace, debug, info, warn, error).
    log_level: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let cfg_path: PathBuf = args
        .next()
        .expect("usage: dump_corners_multiscale <config.json> [--levels N] [--min-size PX] [--roi PX] [--merge R]")
        .into();

    let cfg = load_config(&cfg_path)?;
    let log_level = parse_log_level(cfg.log_level.as_deref())?;
    chess::logger::init_with_level(log_level)
        .map_err(|e| anyhow!("failed to initialize logger: {e}"))?;
    info!("Loaded config from {}", cfg_path.display());
    info!("Log level set to {}", log_level);

    let input = cfg.image.clone();

    let mut cf = CoarseToFineParams::default();
    // Apply config defaults.
    if let Some(v) = cfg.pyramid_levels {
        if v == 0 {
            anyhow::bail!("levels must be >= 1");
        }
        cf.pyramid.num_levels = v;
    }
    if let Some(v) = cfg.min_size {
        if v == 0 {
            anyhow::bail!("min-size must be >= 1");
        }
        cf.pyramid.min_size = v;
    }
    if let Some(v) = cfg.roi_radius {
        if v == 0 {
            anyhow::bail!("roi radius must be >= 1");
        }
        cf.roi_radius = v;
    }
    if let Some(v) = cfg.merge_radius {
        if v <= 0.0 {
            anyhow::bail!("merge radius must be > 0");
        }
        cf.merge_radius = v;
    }

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

    info!(
        "Using params: levels={}, min_size={}, roi_radius={}, merge_radius={:.2}",
        cf.pyramid.num_levels, cf.pyramid.min_size, cf.roi_radius, cf.merge_radius
    );
    info!("Opening image {}", input.display());

    let img = ImageReader::open(&input)?.decode()?.to_luma8();
    info!("Image size: {} x {}", img.width(), img.height());
    let params = ChessParams::default();

    let res = find_corners_coarse_to_fine_image_trace(&img, &params, &cf);

    info!(
        "Detected {} corners, coarsest size: {} x {}",
        res.corners.len(),
        res.coarse_cols,
        res.coarse_rows
    );
    info!("pyramid: {:5.2} ms", res.build_ms);
    info!(" coarse: {:5.2} ms", res.coarse_ms);
    info!(" refine: {:5.2} ms", res.refine_ms);
    info!("  merge: {:5.2} ms", res.merge_ms);

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
    info!("Saved JSON dump to {}", json_out.display());

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
    info!("Saved visualization to {}", out.display());

    Ok(())
}

fn parse_log_level(raw: Option<&str>) -> anyhow::Result<LevelFilter> {
    let raw = raw.unwrap_or("info");
    LevelFilter::from_str(raw).map_err(|_| anyhow!("invalid log level: {raw}"))
}

fn load_config(path: &Path) -> anyhow::Result<DumpConfig> {
    let file = File::open(path).with_context(|| format!("opening config {}", path.display()))?;
    let cfg: DumpConfig = serde_json::from_reader(file)
        .with_context(|| format!("parsing config {}", path.display()))?;
    Ok(cfg)
}
