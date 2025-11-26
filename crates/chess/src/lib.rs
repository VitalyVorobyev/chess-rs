//! Ergonomic wrappers over `chess-core` that accept `image::GrayImage` inputs.

mod pyramid;

pub use chess_core::*;

pub use crate::pyramid::{build_pyramid, Pyramid, PyramidLevel, PyramidParams};

use image::GrayImage;

/// Compute ChESS response map for an `image::GrayImage`.
pub fn chess_response_image(img: &GrayImage, params: &ChessParams) -> ResponseMap {
    chess_core::response::chess_response_u8(
        img.as_raw(),
        img.width() as usize,
        img.height() as usize,
        params,
    )
}

/// Detect subpixel corners from an `image::GrayImage`.
pub fn find_corners_image(
    img: &GrayImage,
    params: &ChessParams,
) -> Vec<chess_core::detect::Corner> {
    chess_core::detect::find_corners_u8(
        img.as_raw(),
        img.width() as usize,
        img.height() as usize,
        params,
    )
}

/// Detect subpixel corners from an `image::GrayImage` and return timing stats.
pub fn find_corners_image_trace(
    img: &GrayImage,
    params: &ChessParams,
) -> chess_core::detect::ChessResult {
    chess_core::detect::find_corners_u8_with_trace(
        img.as_raw(),
        img.width() as usize,
        img.height() as usize,
        params,
    )
}

/// Corner detected in a multiscale run, reported in base-level coordinates.
pub struct MultiscaleCorner {
    /// Position in the original (level 0) image coordinates.
    pub xy: [f32; 2],
    pub strength: f32,
}

/// Detect corners across an image pyramid and merge nearby detections.
///
/// Coordinates are rescaled back to the base image so consumers can treat the
/// output as a single set. Corners closer than `2` pixels are merged with
/// simple strength-based suppression.
///
/// # Example
/// ```rust
/// use chess::{find_corners_multiscale_image, ChessParams, PyramidParams};
/// use image::GrayImage;
///
/// let img = GrayImage::from_pixel(64, 64, image::Luma([0u8]));
/// let params = ChessParams::default();
/// let pyramid = PyramidParams::default();
///
/// let corners = find_corners_multiscale_image(&img, &params, &pyramid);
/// assert!(corners.is_empty());
/// ```
pub fn find_corners_multiscale_image(
    img: &image::GrayImage,
    p: &ChessParams,
    py: &PyramidParams,
) -> Vec<MultiscaleCorner> {
    let pyramid = build_pyramid(img, py);

    let mut all = Vec::new();

    for lvl in pyramid.levels.iter() {
        let w = lvl.img.width() as usize;
        let h = lvl.img.height() as usize;

        let cores = chess_core::detect::find_corners_u8(lvl.img.as_raw(), w, h, p);

        let inv_scale = 1.0 / lvl.scale;

        for c in cores {
            all.push(MultiscaleCorner {
                xy: [c.xy[0] * inv_scale, c.xy[1] * inv_scale],
                strength: c.strength,
            });
        }
    }

    // TODO: merge duplicates (see next point)
    merge_corners_simple(&mut all, 2.0)
}

fn merge_corners_simple(corners: &mut Vec<MultiscaleCorner>, radius: f32) -> Vec<MultiscaleCorner> {
    let r2 = radius * radius;
    let mut out: Vec<MultiscaleCorner> = Vec::new();

    // naive O(N^2) for now; N is small for single chessboard
    'outer: for c in corners.drain(..) {
        for o in &mut out {
            let dx = c.xy[0] - o.xy[0];
            let dy = c.xy[1] - o.xy[1];
            if dx * dx + dy * dy <= r2 {
                // keep the stronger
                if c.strength > o.strength {
                    *o = c;
                }
                continue 'outer;
            }
        }
        out.push(c);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn merge_corners_prefers_stronger_entries() {
        let mut corners = vec![
            MultiscaleCorner {
                xy: [10.0, 10.0],
                strength: 1.0,
            },
            MultiscaleCorner {
                xy: [11.0, 11.0],
                strength: 5.0,
            },
            MultiscaleCorner {
                xy: [20.0, 20.0],
                strength: 3.0,
            },
        ];
        let merged = merge_corners_simple(&mut corners, 2.5);
        assert_eq!(merged.len(), 2);
        assert!(merged
            .iter()
            .any(|c| c.xy == [11.0, 11.0] && (c.strength - 5.0).abs() < 1e-6));
        assert!(merged.iter().any(|c| c.xy == [20.0, 20.0]));
    }

    #[test]
    fn multiscale_path_runs_on_blank_image() {
        let img = GrayImage::from_pixel(32, 32, Luma([0u8]));
        let params = ChessParams::default();
        let pyramid = PyramidParams::default();
        let res = find_corners_multiscale_image(&img, &params, &pyramid);
        assert!(res.is_empty());
    }
}
