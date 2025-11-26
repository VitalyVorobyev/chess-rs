//! Simple image pyramid utilities used by the multiscale corner finder.

use image::imageops::{resize, FilterType};
use image::GrayImage;

/// A single pyramid level. The `scale` is relative to the base image.
pub struct PyramidLevel {
    pub img: GrayImage,
    pub scale: f32, // relative to base (e.g. 1.0, 0.5, 0.25, ...)
}

/// A top-down pyramid where `levels[0]` is the base (full resolution).
pub struct Pyramid {
    pub levels: Vec<PyramidLevel>, // levels[0] is base
}

/// Parameters controlling pyramid generation.
pub struct PyramidParams {
    /// Maximum number of levels (including the base).
    pub num_levels: u8,
    /// Stop building when either dimension falls below this value.
    pub min_size: u32,
    /// Downscale factor between subsequent levels (must be in `(0, 1)`).
    pub scale_factor: f32,
}

impl Default for PyramidParams {
    fn default() -> Self {
        Self {
            num_levels: 3,
            min_size: 128,
            scale_factor: 0.5,
        }
    }
}

/// Build a top-down image pyramid using the provided parameters.
///
/// The base image is always included as level 0. Each subsequent level is
/// resized with a Triangle filter to approximately `scale_factor` of the
/// previous level. Construction stops when:
/// - either dimension would fall below `min_size`, or
/// - `num_levels` is reached, or
/// - a downscale would not reduce the dimensions (guards against `scale_factor`
///   values that are too close to 1.0).
pub fn build_pyramid(base: &GrayImage, pp: &PyramidParams) -> Pyramid {
    let mut levels = Vec::new();
    let mut img = base.clone();
    let mut scale = 1.0f32;

    for level in 0..pp.num_levels {
        if img.width() < pp.min_size || img.height() < pp.min_size {
            break;
        }
        levels.push(PyramidLevel {
            img: img.clone(),
            scale,
        });

        if level + 1 < pp.num_levels {
            let next_w = ((img.width() as f32) * pp.scale_factor).floor().max(1.0) as u32;
            let next_h = ((img.height() as f32) * pp.scale_factor).floor().max(1.0) as u32;

            // Prevent infinite loops if the scale factor is too close to 1.0.
            if next_w == img.width() || next_h == img.height() {
                break;
            }

            img = resize(&img, next_w, next_h, FilterType::Triangle);
            scale *= pp.scale_factor;
        }
    }

    Pyramid { levels }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn blank(w: u32, h: u32) -> GrayImage {
        GrayImage::from_pixel(w, h, Luma([0u8]))
    }

    #[test]
    fn build_pyramid_respects_min_size_and_levels() {
        let base = blank(64, 64);
        let params = PyramidParams {
            num_levels: 4,
            min_size: 8,
            scale_factor: 0.5,
        };
        let pyramid = build_pyramid(&base, &params);
        let dims: Vec<_> = pyramid
            .levels
            .iter()
            .map(|lvl| (lvl.img.width(), lvl.img.height(), lvl.scale))
            .collect();
        assert_eq!(
            dims,
            vec![(64, 64, 1.0), (32, 32, 0.5), (16, 16, 0.25), (8, 8, 0.125)]
        );
    }

    #[test]
    fn build_pyramid_stops_when_scale_factor_too_close_to_one() {
        let base = blank(32, 32);
        let params = PyramidParams {
            num_levels: 5,
            min_size: 4,
            scale_factor: 0.95,
        };
        let pyramid = build_pyramid(&base, &params);
        // 32 -> 30 -> 28 -> 26 -> 24 ... ensure it terminates and keeps shrinking
        assert!(pyramid.levels.len() >= 2);
        for pair in pyramid.levels.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            assert!(b.img.width() < a.img.width());
            assert!(b.img.height() < a.img.height());
        }
    }
}
