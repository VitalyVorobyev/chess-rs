//! Simple image pyramid utilities used by the multiscale corner finder.

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
}

impl Default for PyramidParams {
    fn default() -> Self {
        Self {
            num_levels: 3,
            min_size: 128,
        }
    }
}

fn downsample_2x(src: &GrayImage) -> GrayImage {
    let (w, h) = src.dimensions();
    let w2 = w / 2;
    let h2 = h / 2;

    let mut dst = GrayImage::new(w2, h2);

    for y in 0..h2 {
        for x in 0..w2 {
            // simple 2x2 box filter
            let sx = x * 2;
            let sy = y * 2;

            let p00 = src.get_pixel(sx, sy)[0] as u16;
            let p01 = src.get_pixel(sx + 1, sy)[0] as u16;
            let p10 = src.get_pixel(sx, sy + 1)[0] as u16;
            let p11 = src.get_pixel(sx + 1, sy + 1)[0] as u16;
            let avg = ((p00 + p01 + p10 + p11) / 4) as u8;

            dst.put_pixel(x, y, image::Luma([avg]));
        }
    }

    dst
}

/// Build a top-down image pyramid using fixed 2× downsampling.
///
/// The base image is always included as level 0. Each subsequent level is a
/// 2× downsampled copy (box filter). Construction stops when:
/// - either dimension would fall below `min_size`, or
/// - `num_levels` is reached.
pub fn build_pyramid(base: &GrayImage, pp: &PyramidParams) -> Pyramid {
    let mut levels = Vec::new();
    let mut img = base.clone();
    let mut scale = 1.0f32;

    for _level in 0..pp.num_levels {
        if img.width() < pp.min_size || img.height() < pp.min_size {
            break;
        }
        levels.push(PyramidLevel {
            img: img.clone(),
            scale,
        });

        if img.width() < 2 || img.height() < 2 {
            break;
        }

        img = downsample_2x(&img);
        scale *= 0.5;
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
    fn build_pyramid_stops_when_reaching_min_size() {
        let base = blank(20, 20);
        let params = PyramidParams {
            num_levels: 5,
            min_size: 6,
        };
        let pyramid = build_pyramid(&base, &params);
        // 20 -> 10 -> 5 (stop at min_size)
        let dims: Vec<_> = pyramid
            .levels
            .iter()
            .map(|lvl| (lvl.img.width(), lvl.img.height()))
            .collect();
        assert_eq!(dims, vec![(20, 20), (10, 10)]);
    }
}
