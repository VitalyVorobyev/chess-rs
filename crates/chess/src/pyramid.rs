//! Simple image pyramid utilities used by the multiscale corner finder.
//!
//! The API is allocation-friendly: construct a [`PyramidBuffers`] once, then
//! reuse it to build pyramids for successive frames without re-allocating
//! intermediate levels.

use image::GrayImage;

/// Reusable backing storage for pyramid construction.
///
/// Call [`PyramidBuffers::prepare_for_image`] once per frame to ensure the
/// internal buffers match the planned pyramid shape, then pass the same
/// instance to [`build_pyramid`] to fill those buffers.
pub struct PyramidBuffers {
    levels: Vec<GrayImage>,
}

impl PyramidBuffers {
    /// Create an empty buffer set.
    pub fn new() -> Self {
        Self { levels: Vec::new() }
    }

    /// Create a buffer set with capacity reserved for `num_levels`.
    pub fn with_capacity(num_levels: u8) -> Self {
        Self {
            levels: Vec::with_capacity(num_levels.saturating_sub(1) as usize),
        }
    }

    /// Ensure buffers exist for all downsampled levels that will be produced
    /// for `base` and `params`. Returns the number of downsampled levels
    /// (excluding the base) that were sized.
    pub fn prepare_for_image(&mut self, base: &GrayImage, params: &PyramidParams) -> usize {
        if params.num_levels == 0
            || base.width() < params.min_size
            || base.height() < params.min_size
        {
            return 0;
        }

        let mut w = base.width();
        let mut h = base.height();
        let mut prepared = 0usize;

        for _ in 1..params.num_levels {
            if w < 2 || h < 2 {
                break;
            }

            w /= 2;
            h /= 2;

            if w < params.min_size || h < params.min_size {
                break;
            }

            self.ensure_level_shape(prepared, w, h);
            prepared += 1;
        }

        prepared
    }

    fn ensure_level_shape(&mut self, idx: usize, w: u32, h: u32) {
        if idx >= self.levels.len() {
            self.levels.resize_with(idx + 1, || GrayImage::new(w, h));
        }

        let level = &mut self.levels[idx];
        if level.width() != w || level.height() != h {
            *level = GrayImage::new(w, h);
        }
    }
}

/// A single pyramid level. The `scale` is relative to the base image.
pub struct PyramidLevel<'a> {
    pub img: &'a GrayImage,
    pub scale: f32, // relative to base (e.g. 1.0, 0.5, 0.25, ...)
}

/// A top-down pyramid where `levels[0]` is the base (full resolution).
pub struct Pyramid<'a> {
    pub levels: Vec<PyramidLevel<'a>>, // levels[0] is base
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

/// Build a top-down image pyramid using fixed 2× downsampling.
///
/// The base image is always included as level 0. Each subsequent level is a
/// 2× downsampled copy (box filter) written into `buffers`. Construction stops
/// when:
/// - either dimension would fall below `min_size`, or
/// - `num_levels` is reached.
pub fn build_pyramid<'a>(
    base: &'a GrayImage,
    params: &PyramidParams,
    buffers: &'a mut PyramidBuffers,
) -> Pyramid<'a> {
    if params.num_levels == 0 || base.width() < params.min_size || base.height() < params.min_size {
        return Pyramid { levels: Vec::new() };
    }

    #[derive(Clone, Copy)]
    enum LevelSource {
        Base,
        Buffer(usize),
    }

    let mut sources: Vec<(LevelSource, f32)> = Vec::with_capacity(params.num_levels as usize);
    sources.push((LevelSource::Base, 1.0));

    let mut current_src = LevelSource::Base;
    let mut current_w = base.width();
    let mut current_h = base.height();
    let mut scale = 1.0f32;

    for level_idx in 1..params.num_levels {
        let w2 = current_w / 2;
        let h2 = current_h / 2;

        if w2 == 0 || h2 == 0 || w2 < params.min_size || h2 < params.min_size {
            break;
        }

        let buf_idx = (level_idx - 1) as usize;
        buffers.ensure_level_shape(buf_idx, w2, h2);

        let (src_img, dst): (&GrayImage, &mut GrayImage) = match current_src {
            LevelSource::Base => (base, &mut buffers.levels[buf_idx]),
            LevelSource::Buffer(src_idx) => {
                debug_assert!(src_idx < buf_idx);
                let (head, tail) = buffers.levels.split_at_mut(buf_idx);
                (&head[src_idx], &mut tail[0])
            }
        };

        downsample_2x(src_img, dst);

        scale *= 0.5;
        current_src = LevelSource::Buffer(buf_idx);
        current_w = w2;
        current_h = h2;
        sources.push((current_src, scale));
    }

    let mut levels = Vec::with_capacity(sources.len());
    for (source, lvl_scale) in sources {
        let img = match source {
            LevelSource::Base => base,
            LevelSource::Buffer(idx) => &buffers.levels[idx],
        };
        levels.push(PyramidLevel {
            img,
            scale: lvl_scale,
        });
    }

    Pyramid { levels }
}

/// Fast 2× downsample with a 2×2 box filter into a pre-allocated destination.
fn downsample_2x(src: &GrayImage, dst: &mut GrayImage) {
    debug_assert_eq!(src.width() / 2, dst.width());
    debug_assert_eq!(src.height() / 2, dst.height());

    let src_w = src.width() as usize;
    let dst_w = dst.width() as usize;
    let dst_h = dst.height() as usize;

    let src_pixels = src.as_raw();
    let dst_pixels = dst.as_mut();

    for y in 0..dst_h {
        let row0 = (y * 2) * src_w;
        let row1 = row0 + src_w;

        for x in 0..dst_w {
            let sx = x * 2;
            let p00 = src_pixels[row0 + sx] as u16;
            let p01 = src_pixels[row0 + sx + 1] as u16;
            let p10 = src_pixels[row1 + sx] as u16;
            let p11 = src_pixels[row1 + sx + 1] as u16;
            dst_pixels[y * dst_w + x] = ((p00 + p01 + p10 + p11) >> 2) as u8;
        }
    }
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
        let mut buffers = PyramidBuffers::with_capacity(params.num_levels);
        buffers.prepare_for_image(&base, &params);
        let pyramid = build_pyramid(&base, &params, &mut buffers);
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
        let mut buffers = PyramidBuffers::new();
        buffers.prepare_for_image(&base, &params);
        let pyramid = build_pyramid(&base, &params, &mut buffers);
        // 20 -> 10 -> 5 (stop at min_size)
        let dims: Vec<_> = pyramid
            .levels
            .iter()
            .map(|lvl| (lvl.img.width(), lvl.img.height()))
            .collect();
        assert_eq!(dims, vec![(20, 20), (10, 10)]);
    }

    #[test]
    fn prepare_reuses_buffers_when_shape_matches() {
        let base = blank(512, 512);
        let params = PyramidParams::default();
        let mut buffers = PyramidBuffers::new();
        let first = buffers.prepare_for_image(&base, &params);
        let ptrs: Vec<*const u8> = buffers.levels.iter().map(|l| l.as_ptr()).collect();
        let second = buffers.prepare_for_image(&base, &params);
        let ptrs_after: Vec<*const u8> = buffers.levels.iter().map(|l| l.as_ptr()).collect();
        assert_eq!(first, 2); // 512 -> 256 -> 128 with default params
        assert_eq!(second, first);
        assert_eq!(
            ptrs, ptrs_after,
            "buffers should be reused when shapes match"
        );
    }
}
