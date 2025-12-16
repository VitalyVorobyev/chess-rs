/// Minimal grayscale view for refinement without taking a dependency on `image`.
#[derive(Copy, Clone, Debug)]
pub struct ImageView<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    /// Origin of the view in the coordinate system of the response map / base image.
    pub origin: [i32; 2],
}

impl<'a> ImageView<'a> {
    pub fn from_u8_slice(width: usize, height: usize, data: &'a [u8]) -> Option<Self> {
        if width.checked_mul(height)? != data.len() {
            return None;
        }
        Some(Self {
            data,
            width,
            height,
            origin: [0, 0],
        })
    }

    pub fn with_origin(
        width: usize,
        height: usize,
        data: &'a [u8],
        origin: [i32; 2],
    ) -> Option<Self> {
        Self::from_u8_slice(width, height, data).map(|mut view| {
            view.origin = origin;
            view
        })
    }

    #[inline]
    pub fn supports_patch(&self, cx: i32, cy: i32, radius: i32) -> bool {
        if self.width == 0 || self.height == 0 {
            return false;
        }

        let gx = cx + self.origin[0];
        let gy = cy + self.origin[1];
        let min_x = 0;
        let min_y = 0;
        let max_x = self.width as i32 - 1;
        let max_y = self.height as i32 - 1;
        gx - radius >= min_x && gy - radius >= min_y && gx + radius <= max_x && gy + radius <= max_y
    }

    #[inline]
    pub fn sample(&self, gx: i32, gy: i32) -> f32 {
        if self.width == 0 || self.height == 0 {
            return 0.0;
        }
        let gx = gx + self.origin[0];
        let gy = gy + self.origin[1];
        let lx = gx.clamp(0, self.width.saturating_sub(1) as i32) as usize;
        let ly = gy.clamp(0, self.height.saturating_sub(1) as i32) as usize;
        self.data[ly * self.width + lx] as f32
    }
}
