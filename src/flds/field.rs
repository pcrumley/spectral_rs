use crate::{Float, Sim};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

pub struct Pos {
    pub row: usize,
    pub col: usize,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FieldDim {
    pub size_x: usize,
    pub size_y: usize,
}

pub struct Field {
    pub spatial: Vec<Float>,
    pub spectral: Vec<Complex<Float>>,
    pub with_ghost_dim: FieldDim,
    pub no_ghost_dim: FieldDim,
    pub name: String,
}

impl FieldDim {
    pub fn get_index(&self, pos: Pos) -> usize {
        // Convenience method to get a position in the array.
        // Slightly complicated because
        // Using a 1d vec to represent 2D array for speed.
        // Here is the layout if it were a 2d array,
        // with the 1D vec position in []
        // +----------+----------+----------+
        // |   [0]    |   [1]    |   [2]    |
        // |  row: 0  |  row: 0  |  row: 0  |
        // |  col: 0  |  col: 1  |  col: 2  |
        // |          |          |          |
        // +----------+----------+----------+
        // |   [3]    |   [4]    |   [5]    |
        // |  row: 1  |  row: 1  |  row: 1  |
        // |  col: 0  |  col: 1  |  col: 2  |
        // |          |          |          |
        // +----------+----------+----------+
        // |   [6]    |   [7]    |   [8]    |
        // |  row: 2  |  row: 2  |  row: 2  |
        // |  col: 1  |  col: 1  |  col: 2  |
        // |          |          |          |
        // +----------+----------+----------+

        if !cfg!(feature = "unchecked") {
            assert!(pos.col < self.size_x);
            assert!(pos.row < self.size_y);
        }

        pos.row * self.size_x + pos.col
    }
}
impl Field {
    pub fn default(sim: &Sim) -> Field {
        Field {
            spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            with_ghost_dim: FieldDim {
                size_x: sim.size_x + 2,
                size_y: sim.size_y + 2,
            },
            no_ghost_dim: FieldDim {
                size_x: sim.size_x,
                size_y: sim.size_y,
            },
            name: "".into(),
        }
    }
    pub fn new(sim: &Sim, name: String) -> Field {
        Field {
            name,
            ..Field::default(sim)
        }
    }

    #[inline(always)]
    pub fn copy_to_spectral(&mut self) -> () {
        // assert stuff about ghost zones etc.
        let spatial = &self.spatial;
        let spectral = &mut self.spectral;
        if !cfg!(feature = "unchecked") {
            assert_eq!(
                spatial.len(),
                (self.no_ghost_dim.size_x + 2) * (self.no_ghost_dim.size_y + 2)
            );
            assert_eq!(
                spectral.len(),
                (self.with_ghost_dim.size_x - 2) * (self.with_ghost_dim.size_y - 2)
            );
        }
        let size_x = self.no_ghost_dim.size_x;
        let size_y = self.no_ghost_dim.size_y;
        for iy in 0..size_y {
            let ij = iy * (size_x);
            let ij_ghosts = (iy + 1) * (size_x + 2);
            for ix in 0..size_x {
                unsafe {
                    // completely safe due to assert above... unless unchecked is run.
                    // still should be fine but not guaranteed at runtime.
                    spectral.get_unchecked_mut(ij + ix).re =
                        *spatial.get_unchecked(ij_ghosts + ix + 1);
                    spectral.get_unchecked_mut(ij + ix).im = 0.0;
                }
            }
        }
    }

    #[inline(always)]
    pub fn copy_to_spatial(&mut self, sim: &Sim) -> () {
        // assert stuff about ghost zones etc.
        let spatial = &mut self.spatial;
        let spectral = &self.spectral;
        if !cfg!(feature = "unchecked") {
            assert!(spatial.len() == (sim.size_x + 2) * (sim.size_y + 2));
            assert!(spectral.len() == sim.size_x * sim.size_y);
        }
        let size_x = sim.size_x;
        let size_y = sim.size_y;
        for iy in 0..size_y {
            let ij = iy * (size_x);
            let ij_ghosts = (iy + 1) * (size_x + 2);
            for ix in 0..size_x {
                unsafe {
                    // completely safe due to assert above... unless unchecked is run.
                    // still should be fine but not guaranteed at runtime.
                    *spatial.get_unchecked_mut(ij_ghosts + ix + 1) =
                        spectral.get_unchecked(ij + ix).re;
                }
            }
        }
        self.update_ghosts();
    }

    #[inline(always)]
    pub fn update_ghosts(&mut self) -> () {
        let size_x = self.no_ghost_dim.size_x;
        let size_y = self.no_ghost_dim.size_y;
        if !cfg!(feature = "unchecked") {
            assert!(self.spatial.len() == (size_x + 2) * (size_y + 2));
            assert!(
                self.spectral.len()
                    == (self.with_ghost_dim.size_x - 2) * (self.with_ghost_dim.size_y - 2)
            );
        }

        let fld = &mut self.spatial;
        // Copy bottom row into top ghost row
        let ghost_start = self.with_ghost_dim.get_index(Pos { row: 0, col: 1 });
        let ghost_range = ghost_start..ghost_start + size_x;
        let real_start = self.with_ghost_dim.get_index(Pos {
            row: size_y,
            col: 1,
        });
        let real_range = real_start..real_start + size_x;
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
            }
        }
        // Copy top row into bottom ghost row
        let ghost_start = self.with_ghost_dim.get_index(Pos {
            row: size_y + 1,
            col: 1,
        });
        let ghost_range = ghost_start..ghost_start + size_x;
        let real_start = self.with_ghost_dim.get_index(Pos { row: 1, col: 1 });
        let real_range = real_start..real_start + size_x;
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
            }
        }
        // copy into left ghost columns from right real column
        let ghost_start = self.with_ghost_dim.get_index(Pos { row: 1, col: 0 });
        let ghost_end = self.with_ghost_dim.get_index(Pos {
            row: 1 + size_y,
            col: 0,
        });
        let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
        let real_start = self.with_ghost_dim.get_index(Pos {
            row: 1,
            col: size_x,
        });
        let real_end = self.with_ghost_dim.get_index(Pos {
            row: 1 + size_y,
            col: size_x,
        });
        let real_range = (real_start..real_end).step_by(2 + size_x);
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
            }
        }

        // copy into right ghost columns from left real column
        let ghost_start = self.with_ghost_dim.get_index(Pos {
            row: 1,
            col: size_x + 1,
        });
        let ghost_end = self.with_ghost_dim.get_index(Pos {
            row: 1 + size_y,
            col: size_x + 1,
        });
        let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
        let real_start = self.with_ghost_dim.get_index(Pos { row: 1, col: 1 });

        let real_end = self.with_ghost_dim.get_index(Pos {
            row: 1 + size_y,
            col: 1,
        });
        let real_range = (real_start..real_end).step_by(2 + size_x);
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
            }
        }
        // now do the corners
        // copy into top left from bottom right
        let btm_right = self.with_ghost_dim.get_index(Pos {
            row: size_y,
            col: size_x,
        });
        unsafe { *fld.get_unchecked_mut(0) = *fld.get_unchecked(btm_right) }

        // copy into top right from bottom left
        let btm_left = self.with_ghost_dim.get_index(Pos {
            row: size_y,
            col: 1,
        });
        unsafe { *fld.get_unchecked_mut(size_x + 1) = *fld.get_unchecked(btm_left) }

        // copy into bottom left from top right
        let ghost_btm_left = self.with_ghost_dim.get_index(Pos {
            row: size_y + 1,
            col: 0,
        });
        let top_right = self.with_ghost_dim.get_index(Pos {
            row: 1,
            col: size_x,
        });
        unsafe { *fld.get_unchecked_mut(ghost_btm_left) = *fld.get_unchecked(top_right) }

        // Copy into bottom right from top left
        let ghost_btm_right = self.with_ghost_dim.get_index(Pos {
            row: size_y + 1,
            col: size_x + 1,
        });
        let top_left = self.with_ghost_dim.get_index(Pos { row: 1, col: 1 });
        unsafe { *fld.get_unchecked_mut(ghost_btm_right) = *fld.get_unchecked(top_left) }
    }

    #[inline(always)]
    pub fn deposit_ghosts(&mut self) -> () {
        let size_x = self.no_ghost_dim.size_x;
        let size_y = self.no_ghost_dim.size_y;
        if !cfg!(feature = "unchecked") {
            assert_eq!(self.spatial.len(), (size_x + 2) * (size_y + 2));
            assert_eq!(size_x, (self.with_ghost_dim.size_x - 2));
            assert_eq!(size_y, (self.with_ghost_dim.size_y - 2));
        }

        let fld = &mut self.spatial;

        // deposit top ghost row into last row
        let ghost_start = self.with_ghost_dim.get_index(Pos { row: 0, col: 1 });
        let ghost_range = ghost_start..ghost_start + size_x;
        let real_start = self.with_ghost_dim.get_index(Pos {
            row: size_y,
            col: 1,
        });
        let real_range = real_start..real_start + size_x;
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
            }
        }
        // deposit bottom ghost row into top real row
        let ghost_start = self.with_ghost_dim.get_index(Pos {
            row: size_y + 1,
            col: 1,
        });
        let ghost_range = ghost_start..ghost_start + size_x;
        let real_start = self.with_ghost_dim.get_index(Pos { row: 1, col: 1 });
        let real_range = real_start..real_start + size_x;
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
            }
        }
        // deposit left ghost columns into right real column
        let ghost_start = self.with_ghost_dim.get_index(Pos { row: 1, col: 0 });
        let ghost_end = self.with_ghost_dim.get_index(Pos {
            row: 1 + size_y,
            col: 0,
        });
        let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
        let real_start = self.with_ghost_dim.get_index(Pos {
            row: 1,
            col: size_x,
        });
        let real_end = self.with_ghost_dim.get_index(Pos {
            row: 1 + size_y,
            col: size_x,
        });
        let real_range = (real_start..real_end).step_by(2 + size_x);
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
            }
        }

        // deposit right ghost columns into left real column
        let ghost_start = self.with_ghost_dim.get_index(Pos {
            row: 1,
            col: size_x + 1,
        });
        let ghost_end = self.with_ghost_dim.get_index(Pos {
            row: 1 + size_y,
            col: size_x + 1,
        });
        let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
        let real_start = self.with_ghost_dim.get_index(Pos { row: 1, col: 1 });
        let real_end = self.with_ghost_dim.get_index(Pos {
            row: 1 + size_y,
            col: 1,
        });
        let real_range = (real_start..real_end).step_by(2 + size_x);
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
            }
        }
        // now do the corners
        // deposit top left into bottom right
        let btm_right = self.with_ghost_dim.get_index(Pos {
            row: size_y,
            col: size_x,
        });
        unsafe { *fld.get_unchecked_mut(btm_right) += *fld.get_unchecked(0) }

        // depost top right into bottom left
        let btm_left = self.with_ghost_dim.get_index(Pos {
            row: size_y,
            col: 1,
        });
        unsafe { *fld.get_unchecked_mut(btm_left) += *fld.get_unchecked(size_x + 1) }

        // depost bottom left into top right
        let ghost_btm_left = self.with_ghost_dim.get_index(Pos {
            row: size_y + 1,
            col: 0,
        });
        let top_right = self.with_ghost_dim.get_index(Pos {
            row: 1,
            col: size_x,
        });
        unsafe { *fld.get_unchecked_mut(top_right) += *fld.get_unchecked(ghost_btm_left) }

        // depost bottom right into top left
        let ghost_btm_right = self.with_ghost_dim.get_index(Pos {
            row: size_y + 1,
            col: size_x + 1,
        });
        let top_left = self.with_ghost_dim.get_index(Pos { row: 1, col: 1 });
        unsafe { *fld.get_unchecked_mut(top_left) += *fld.get_unchecked(ghost_btm_right) }
    }

    #[inline(always)]
    pub fn binomial_filter_2_d(&mut self, field_buf: &mut Field) {
        let in_vec = &self.spatial;
        let size_x = self.no_ghost_dim.size_x;
        let size_y = self.no_ghost_dim.size_y;
        // wrkspace should be same size as fld
        if !cfg!(feature = "unchecked") {
            assert!(in_vec.len() == field_buf.spatial.len());
            assert!(in_vec.len() == (size_x + 2) * (size_y + 2));
        }
        let wrkspace = &mut field_buf.spatial;
        for iy in 1..size_y + 1 {
            for ix in 1..size_x + 1 {
                let mut ijm1 = iy - 1;
                let mut ijp1 = iy + 1;
                let ij = iy * (2 + size_x);
                ijm1 *= 2 + size_x;
                ijp1 *= 2 + size_x;
                // CALC WEIGHTS
                // 2D binomial filter
                // The weighting scheme prtl is in middle
                // +------------------
                // |  1  |  2  |  1  |
                // -------------------
                // |  2  |  4  |  2  |   X   1/16
                // -------------------
                // |  1  |  2  |  1  |
                // -------------------

                // safe because of assertion that vec_size is (size_x + 2)* size_y+2)

                unsafe {
                    let mut res = 0.0625 * in_vec.get_unchecked(ijm1 + ix - 1);
                    res += 0.125 * in_vec.get_unchecked(ijm1 + ix);
                    res += 0.0625 * in_vec.get_unchecked(ijm1 + ix + 1);
                    res += 0.125 * in_vec.get_unchecked(ij + ix - 1);
                    res += 0.25 * in_vec.get_unchecked(ij + ix);
                    res += 0.125 * in_vec.get_unchecked(ij + ix + 1);
                    res += 0.0625 * in_vec.get_unchecked(ijp1 + ix - 1);
                    res += 0.125 * in_vec.get_unchecked(ijp1 + ix);
                    res += 0.0625 * in_vec.get_unchecked(ijp1 + ix + 1);
                    *wrkspace.get_unchecked_mut(ij + ix) = res;
                }
            }
        }

        field_buf.update_ghosts();
        for (v1, v2) in self.spatial.iter_mut().zip(field_buf.spatial.iter()) {
            *v1 = *v2;
        }
    }
}
#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{build_test_sim, E_TOL};
    #[test]
    fn field_init() {
        // checks that all fields are intialized to the correct
        // size and to zero
        let expected_spatial_val: Vec<Float> = vec![0.; (24 + 2) * (12 + 2)];
        let expected_complex_val: Vec<Complex<Float>> = vec![Complex::zero(); 24 * 12];

        let sim = build_test_sim();
        let fld = Field::default(&sim);
        assert_eq!(fld.with_ghost_dim.size_x, sim.size_x + 2);
        assert_eq!(fld.with_ghost_dim.size_y, sim.size_y + 2);
        assert_eq!(fld.no_ghost_dim.size_x, sim.size_x);
        assert_eq!(fld.no_ghost_dim.size_y, sim.size_y);
        assert_eq!(fld.spatial.len(), expected_spatial_val.len());
        assert_eq!(
            fld.spatial.len(),
            fld.with_ghost_dim.size_x * fld.with_ghost_dim.size_y
        );
        assert_eq!(fld.spectral.len(), expected_complex_val.len());
        assert_eq!(
            fld.spectral.len(),
            fld.no_ghost_dim.size_x * fld.no_ghost_dim.size_y
        );
        for (v, expected_v) in fld.spatial.iter().zip(expected_spatial_val.iter()) {
            assert_eq!(v, expected_v);
        }
        for (v, expected_v) in fld.spectral.iter().zip(expected_complex_val.iter()) {
            assert_eq!(v, expected_v);
        }
    }

    #[test]
    fn row_major_order() {
        let sim = build_test_sim();
        let fld = Field::default(&sim);
        //iterate over rows as outer iter

        let mut index = 0;
        for i in 0..fld.with_ghost_dim.size_y {
            for j in 0..fld.with_ghost_dim.size_x {
                assert_eq!(fld.with_ghost_dim.get_index(Pos { row: i, col: j }), index);
                index += 1;
            }
        }

        let mut index = 0;
        for i in 0..fld.no_ghost_dim.size_y {
            for j in 0..fld.no_ghost_dim.size_x {
                assert_eq!(fld.no_ghost_dim.get_index(Pos { row: i, col: j }), index);
                index += 1;
            }
        }
    }

    #[test]
    fn copy_to_spectral() {
        let input: Vec<Float> = vec![
            0.668, 0.157, 0.249, 0.448, 0.762, 0.348, 0.242, 0.282, 0.41, 0.894, 0.856, 0.794,
            0.354, 0.925, 0.563, 0.65, 0.341, 0.798, 0.223, 0.608, 0.427, 0.102, 0.945, 0.832,
            0.668, 0.157, 0.46, 0.691, 0.675, 0.495, 0.024, 0.207, 0.413, 0.254, 0.123, 0.131,
            0.814, 0.18, 0.863, 0.605, 0.946, 0.994, 0.949, 0.06, 0.644, 0.651, 0.6, 0.89, 0.206,
            0.893, 0.46, 0.691, 0.804, 0.747, 0.437, 0.773, 0.584, 0.681, 0.766, 0.732, 0.477,
            0.565, 0.517, 0.923, 0.757, 0.3, 0.186, 0.759, 0.973, 0.927, 0.962, 0.389, 0.963,
            0.339, 0.778, 0.496, 0.804, 0.747, 0.143, 0.354, 0.301, 0.766, 0.203, 0.587, 0.462,
            0.571, 0.496, 0.68, 0.664, 0.994, 0.608, 0.713, 0.446, 0.324, 0.421, 0.987, 0.195,
            0.56, 0.486, 0.149, 0.255, 0.562, 0.143, 0.354, 0.818, 0.965, 0.691, 0.399, 0.953,
            0.997, 0.035, 0.928, 0.277, 0.442, 0.05, 0.254, 0.784, 0.345, 0.26, 0.759, 0.113,
            0.699, 0.268, 0.867, 0.613, 0.271, 0.789, 0.83, 0.818, 0.965, 0.16, 0.701, 0.182,
            0.335, 0.959, 0.013, 0.915, 0.598, 0.777, 0.441, 0.437, 0.582, 0.897, 0.32, 0.584,
            0.824, 0.698, 0.214, 0.442, 0.708, 0.384, 0.862, 0.377, 0.359, 0.16, 0.701, 0.319,
            0.943, 0.947, 0.126, 0.481, 0.483, 0.967, 0.187, 0.297, 0.069, 0.283, 0.513, 0.669,
            0.314, 0.583, 0.992, 0.934, 0.24, 0.3, 0.39, 0.307, 0.353, 0.44, 0.752, 0.319, 0.943,
            0.368, 0.099, 0.049, 0.273, 0.062, 0.01, 0.702, 0.158, 0.526, 0.674, 0.101, 0.252,
            0.963, 0.661, 0.747, 0.474, 0.906, 0.215, 0.188, 0.375, 0.496, 0.103, 0.596, 0.228,
            0.368, 0.099, 0.472, 0.906, 0.096, 0.486, 0.533, 0.181, 0.235, 0.905, 0.862, 0.44,
            0.62, 0.655, 0.024, 0.513, 0.748, 0.533, 0.209, 0.246, 0.706, 0.631, 0.112, 0.693,
            0.379, 0.576, 0.472, 0.906, 0.997, 0.728, 0.439, 0.416, 0.301, 0.071, 0.287, 0.946,
            0.211, 0.664, 0.394, 0.532, 0.588, 0.368, 0.924, 0.771, 0.102, 0.24, 0.13, 0.12, 0.955,
            0.407, 0.741, 0.73, 0.997, 0.728, 0.74, 0.115, 0.109, 0.455, 0.355, 0.302, 0.364,
            0.207, 0.988, 0.158, 0.03, 0.498, 0.931, 0.92, 0.485, 0.698, 0.601, 0.568, 0.743,
            0.794, 0.2, 0.553, 0.392, 0.552, 0.74, 0.115, 0.238, 0.277, 0.092, 0.747, 0.361, 0.072,
            0.778, 0.428, 0.702, 0.811, 0.116, 0.666, 0.155, 0.987, 0.163, 0.411, 0.124, 0.499,
            0.861, 0.911, 0.468, 0.959, 0.705, 0.724, 0.238, 0.277, 0.668, 0.157, 0.249, 0.448,
            0.762, 0.348, 0.242, 0.282, 0.41, 0.894, 0.856, 0.794, 0.354, 0.925, 0.563, 0.65,
            0.341, 0.798, 0.223, 0.608, 0.427, 0.102, 0.945, 0.832, 0.668, 0.157, 0.46, 0.691,
            0.675, 0.495, 0.024, 0.207, 0.413, 0.254, 0.123, 0.131, 0.814, 0.18, 0.863, 0.605,
            0.946, 0.994, 0.949, 0.06, 0.644, 0.651, 0.6, 0.89, 0.206, 0.893, 0.46, 0.691,
        ];
        let expected_output: Vec<Float> = vec![
            0.691, 0.675, 0.495, 0.024, 0.207, 0.413, 0.254, 0.123, 0.131, 0.814, 0.18, 0.863,
            0.605, 0.946, 0.994, 0.949, 0.06, 0.644, 0.651, 0.6, 0.89, 0.206, 0.893, 0.46, 0.747,
            0.437, 0.773, 0.584, 0.681, 0.766, 0.732, 0.477, 0.565, 0.517, 0.923, 0.757, 0.3,
            0.186, 0.759, 0.973, 0.927, 0.962, 0.389, 0.963, 0.339, 0.778, 0.496, 0.804, 0.354,
            0.301, 0.766, 0.203, 0.587, 0.462, 0.571, 0.496, 0.68, 0.664, 0.994, 0.608, 0.713,
            0.446, 0.324, 0.421, 0.987, 0.195, 0.56, 0.486, 0.149, 0.255, 0.562, 0.143, 0.965,
            0.691, 0.399, 0.953, 0.997, 0.035, 0.928, 0.277, 0.442, 0.05, 0.254, 0.784, 0.345,
            0.26, 0.759, 0.113, 0.699, 0.268, 0.867, 0.613, 0.271, 0.789, 0.83, 0.818, 0.701,
            0.182, 0.335, 0.959, 0.013, 0.915, 0.598, 0.777, 0.441, 0.437, 0.582, 0.897, 0.32,
            0.584, 0.824, 0.698, 0.214, 0.442, 0.708, 0.384, 0.862, 0.377, 0.359, 0.16, 0.943,
            0.947, 0.126, 0.481, 0.483, 0.967, 0.187, 0.297, 0.069, 0.283, 0.513, 0.669, 0.314,
            0.583, 0.992, 0.934, 0.24, 0.3, 0.39, 0.307, 0.353, 0.44, 0.752, 0.319, 0.099, 0.049,
            0.273, 0.062, 0.01, 0.702, 0.158, 0.526, 0.674, 0.101, 0.252, 0.963, 0.661, 0.747,
            0.474, 0.906, 0.215, 0.188, 0.375, 0.496, 0.103, 0.596, 0.228, 0.368, 0.906, 0.096,
            0.486, 0.533, 0.181, 0.235, 0.905, 0.862, 0.44, 0.62, 0.655, 0.024, 0.513, 0.748,
            0.533, 0.209, 0.246, 0.706, 0.631, 0.112, 0.693, 0.379, 0.576, 0.472, 0.728, 0.439,
            0.416, 0.301, 0.071, 0.287, 0.946, 0.211, 0.664, 0.394, 0.532, 0.588, 0.368, 0.924,
            0.771, 0.102, 0.24, 0.13, 0.12, 0.955, 0.407, 0.741, 0.73, 0.997, 0.115, 0.109, 0.455,
            0.355, 0.302, 0.364, 0.207, 0.988, 0.158, 0.03, 0.498, 0.931, 0.92, 0.485, 0.698,
            0.601, 0.568, 0.743, 0.794, 0.2, 0.553, 0.392, 0.552, 0.74, 0.277, 0.092, 0.747, 0.361,
            0.072, 0.778, 0.428, 0.702, 0.811, 0.116, 0.666, 0.155, 0.987, 0.163, 0.411, 0.124,
            0.499, 0.861, 0.911, 0.468, 0.959, 0.705, 0.724, 0.238, 0.157, 0.249, 0.448, 0.762,
            0.348, 0.242, 0.282, 0.41, 0.894, 0.856, 0.794, 0.354, 0.925, 0.563, 0.65, 0.341,
            0.798, 0.223, 0.608, 0.427, 0.102, 0.945, 0.832, 0.668,
        ];

        let sim = build_test_sim();

        assert_eq!(sim.size_x * sim.size_y, expected_output.len());
        assert_eq!(input.len(), (sim.size_x + 2) * (sim.size_y + 2));

        let mut test_fld = Field::default(&sim);
        test_fld.spatial = input.clone();

        test_fld.copy_to_spectral();

        assert_eq!(test_fld.spatial.len(), input.len());

        for (v1, v2) in test_fld.spatial.iter().zip(input) {
            assert_eq!(*v1, v2);
        }

        assert_eq!(test_fld.spectral.len(), expected_output.len());

        for (v1, v2) in test_fld.spectral.iter().zip(expected_output) {
            assert_eq!(v1.re, v2);
            assert_eq!(v1.im, 0.0);
        }
    }

    #[test]
    fn copy_to_spatial() {
        let expected_output: Vec<Float> = vec![
            0.668, 0.157, 0.249, 0.448, 0.762, 0.348, 0.242, 0.282, 0.41, 0.894, 0.856, 0.794,
            0.354, 0.925, 0.563, 0.65, 0.341, 0.798, 0.223, 0.608, 0.427, 0.102, 0.945, 0.832,
            0.668, 0.157, 0.46, 0.691, 0.675, 0.495, 0.024, 0.207, 0.413, 0.254, 0.123, 0.131,
            0.814, 0.18, 0.863, 0.605, 0.946, 0.994, 0.949, 0.06, 0.644, 0.651, 0.6, 0.89, 0.206,
            0.893, 0.46, 0.691, 0.804, 0.747, 0.437, 0.773, 0.584, 0.681, 0.766, 0.732, 0.477,
            0.565, 0.517, 0.923, 0.757, 0.3, 0.186, 0.759, 0.973, 0.927, 0.962, 0.389, 0.963,
            0.339, 0.778, 0.496, 0.804, 0.747, 0.143, 0.354, 0.301, 0.766, 0.203, 0.587, 0.462,
            0.571, 0.496, 0.68, 0.664, 0.994, 0.608, 0.713, 0.446, 0.324, 0.421, 0.987, 0.195,
            0.56, 0.486, 0.149, 0.255, 0.562, 0.143, 0.354, 0.818, 0.965, 0.691, 0.399, 0.953,
            0.997, 0.035, 0.928, 0.277, 0.442, 0.05, 0.254, 0.784, 0.345, 0.26, 0.759, 0.113,
            0.699, 0.268, 0.867, 0.613, 0.271, 0.789, 0.83, 0.818, 0.965, 0.16, 0.701, 0.182,
            0.335, 0.959, 0.013, 0.915, 0.598, 0.777, 0.441, 0.437, 0.582, 0.897, 0.32, 0.584,
            0.824, 0.698, 0.214, 0.442, 0.708, 0.384, 0.862, 0.377, 0.359, 0.16, 0.701, 0.319,
            0.943, 0.947, 0.126, 0.481, 0.483, 0.967, 0.187, 0.297, 0.069, 0.283, 0.513, 0.669,
            0.314, 0.583, 0.992, 0.934, 0.24, 0.3, 0.39, 0.307, 0.353, 0.44, 0.752, 0.319, 0.943,
            0.368, 0.099, 0.049, 0.273, 0.062, 0.01, 0.702, 0.158, 0.526, 0.674, 0.101, 0.252,
            0.963, 0.661, 0.747, 0.474, 0.906, 0.215, 0.188, 0.375, 0.496, 0.103, 0.596, 0.228,
            0.368, 0.099, 0.472, 0.906, 0.096, 0.486, 0.533, 0.181, 0.235, 0.905, 0.862, 0.44,
            0.62, 0.655, 0.024, 0.513, 0.748, 0.533, 0.209, 0.246, 0.706, 0.631, 0.112, 0.693,
            0.379, 0.576, 0.472, 0.906, 0.997, 0.728, 0.439, 0.416, 0.301, 0.071, 0.287, 0.946,
            0.211, 0.664, 0.394, 0.532, 0.588, 0.368, 0.924, 0.771, 0.102, 0.24, 0.13, 0.12, 0.955,
            0.407, 0.741, 0.73, 0.997, 0.728, 0.74, 0.115, 0.109, 0.455, 0.355, 0.302, 0.364,
            0.207, 0.988, 0.158, 0.03, 0.498, 0.931, 0.92, 0.485, 0.698, 0.601, 0.568, 0.743,
            0.794, 0.2, 0.553, 0.392, 0.552, 0.74, 0.115, 0.238, 0.277, 0.092, 0.747, 0.361, 0.072,
            0.778, 0.428, 0.702, 0.811, 0.116, 0.666, 0.155, 0.987, 0.163, 0.411, 0.124, 0.499,
            0.861, 0.911, 0.468, 0.959, 0.705, 0.724, 0.238, 0.277, 0.668, 0.157, 0.249, 0.448,
            0.762, 0.348, 0.242, 0.282, 0.41, 0.894, 0.856, 0.794, 0.354, 0.925, 0.563, 0.65,
            0.341, 0.798, 0.223, 0.608, 0.427, 0.102, 0.945, 0.832, 0.668, 0.157, 0.46, 0.691,
            0.675, 0.495, 0.024, 0.207, 0.413, 0.254, 0.123, 0.131, 0.814, 0.18, 0.863, 0.605,
            0.946, 0.994, 0.949, 0.06, 0.644, 0.651, 0.6, 0.89, 0.206, 0.893, 0.46, 0.691,
        ];

        let input: Vec<Float> = vec![
            0.691, 0.675, 0.495, 0.024, 0.207, 0.413, 0.254, 0.123, 0.131, 0.814, 0.18, 0.863,
            0.605, 0.946, 0.994, 0.949, 0.06, 0.644, 0.651, 0.6, 0.89, 0.206, 0.893, 0.46, 0.747,
            0.437, 0.773, 0.584, 0.681, 0.766, 0.732, 0.477, 0.565, 0.517, 0.923, 0.757, 0.3,
            0.186, 0.759, 0.973, 0.927, 0.962, 0.389, 0.963, 0.339, 0.778, 0.496, 0.804, 0.354,
            0.301, 0.766, 0.203, 0.587, 0.462, 0.571, 0.496, 0.68, 0.664, 0.994, 0.608, 0.713,
            0.446, 0.324, 0.421, 0.987, 0.195, 0.56, 0.486, 0.149, 0.255, 0.562, 0.143, 0.965,
            0.691, 0.399, 0.953, 0.997, 0.035, 0.928, 0.277, 0.442, 0.05, 0.254, 0.784, 0.345,
            0.26, 0.759, 0.113, 0.699, 0.268, 0.867, 0.613, 0.271, 0.789, 0.83, 0.818, 0.701,
            0.182, 0.335, 0.959, 0.013, 0.915, 0.598, 0.777, 0.441, 0.437, 0.582, 0.897, 0.32,
            0.584, 0.824, 0.698, 0.214, 0.442, 0.708, 0.384, 0.862, 0.377, 0.359, 0.16, 0.943,
            0.947, 0.126, 0.481, 0.483, 0.967, 0.187, 0.297, 0.069, 0.283, 0.513, 0.669, 0.314,
            0.583, 0.992, 0.934, 0.24, 0.3, 0.39, 0.307, 0.353, 0.44, 0.752, 0.319, 0.099, 0.049,
            0.273, 0.062, 0.01, 0.702, 0.158, 0.526, 0.674, 0.101, 0.252, 0.963, 0.661, 0.747,
            0.474, 0.906, 0.215, 0.188, 0.375, 0.496, 0.103, 0.596, 0.228, 0.368, 0.906, 0.096,
            0.486, 0.533, 0.181, 0.235, 0.905, 0.862, 0.44, 0.62, 0.655, 0.024, 0.513, 0.748,
            0.533, 0.209, 0.246, 0.706, 0.631, 0.112, 0.693, 0.379, 0.576, 0.472, 0.728, 0.439,
            0.416, 0.301, 0.071, 0.287, 0.946, 0.211, 0.664, 0.394, 0.532, 0.588, 0.368, 0.924,
            0.771, 0.102, 0.24, 0.13, 0.12, 0.955, 0.407, 0.741, 0.73, 0.997, 0.115, 0.109, 0.455,
            0.355, 0.302, 0.364, 0.207, 0.988, 0.158, 0.03, 0.498, 0.931, 0.92, 0.485, 0.698,
            0.601, 0.568, 0.743, 0.794, 0.2, 0.553, 0.392, 0.552, 0.74, 0.277, 0.092, 0.747, 0.361,
            0.072, 0.778, 0.428, 0.702, 0.811, 0.116, 0.666, 0.155, 0.987, 0.163, 0.411, 0.124,
            0.499, 0.861, 0.911, 0.468, 0.959, 0.705, 0.724, 0.238, 0.157, 0.249, 0.448, 0.762,
            0.348, 0.242, 0.282, 0.41, 0.894, 0.856, 0.794, 0.354, 0.925, 0.563, 0.65, 0.341,
            0.798, 0.223, 0.608, 0.427, 0.102, 0.945, 0.832, 0.668,
        ];

        let sim = build_test_sim();
        assert_eq!(sim.size_x * sim.size_y, input.len());
        assert_eq!(expected_output.len(), (sim.size_x + 2) * (sim.size_y + 2));

        let mut test_fld = Field::default(&sim);

        for (v1, v2) in test_fld.spectral.iter_mut().zip(input.iter()) {
            v1.re = *v2;
        }

        test_fld.copy_to_spatial(&sim);

        assert_eq!(expected_output.len(), test_fld.spatial.len());
        for (v1, v2) in test_fld.spatial.iter().zip(expected_output.iter()) {
            assert_eq!(*v1, *v2);
        }

        assert_eq!(input.len(), test_fld.spectral.len());
        for (v1, v2) in test_fld.spectral.iter().zip(input.iter()) {
            assert_eq!(v1.re, *v2);
            assert_eq!(v1.im, 0.0);
        }
    }
    #[test]
    fn ghost_deposit() {
        // made an example using google sheets
        let input: Vec<Float> = vec![
            1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.8, 1.8, 1.8, 1.8,
            1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1,
            1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.3, 1.3, 1.3, 1.3, 1.3,
            1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3,
        ];
        let expected_output: Vec<Float> = vec![
            1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.8, 1.8, 1.8, 1.8,
            1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.4, 4.21, 1.11, 1.11, 1.11, 1.11, 1.11,
            1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31,
            1.31, 1.31, 1.31, 3.81, 1.8, 1.4, 1.81, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.41, 1.8, 1.4, 1.81, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.41, 1.8, 1.4, 1.81,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.41, 1.8, 1.4, 1.81, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 1.41, 1.8, 1.1, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 1.11, 1.3, 1.1, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11,
            1.3, 1.1, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.3, 1.1, 1.31, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.3, 1.1, 1.31, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 1.11, 1.3, 1.1, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 1.11, 1.3, 1.1, 4.51, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41,
            1.41, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 4.31, 1.3, 1.1,
            1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.3, 1.3, 1.3, 1.3, 1.3,
            1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3,
        ];
        let sim = build_test_sim();
        let mut fld = Field::default(&sim);
        fld.spatial = input.clone();
        assert_eq!(fld.spatial.len(), expected_output.len());
        assert_eq!(fld.spatial.len(), (sim.size_x + 2) * (sim.size_y + 2));
        fld.deposit_ghosts();
        assert_eq!(fld.spatial.len(), (sim.size_x + 2) * (sim.size_y + 2));
        for (v, expected_v) in fld.spatial.into_iter().zip(expected_output) {
            assert!((v - expected_v).abs() < E_TOL);
        }
    }

    #[test]
    fn ghosts_update() {
        // made an example using google sheets
        let input: Vec<Float> = vec![
            1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.8, 1.8, 1.8, 1.8,
            1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1,
            1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.3, 1.3, 1.3, 1.3, 1.3,
            1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3,
        ];
        let expected_output: Vec<Float> = vec![
            4.31, 4.51, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.81,
            1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 4.31, 4.51, 3.81, 4.21,
            1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.31, 1.31, 1.31,
            1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 3.81, 4.21, 1.41, 1.81, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.41, 1.81, 1.41, 1.81, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 1.41, 1.81, 1.41, 1.81, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 1.41, 1.81, 1.41, 1.81, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.41, 1.81, 1.11, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.31,
            1.11, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.31, 1.11, 1.31,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.31, 1.11, 1.31, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.31, 1.11, 1.31, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 1.11, 1.31, 1.11, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 1.11, 1.31, 4.31, 4.51, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41,
            1.41, 1.41, 1.41, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81,
            4.31, 4.51, 3.81, 4.21, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11,
            1.11, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 3.81, 4.21,
        ];
        let mut fld = Field::default(&build_test_sim());
        fld.spatial = input.clone();
        assert_eq!(input.len(), expected_output.len());
        assert_eq!(
            input.len(),
            fld.with_ghost_dim.size_x * fld.with_ghost_dim.size_y
        );
        fld.deposit_ghosts();
        fld.update_ghosts();
        for (v, expected_v) in fld.spatial.into_iter().zip(expected_output) {
            assert!((v - expected_v).abs() < E_TOL);
        }
    }

    #[test]
    fn binomial_filter() {
        let sim = build_test_sim();
        let mut fld = Field::default(&sim);
        let mut wrkspace = Field::default(&sim);
        let mut expected_out = Field::default(&sim);
        let input = vec![
            0.6226363025150204,
            0.8736968516461667,
            0.6313246531427373,
            0.9148662650317788,
            0.5353520426271431,
            0.7723080641713831,
            0.30830334890087685,
            0.7896678860213547,
            0.46285168819097655,
            0.649335790063356,
            0.3045376443132819,
            0.42203248531568416,
            0.8940293130207309,
            0.9998204363675148,
            0.22538954850823567,
            0.3431440723150858,
            0.9111244187689314,
            0.06771933102673611,
            0.7129856018467182,
            0.7292061170275415,
            0.5278567470389417,
            0.738346083617137,
            0.3622634535026632,
            0.4594872299582007,
            0.6226363025150204,
            0.8736968516461667,
            0.23254157512182516,
            0.8754218580998687,
            0.8196211242893664,
            0.6153589212871639,
            0.9443058438379447,
            0.3876664281852297,
            0.4297447758423335,
            0.3208986607179327,
            0.5241797458098827,
            0.4807642535183938,
            0.3465012111432544,
            0.06601497783523991,
            0.0884403654014202,
            0.12076738489431538,
            0.604213398924789,
            0.2301524725476759,
            0.5560215771413893,
            0.3524292501977686,
            0.07519885383786795,
            0.23003969910392652,
            0.08112346773125134,
            0.1869475636334459,
            0.6150080938809916,
            0.03269874432047615,
            0.23254157512182516,
            0.8754218580998687,
            0.9793000852834151,
            0.38009803070443315,
            0.9895472671573626,
            0.6758192508628587,
            0.04865091955988354,
            0.11786426518832327,
            0.8886282577350411,
            0.281497687634673,
            0.4264012341756579,
            0.06355112966911025,
            0.9234273583417115,
            0.7783153261792898,
            0.5044669032080984,
            0.4070309849544166,
            0.611686946737498,
            0.9780688819203818,
            0.9035029196308407,
            0.2931174567462479,
            0.614799690555815,
            0.9885999455772961,
            0.6106431416220134,
            0.6347821653409537,
            0.6282513876989351,
            0.34502743744937314,
            0.9793000852834151,
            0.38009803070443315,
            0.45443739412580786,
            0.31387267809374564,
            0.8217161894165835,
            0.7800260394049603,
            0.325173587444614,
            0.7020516641362825,
            0.7069961059835483,
            0.09944452904731926,
            0.12230844868682134,
            0.12829956373327667,
            0.0684754991941916,
            0.5172409583814983,
            0.9806914939051223,
            0.7389721110945544,
            0.21817812724368224,
            0.5760358449880334,
            0.2679643707499989,
            0.9690262950898088,
            0.6364750359220441,
            0.65395677977867,
            0.11333121976422988,
            0.5956957537726449,
            0.29714276633388503,
            0.10146281268363688,
            0.45443739412580786,
            0.31387267809374564,
            0.6689416871871577,
            0.7883301598388488,
            0.22998234919061744,
            0.9241382267500085,
            0.8164194231744707,
            0.620040208692632,
            0.9473281124960854,
            0.9711059406925603,
            0.8214349761420985,
            0.34416552260742905,
            0.7872254861528508,
            0.10269669205687548,
            0.9815623841047764,
            0.4974415556372809,
            0.07102825002888158,
            0.0979768666036347,
            0.2431955743645241,
            0.25364676153388666,
            0.47969945185469043,
            0.5188239553375823,
            0.5305055221026139,
            0.6510920010516495,
            0.9015352357655272,
            0.9910483935822859,
            0.6689416871871577,
            0.7883301598388488,
            0.9633411252220122,
            0.15080814427026878,
            0.313068334550492,
            0.24046412776128057,
            0.25920612247027663,
            0.10808444840872078,
            0.7609984511016052,
            0.9561422126302774,
            0.9326703005937645,
            0.030382678706940225,
            0.6555319048307936,
            0.930241793588751,
            0.08861299977438408,
            0.07982675372261872,
            0.8216094702973412,
            0.8025524585745841,
            0.6024773760171437,
            0.9081738698689505,
            0.011248562500916104,
            0.7295334824349476,
            0.8823816801472655,
            0.5949166692943862,
            0.749440076528082,
            0.9151536584774281,
            0.9633411252220122,
            0.15080814427026878,
            0.8526269577853129,
            0.609352932524706,
            0.2495113042184941,
            0.1872508541228871,
            0.923228253163143,
            0.6231204602504072,
            0.5340963298857548,
            0.7573897330586413,
            0.02914873972575016,
            0.3881632879327942,
            0.9460497136216258,
            0.03255141574506515,
            0.07517113562777566,
            0.10029490316602796,
            0.0068160288404607305,
            0.21853899093066287,
            0.16678517241291424,
            0.5082040260146182,
            0.4547264666979748,
            0.37720330755254305,
            0.5288821748709395,
            0.5512655716797082,
            0.08943129043728515,
            0.9677566398195793,
            0.8526269577853129,
            0.609352932524706,
            0.7444178390133032,
            0.28469869232400147,
            0.08887679665794701,
            0.8947899843650676,
            0.26741655340017445,
            0.3065706756119446,
            0.34806987300812764,
            0.7827854916383926,
            0.9421602873749939,
            0.25020711235903437,
            0.18458491924055764,
            0.3937813886008582,
            0.48227951377081923,
            0.816223906287338,
            0.942030455627183,
            0.12741064361741083,
            0.05062102252711653,
            0.3881262823526117,
            0.7849602981118912,
            0.24463285431650095,
            0.5199761985956127,
            0.377446187505201,
            0.31062172545151345,
            0.042555105874163335,
            0.7444178390133032,
            0.28469869232400147,
            0.6672671656302147,
            0.8010253306484707,
            0.6843324915741085,
            0.4891338404067158,
            0.5458710386615103,
            0.4814985254729476,
            0.7955464415706903,
            0.2045135614174295,
            0.5971853982994425,
            0.11601897066577471,
            0.28481122054879326,
            0.7533902668097618,
            0.7342101112420762,
            0.14343595456396774,
            0.7206958435471132,
            0.8442386537398704,
            0.28965069441486657,
            0.8710832994729301,
            0.3826714478407156,
            0.5530307491801266,
            0.6757002853978776,
            0.5134790397000708,
            0.9888916137510659,
            0.5570550171046834,
            0.6672671656302147,
            0.8010253306484707,
            0.3497661991344927,
            0.767363616642517,
            0.3785440286165791,
            0.7621045356752713,
            0.07165262952486562,
            0.49598715120335823,
            0.11104825293027598,
            0.15722960373929884,
            0.20226174027249455,
            0.7630858656525251,
            0.14244511498831125,
            0.6495734778520625,
            0.011083782708110279,
            0.2800846368321297,
            0.59497493089738,
            0.2932425234849785,
            0.17417184003043318,
            0.7207339294908739,
            0.2739278315508884,
            0.9690423075467055,
            0.5703750177274564,
            0.15595716929253012,
            0.20723282946376198,
            0.33192906310585335,
            0.3497661991344927,
            0.767363616642517,
            0.9970040188943813,
            0.43302230769426475,
            0.8176614429087246,
            0.4641945153542535,
            0.6577357236152713,
            0.14558385156236875,
            0.17846179991613487,
            0.5786998271367522,
            0.8045627733355899,
            0.7982067139611896,
            0.0022347582659495435,
            0.26136214426075455,
            0.5170059959057882,
            0.4905460497076415,
            0.7515928381657782,
            0.24294001888530248,
            0.7113096018227462,
            0.6935287771621821,
            0.72434288721977,
            0.7234070039923058,
            0.17467970607208483,
            0.24031725661422199,
            0.6250196416173749,
            0.9400319389968005,
            0.9970040188943813,
            0.43302230769426475,
            0.2765006530965286,
            0.5178020139312068,
            0.3215164149950773,
            0.9267666450478018,
            0.7712380172204503,
            0.9834189626246733,
            0.22253758016918201,
            0.567926744007949,
            0.3436795662306408,
            0.5809625298678344,
            0.6567536469018707,
            0.7080613759001594,
            0.40925325246209987,
            0.9388548450940358,
            0.5355294904679163,
            0.8869113637052705,
            0.5951284219582624,
            0.9634071075595242,
            0.8925791227446886,
            0.12114915140944504,
            0.5330147291223981,
            0.013546696056700447,
            0.3321319627732855,
            0.28249560727577294,
            0.2765006530965286,
            0.5178020139312068,
            0.6226363025150204,
            0.8736968516461667,
            0.6313246531427373,
            0.9148662650317788,
            0.5353520426271431,
            0.7723080641713831,
            0.30830334890087685,
            0.7896678860213547,
            0.46285168819097655,
            0.649335790063356,
            0.3045376443132819,
            0.42203248531568416,
            0.8940293130207309,
            0.9998204363675148,
            0.22538954850823567,
            0.3431440723150858,
            0.9111244187689314,
            0.06771933102673611,
            0.7129856018467182,
            0.7292061170275415,
            0.5278567470389417,
            0.738346083617137,
            0.3622634535026632,
            0.4594872299582007,
            0.6226363025150204,
            0.8736968516461667,
            0.23254157512182516,
            0.8754218580998687,
            0.8196211242893664,
            0.6153589212871639,
            0.9443058438379447,
            0.3876664281852297,
            0.4297447758423335,
            0.3208986607179327,
            0.5241797458098827,
            0.4807642535183938,
            0.3465012111432544,
            0.06601497783523991,
            0.0884403654014202,
            0.12076738489431538,
            0.604213398924789,
            0.2301524725476759,
            0.5560215771413893,
            0.3524292501977686,
            0.07519885383786795,
            0.23003969910392652,
            0.08112346773125134,
            0.1869475636334459,
            0.6150080938809916,
            0.03269874432047615,
            0.23254157512182516,
            0.8754218580998687,
        ];
        expected_out.spatial = vec![
            0.5318679152084202,
            0.6041640131767214,
            0.6531742758290959,
            0.6577552256183432,
            0.6178842169915131,
            0.556480729424416,
            0.506484499257798,
            0.4846729452431421,
            0.4802175242051586,
            0.47547449971726774,
            0.4715274098537896,
            0.4824981896465676,
            0.5085520841937188,
            0.5326829130624663,
            0.5461977731486505,
            0.5550358734910248,
            0.559725254472766,
            0.5516466189441388,
            0.5260323327988242,
            0.48673194181531754,
            0.4466296633405282,
            0.42327802032924,
            0.4285506766429199,
            0.46642265547453765,
            0.5318679152084202,
            0.6041640131767214,
            0.5331708254981768,
            0.6098250764499792,
            0.6575178118496154,
            0.6508242876230148,
            0.6030018548682494,
            0.5466213819267685,
            0.5017683383663064,
            0.46995921564781606,
            0.44971880664316943,
            0.44366795222677746,
            0.453202770368208,
            0.474048289745795,
            0.49648346766462104,
            0.5130065594913967,
            0.525767264834882,
            0.5367718682912068,
            0.53837867281799,
            0.52635221153758,
            0.5069277681315055,
            0.48495140141022663,
            0.4626437940388718,
            0.44583288544927435,
            0.44381661752990964,
            0.47053686281167,
            0.5331708254981768,
            0.6098250764499792,
            0.5458390740579919,
            0.6035429064131435,
            0.6368085309281997,
            0.6229079972591803,
            0.5799800454956429,
            0.5407889995957714,
            0.5094837058073337,
            0.4714748797115328,
            0.4338933731428194,
            0.4225709849314898,
            0.44521870244089723,
            0.4803767123714944,
            0.5018447101143146,
            0.5054354358630471,
            0.507187954347425,
            0.516016897190763,
            0.5231798844417547,
            0.5221173534076616,
            0.5163311967665113,
            0.5074840909160825,
            0.49586650967843315,
            0.4857782258584896,
            0.48371587786094866,
            0.5008550815636161,
            0.5458390740579919,
            0.6035429064131435,
            0.5720577168762397,
            0.5884805878281718,
            0.5962791738987869,
            0.5855145469840709,
            0.5668163521101278,
            0.5587507522586564,
            0.5507126140003294,
            0.5158637316599726,
            0.463358502598609,
            0.4364810377781553,
            0.4554057601385197,
            0.49206834951428846,
            0.5069881394430568,
            0.4940944626651677,
            0.4794084877841306,
            0.48286120519597564,
            0.4991630711573942,
            0.5154864051976301,
            0.5265093662534079,
            0.5315848427499446,
            0.5336266171349366,
            0.5379532895668251,
            0.5462567623549865,
            0.5574433523516666,
            0.5720577168762397,
            0.5884805878281718,
            0.6011994496349105,
            0.5691043745027541,
            0.5439502736204297,
            0.5372723613969376,
            0.5482885425419857,
            0.5747619700273704,
            0.5956643255304307,
            0.576448882711783,
            0.519786328255309,
            0.47391503077648117,
            0.46939780526819913,
            0.48306848433289096,
            0.4790429254072764,
            0.45542737700641456,
            0.4372970848655028,
            0.4420559460519966,
            0.4649288748414777,
            0.49201179788368304,
            0.5158198907291166,
            0.5364928721556508,
            0.5566566564963795,
            0.5795404996699752,
            0.6037181922561183,
            0.6156507529311988,
            0.6011994496349105,
            0.5691043745027541,
            0.6077800728784944,
            0.5466952486825789,
            0.4967964427638948,
            0.48919936197735436,
            0.5169548195162452,
            0.5629129182965571,
            0.6027961939799411,
            0.6008721359605645,
            0.5511835821833126,
            0.495761523969524,
            0.46736962715841524,
            0.4525715923388255,
            0.4315404173434085,
            0.41055556876944216,
            0.4032023872820063,
            0.4127264353384646,
            0.435530963862229,
            0.4647612612529941,
            0.4946555240806856,
            0.5240074637519626,
            0.5524586987375938,
            0.58209505349126,
            0.6147689862165985,
            0.6328791521732973,
            0.6077800728784944,
            0.5466952486825789,
            0.5853271832631741,
            0.5301414081707394,
            0.4786686848585831,
            0.4673368718146639,
            0.4905256849742188,
            0.5299978888762862,
            0.5670616025517643,
            0.5720860185935266,
            0.5335602775404675,
            0.4817957226671952,
            0.4469292833422075,
            0.42527534176494813,
            0.4083770202892253,
            0.40126505367898546,
            0.4028089211947637,
            0.4079983919520376,
            0.42311946158299396,
            0.451664426452955,
            0.4838826281525204,
            0.5105598999020405,
            0.5299921795595904,
            0.548510599999211,
            0.5757070559706207,
            0.5993283594075406,
            0.5853271832631741,
            0.5301414081707394,
            0.5601003593594904,
            0.5329517357274873,
            0.49674558903665966,
            0.4785436519623544,
            0.4791923816950242,
            0.49194138534836596,
            0.5101239900308198,
            0.5141146749135403,
            0.48897417522245656,
            0.45064431788645654,
            0.4250743982724042,
            0.41697629133405545,
            0.41995427238892463,
            0.42914921639611703,
            0.43342176634508894,
            0.42934899222885503,
            0.4356919097843998,
            0.46340090481834945,
            0.4951068180958357,
            0.5114498667309884,
            0.512081865792444,
            0.5116779157359503,
            0.5270634518779471,
            0.5533320248189745,
            0.5601003593594904,
            0.5329517357274873,
            0.5542671596385682,
            0.5538089400848614,
            0.5330431533355049,
            0.5056846479224535,
            0.47836058621005456,
            0.45898857506612195,
            0.4555583834530601,
            0.4598751239869745,
            0.4530430681674907,
            0.4330099787863812,
            0.4182048894005972,
            0.4214668606449042,
            0.4395166007804828,
            0.46079655098006955,
            0.47025760652671994,
            0.4669937208689873,
            0.47343386594311376,
            0.5006340452100582,
            0.5261044562929101,
            0.5257240062215092,
            0.5039663425934162,
            0.4868677261203658,
            0.4966060903755267,
            0.5285120268247585,
            0.5542671596385682,
            0.5538089400848614,
            0.5586933809845538,
            0.5753270719381955,
            0.5675965096419938,
            0.5392511972640034,
            0.4949808572772374,
            0.4507595521734705,
            0.4296386475750726,
            0.4364785050713707,
            0.4480948327831062,
            0.44335621668755326,
            0.4315683612999702,
            0.43473512640997514,
            0.4575812611342963,
            0.4859401646995744,
            0.5029898247952467,
            0.509555236788184,
            0.5242033987199387,
            0.5501124512647324,
            0.5613132374755092,
            0.5366658006753242,
            0.49065677435546684,
            0.46203038496868226,
            0.47556383554086384,
            0.5186692136421895,
            0.5586933809845538,
            0.5753270719381955,
            0.5543834785058257,
            0.5866690572307853,
            0.5978485706576383,
            0.5836187524435692,
            0.5400321426688532,
            0.4831013628876609,
            0.44770521211368886,
            0.45120629091820014,
            0.4696835138254466,
            0.4706899790385802,
            0.45799884304835325,
            0.4599386316030667,
            0.485730495017657,
            0.5167399452639161,
            0.535434278278887,
            0.5473115813449854,
            0.566543118937217,
            0.5859020276505494,
            0.5782234027084813,
            0.5308656281056654,
            0.46828040687508043,
            0.43466380358753565,
            0.4525240697187949,
            0.5036804722814088,
            0.5543834785058257,
            0.5866690572307853,
            0.5410819047119357,
            0.5941509789525924,
            0.6285579620385011,
            0.6312428106443501,
            0.5941249534153842,
            0.5331335358738083,
            0.4869179447616062,
            0.4784758779118947,
            0.488854185706038,
            0.48754514736892884,
            0.4766560161165265,
            0.48222457102093247,
            0.5105162778283677,
            0.5396013038332012,
            0.5543938873205945,
            0.5641486730571083,
            0.5779624438773713,
            0.5838630354777599,
            0.5618811934114201,
            0.5091990637978197,
            0.44992400848411496,
            0.4190493463528253,
            0.43335230930462343,
            0.4815887426852624,
            0.5410819047119357,
            0.5941509789525924,
            0.5318679152084202,
            0.6041640131767214,
            0.6531742758290959,
            0.6577552256183432,
            0.6178842169915131,
            0.556480729424416,
            0.506484499257798,
            0.4846729452431421,
            0.4802175242051586,
            0.47547449971726774,
            0.4715274098537896,
            0.4824981896465676,
            0.5085520841937188,
            0.5326829130624663,
            0.5461977731486505,
            0.5550358734910248,
            0.559725254472766,
            0.5516466189441388,
            0.5260323327988242,
            0.48673194181531754,
            0.4466296633405282,
            0.42327802032924,
            0.4285506766429199,
            0.46642265547453765,
            0.5318679152084202,
            0.6041640131767214,
            0.5331708254981768,
            0.6098250764499792,
            0.6575178118496154,
            0.6508242876230148,
            0.6030018548682494,
            0.5466213819267685,
            0.5017683383663064,
            0.46995921564781606,
            0.44971880664316943,
            0.44366795222677746,
            0.453202770368208,
            0.474048289745795,
            0.49648346766462104,
            0.5130065594913967,
            0.525767264834882,
            0.5367718682912068,
            0.53837867281799,
            0.52635221153758,
            0.5069277681315055,
            0.48495140141022663,
            0.4626437940388718,
            0.44583288544927435,
            0.44381661752990964,
            0.47053686281167,
            0.5331708254981768,
            0.6098250764499792,
        ];
        fld.spatial = input.clone();
        assert_eq!(sim.n_pass, 4);
        for _ in 0..sim.n_pass {
            fld.binomial_filter_2_d(&mut wrkspace);
        }
        for (&v1, &v2) in fld.spatial.iter().zip(expected_out.spatial.iter()) {
            assert!((v1 - v2).abs() < E_TOL);
        }
    }
}
