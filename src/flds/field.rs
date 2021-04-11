use crate::{Float, Sim};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

pub struct Pos {
    pub row: usize,
    pub col: usize,
}

pub struct FieldDim {
    size_x: usize,
    size_y: usize,
}

pub struct Field {
    pub spatial: Vec<Float>,
    pub spectral: Vec<Complex<Float>>,
    with_ghost_dim: FieldDim,
    no_ghost_dim: FieldDim,
}

impl FieldDim {
    pub fn get_index(&self, pos: Pos) -> usize {
        // Convenience method to get a position in the array.
        // Slightly complicated because
        // Using a 1d vec to represent 2D array for speed.
        // Here is the layout if it were a 2d array,
        // with the 1D vec position in []
        // ----------------------------------
        // |   [0]    |   [1]    |   [2]    |
        // |  row: 0  |  row: 0  |  row: 0  |
        // |  col: 0  |  col: 1  |  col: 2  |
        // |          |          |          |
        // ----------------------------------
        // |   [3]    |   [4]    |   [5]    |
        // |  row: 1  |  row: 1  |  row: 1  |
        // |  col: 0  |  col: 1  |  col: 2  |
        // |          |          |          |
        // ----------------------------------
        // |   [6]    |   [7]    |   [8]    |
        // |  row: 2  |  row: 2  |  row: 2  |
        // |  col: 1  |  col: 1  |  col: 2  |
        // |          |          |          |
        // ----------------------------------

        if !cfg!(feature = "unchecked") {
            assert!(pos.col < self.size_x);
            assert!(pos.row < self.size_y);
        }

        pos.row * self.size_x + pos.col
    }
}
impl Field {
    pub fn new(sim: &Sim) -> Field {
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
        let in_vec = &mut self.spatial;
        let size_x = self.no_ghost_dim.size_x;
        let size_y = self.no_ghost_dim.size_y;
        // wrkspace should be same size as fld
        if !cfg!(feature = "unchecked") {
            assert!(in_vec.len() == field_buf.spatial.len());
            assert!(in_vec.len() == (size_x + 2) * (size_y + 2));
        }

        let weights: [Float; 3] = [0.25, 0.5, 0.25];
        // account for ghost zones
        // FIRST FILTER IN X-DIRECTION
        {
            // Scoping here to satisfy borrow checker
            let wrkspace = &mut field_buf.spatial;
            for i in ((size_x + 2)..(size_y + 1) * (size_x + 2)).step_by(size_x + 2) {
                for j in 1..size_x + 1 {
                    wrkspace[i] = weights
                        .iter()
                        .zip(&in_vec[i + j - 1..i + j + 1])
                        .map(|(&w, &f)| w * f)
                        .sum::<Float>();
                }
            }
        }

        field_buf.update_ghosts();

        let wrkspace = &field_buf.spatial;
        // NOW FILTER IN Y-DIRECTION AND PUT VALS IN in_vec
        for i in ((size_x + 2)..(size_y + 1) * (size_x + 2)).step_by(size_x + 2) {
            for j in 1..size_x + 1 {
                in_vec[i] = weights
                    .iter()
                    .zip(
                        wrkspace[i + j - (size_x + 2)..i + j + (size_x + 2)]
                            .iter()
                            .step_by(size_x + 2),
                    )
                    .map(|(&w, &f)| w * f)
                    .sum::<Float>();
            }
        }

        self.update_ghosts();
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
        let fld = Field::new(&sim);
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
        let fld = Field::new(&sim);
        // so this function counts the two ghost zones.
        //
        //
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

        let mut test_fld = Field::new(&sim);
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

        let mut test_fld = Field::new(&sim);

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
        let mut fld = Field::new(&sim);
        fld.spatial = input.clone();
        assert_eq!(fld.spatial.len(), expected_output.len());
        assert_eq!(fld.spatial.len(), (sim.size_x + 2) * (sim.size_y + 2));
        fld.deposit_ghosts();
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
        let mut fld = Field::new(&build_test_sim());
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
}
