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
        let ghost_start = self.with_ghost_dim.get_index(    Pos { row: 0, col: 1 });
        let ghost_range = ghost_start..ghost_start + size_x;
        let real_start = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y,
                col: 1,
            },
        );
        let real_range = real_start..real_start + size_x;
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
            }
        }
        // Copy top row into bottom ghost row
        let ghost_start = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y + 1,
                col: 1,
            },
        );
        let ghost_range = ghost_start..ghost_start + size_x;
        let real_start = self.with_ghost_dim.get_index(    Pos { row: 1, col: 1 });
        let real_range = real_start..real_start + size_x;
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
            }
        }
        // copy into left ghost columns from right real column
        let ghost_start = self.with_ghost_dim.get_index(    Pos { row: 1, col: 0 });
        let ghost_end = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1 + size_y,
                col: 0,
            },
        );
        let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
        let real_start = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1,
                col: size_x,
            },
        );
        let real_end = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1 + size_y,
                col: size_x,
            },
        );
        let real_range = (real_start..real_end).step_by(2 + size_x);
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
            }
        }

        // copy into right ghost columns from left real column
        let ghost_start = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1,
                col: size_x + 1,
            },
        );
        let ghost_end = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1 + size_y,
                col: size_x + 1,
            },
        );
        let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
        let real_start = self.with_ghost_dim.get_index(    Pos { row: 1, col: 1 });

        let real_end = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1 + size_y,
                col: 1,
            },
        );
        let real_range = (real_start..real_end).step_by(2 + size_x);
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
            }
        }
        // now do the corners
        // copy into top left from bottom right
        let btm_right = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y,
                col: size_x,
            },
        );
        unsafe { *fld.get_unchecked_mut(0) = *fld.get_unchecked(btm_right) }

        // copy into top right from bottom left
        let btm_left = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y,
                col: 1,
            },
        );
        unsafe { *fld.get_unchecked_mut(size_x + 1) = *fld.get_unchecked(btm_left) }

        // copy into bottom left from top right
        let ghost_btm_left = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y + 1,
                col: 0,
            },
        );
        let top_right = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1,
                col: size_x,
            },
        );
        unsafe { *fld.get_unchecked_mut(ghost_btm_left) = *fld.get_unchecked(top_right) }

        // Copy into bottom right from top left
        let ghost_btm_right = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y + 1,
                col: size_x + 1,
            },
        );
        let top_left = self.with_ghost_dim.get_index(    Pos { row: 1, col: 1 });
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
        let ghost_start = self.with_ghost_dim.get_index(    Pos { row: 0, col: 1 });
        let ghost_range = ghost_start..ghost_start + size_x;
        let real_start = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y,
                col: 1,
            },
        );
        let real_range = real_start..real_start + size_x;
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
            }
        }
        // deposit bottom ghost row into top real row
        let ghost_start = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y + 1,
                col: 1,
            },
        );
        let ghost_range = ghost_start..ghost_start + size_x;
        let real_start = self.with_ghost_dim.get_index(    Pos { row: 1, col: 1 });
        let real_range = real_start..real_start + size_x;
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
            }
        }
        // deposit left ghost columns into right real column
        let ghost_start = self.with_ghost_dim.get_index(    Pos { row: 1, col: 0 });
        let ghost_end = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1 + size_y,
                col: 0,
            },
        );
        let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
        let real_start = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1,
                col: size_x,
            },
        );
        let real_end = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1 + size_y,
                col: size_x,
            },
        );
        let real_range = (real_start..real_end).step_by(2 + size_x);
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
            }
        }

        // deposit right ghost columns into left real column
        let ghost_start = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1,
                col: size_x + 1,
            },
        );
        let ghost_end = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1 + size_y,
                col: size_x + 1,
            },
        );
        let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
        let real_start = self.with_ghost_dim.get_index(    Pos { row: 1, col: 1 });
        let real_end = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1 + size_y,
                col: 1,
            },
        );
        let real_range = (real_start..real_end).step_by(2 + size_x);
        for (ighost, ireal) in ghost_range.zip(real_range) {
            unsafe {
                *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
            }
        }
        // now do the corners
        // deposit top left into bottom right
        let btm_right = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y,
                col: size_x,
            },
        );
        unsafe { *fld.get_unchecked_mut(btm_right) += *fld.get_unchecked(0) }

        // depost top right into bottom left
        let btm_left = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y,
                col: 1,
            },
        );
        unsafe { *fld.get_unchecked_mut(btm_left) += *fld.get_unchecked(size_x + 1) }

        // depost bottom left into top right
        let ghost_btm_left = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y + 1,
                col: 0,
            },
        );
        let top_right = self.with_ghost_dim.get_index(
               
            Pos {
                row: 1,
                col: size_x,
            },
        );
        unsafe { *fld.get_unchecked_mut(top_right) += *fld.get_unchecked(ghost_btm_left) }

        // depost bottom right into top left
        let ghost_btm_right = self.with_ghost_dim.get_index(
               
            Pos {
                row: size_y + 1,
                col: size_x + 1,
            },
        );
        let top_left = self.with_ghost_dim.get_index(    Pos { row: 1, col: 1 });
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
    use crate::build_test_sim;
    #[test]
    fn ghost_deposit() {
        // made an example using google sheets
        let mut input: Vec<Float> = vec![
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
            assert_eq!(v, expected_v);
        }
    }

    #[test]
    fn ghosts_update() {
        // made an example using google sheets
        let mut input: Vec<Float> = vec![
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
            assert_eq!(v, expected_v);
        }
    }
}
