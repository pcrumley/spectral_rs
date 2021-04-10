use crate::flds::ghosts::update_ghosts;
use crate::{Float, Sim};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

pub struct Pos {
    pub row: usize,
    pub col: usize,
}

pub struct Field {
    pub spatial: Vec<Float>,
    pub spectral: Vec<Complex<Float>>,
}

impl Field {
    #[inline(always)]
    pub fn copy_to_spectral(&mut self, sim: &Sim) -> () {
        // assert stuff about ghost zones etc.
        let spatial = &self.spatial;
        let spectral = &mut self.spectral;
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
        update_ghosts(sim, spatial);
    }
}
