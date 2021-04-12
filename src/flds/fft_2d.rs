use crate::flds::field::Field;
use crate::{Float, Sim};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;

pub struct Fft2D {
    size_x: usize,
    size_y: usize,
    fft_x: std::sync::Arc<dyn rustfft::Fft<Float>>,
    ifft_x: std::sync::Arc<dyn rustfft::Fft<Float>>,
    fft_y: std::sync::Arc<dyn rustfft::Fft<Float>>,
    ifft_y: std::sync::Arc<dyn rustfft::Fft<Float>>,
    xscratch: Vec<Complex<Float>>,
    yscratch: Vec<Complex<Float>>,
}

impl Fft2D {
    pub fn new(sim: &Sim) -> Fft2D {
        let mut planner = FftPlanner::new();
        let fft_x = planner.plan_fft_forward(sim.size_x);
        let ifft_x = planner.plan_fft_inverse(sim.size_x);
        let fft_y = planner.plan_fft_forward(sim.size_y);
        let ifft_y = planner.plan_fft_inverse(sim.size_y);
        let xscratch = vec![Complex::zero(); fft_x.get_outofplace_scratch_len()];
        let yscratch = vec![Complex::zero(); fft_y.get_outofplace_scratch_len()];

        Fft2D {
            size_x: sim.size_x,
            size_y: sim.size_y,
            fft_x,
            ifft_x,
            fft_y,
            ifft_y,
            xscratch,
            yscratch,
        }
    }

    pub fn fft(&mut self, fld: &mut Field, wrkspace: &mut Field) {
        if !cfg!(feature = "unchecked") {
            assert_eq!(self.size_x, fld.no_ghost_dim.size_x);
            assert_eq!(self.size_y, fld.no_ghost_dim.size_y);
            assert_eq!(self.size_x, wrkspace.no_ghost_dim.size_x);
            assert_eq!(self.size_y, wrkspace.no_ghost_dim.size_y);
        }

        self.fft_x.process_outofplace_with_scratch(
            &mut fld.spectral,
            &mut wrkspace.spectral,
            &mut self.xscratch,
        );

        wrkspace.transpose_spect_out_of_place(fld);
        self.fft_y.process_outofplace_with_scratch(
            &mut fld.spectral,
            &mut wrkspace.spectral,
            &mut self.yscratch,
        );

        wrkspace.transpose_spect_out_of_place(fld);
    }

    pub fn inv_fft(&mut self, fld: &mut Field, wrkspace: &mut Field) {
        if !cfg!(feature = "unchecked") {
            assert_eq!(self.size_x, fld.no_ghost_dim.size_x);
            assert_eq!(self.size_y, fld.no_ghost_dim.size_y);
            assert_eq!(self.size_x, wrkspace.no_ghost_dim.size_x);
            assert_eq!(self.size_y, wrkspace.no_ghost_dim.size_y);
        }


        self.ifft_x.process_outofplace_with_scratch(
            &mut fld.spectral,
            &mut wrkspace.spectral,
            &mut self.xscratch,
        );

        wrkspace.transpose_spect_out_of_place(fld);
        self.ifft_y.process_outofplace_with_scratch(
            &mut fld.spectral,
            &mut wrkspace.spectral,
            &mut self.yscratch,
        );

        wrkspace.transpose_spect_out_of_place(fld);
    }
}
