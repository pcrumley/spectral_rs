pub mod fft_2d;
pub mod field;
pub mod wave_num;

use crate::flds::fft_2d::Fft2D;
use crate::flds::field::Field;
use crate::flds::wave_num::WaveNumbers;
use crate::{Float, Sim};

use itertools::izip;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

pub struct Flds {
    // The struct that holds all the fields.
    // First off is all the regular fields.
    pub e_x: Field,
    pub e_y: Field,
    pub e_z: Field,
    pub b_x: Field,
    pub b_y: Field,
    pub b_z: Field,
    pub j_x: Field,
    pub j_y: Field,
    pub j_z: Field,
    k_x: Vec<Float>,
    k_y: Vec<Float>,
    k_norm: Vec<Float>,
    b_x_wrk: Vec<Complex<Float>>,
    b_y_wrk: Vec<Complex<Float>>,
    b_z_wrk: Vec<Complex<Float>>,
    wrkspace: Field,
    pub dsty: Field,
    pub fft_2d: Fft2D,
}

impl Flds {
    pub fn new(sim: &Sim) -> Flds {
        // let Bnorm = 0Float;
        let wave_nums = WaveNumbers::new(sim);

        Flds {
            e_x: Field::new(sim, "Ex".into()),
            e_y: Field::new(sim, "Ey".into()),
            e_z: Field::new(sim, "Ez".into()),
            b_x: Field::new(sim, "Bx".into()),
            b_y: Field::new(sim, "By".into()),
            b_z: Field::new(sim, "Bz".into()),
            j_x: Field::new(sim, "Jx".into()),
            j_y: Field::new(sim, "Jy".into()),
            j_z: Field::new(sim, "Jz".into()),
            dsty: Field::new(sim, "Rho".into()),
            k_x: wave_nums.k_x,
            k_y: wave_nums.k_y,
            k_norm: wave_nums.k_norm,
            b_x_wrk: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_y_wrk: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_z_wrk: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            fft_2d: Fft2D::new(sim),
            wrkspace: Field::default(sim),
        }
    }
    fn copy_spatial_to_spectral(&mut self, sim: &Sim) {
        // copy j_x, j_y, j_z, dsty into complex vector
        self.j_x.copy_to_spectral();
        self.j_y.copy_to_spectral();
        self.j_z.copy_to_spectral();
        self.e_x.copy_to_spectral();
        self.e_y.copy_to_spectral();
        self.e_z.copy_to_spectral();
        self.dsty.copy_to_spectral();
        // need to normalize self.dsty.spectral by 1/ sim.dens;
        let norm = 1.0 / (sim.dens as Float);
        for v in self.dsty.spectral.iter_mut() {
            v.re *= norm;
            v.im *= norm;
        }
    }

    pub fn update(&mut self, sim: &Sim) {
        // Filter currents and density fields
        for _ in 0..sim.n_pass {
            for fld in &mut [&mut self.j_x, &mut self.j_y, &mut self.j_z, &mut self.dsty] {
                fld.binomial_filter_2_d(&mut self.wrkspace);
            }
        }
        // copy the flds to the complex arrays to perform ffts;
        self.copy_spatial_to_spectral(sim);

        // Take fft of currents
        for current in &mut [&mut self.j_x, &mut self.j_y, &mut self.j_z, &mut self.dsty] {
            self.fft_2d.fft(current);
        }

        // clean k=0 contributions from currents
        self.j_x.spectral[0] = Complex::zero();
        self.j_y.spectral[0] = Complex::zero();
        // No j_z???  why?

        // wrkspace contains the b field on the minus
        // half step spectral contains it as -1. We need to
        // advance spectral by 1/2 time step
        for (b_prev, b_prev_half) in &mut [
            (&mut self.b_x.spectral, &self.b_x_wrk),
            (&mut self.b_y.spectral, &self.b_y_wrk),
            (&mut self.b_z.spectral, &self.b_z_wrk),
        ] {
            for (vm1, vmhalf) in b_prev.iter_mut().zip(b_prev_half.iter()) {
                *vm1 = *vmhalf;
            }
        }
        // Take fft of electric fields
        for e_fld in &mut [&mut self.e_x, &mut self.e_y, &mut self.e_z] {
            self.fft_2d.fft(e_fld);
        }

        // push on electric field
        let ckc = sim.dt / sim.dens as Float;
        let cdt = sim.dt * sim.c as Float;

        for (e_x, k_y, b_z, j_x) in izip!(
            &mut self.e_x.spectral,
            &self.k_y,
            &self.b_z.spectral,
            &self.j_x.spectral,
        ) {
            *e_x += Complex::new(0.0, cdt) * k_y * b_z - ckc * j_x;
        }

        for (e_y, k_x, b_z, j_y) in izip!(
            &mut self.e_y.spectral,
            &self.k_x,
            &self.b_z.spectral,
            &self.j_y.spectral,
        ) {
            *e_y += Complex::new(0.0, -cdt) * k_x * b_z - ckc * j_y;
        }

        for (e_z, k_x, b_y, k_y, b_x, j_z) in izip!(
            &mut self.e_z.spectral,
            &self.k_x,
            &self.b_y.spectral,
            &self.k_y,
            &self.b_x.spectral,
            &self.j_z.spectral
        ) {
            *e_z += Complex::new(0.0, cdt) * (k_x * b_y - k_y * b_x);
            *e_z -= ckc * j_z;
        }

        // save k=0 components because impossible to apply correction
        // (division by 0)
        let ex0 = self.e_x.spectral[0];
        let ey0 = self.e_y.spectral[0];

        // Boris correction:
        // some helper vars we initialize. I don't know if it helps to do
        // this outside of the loop in rust or not.

        let mut tmp: Complex<Float>;

        for (e_x, e_y, k_x, k_y, dsty, norm) in izip!(
            &mut self.e_x.spectral,
            &mut self.e_y.spectral,
            &self.k_x,
            &self.k_y,
            &self.dsty.spectral,
            &self.k_norm
        ) {
            tmp = (k_x * *e_x + k_y * *e_y + Complex::new(0., 1.) * dsty) * norm;
            *e_x -= tmp * k_x;
            *e_y -= tmp * k_y;
        }

        // restablish uncorrected longitudinal electric field...
        // needed to conserve the contribution from motional electric field
        // if upstream moving plasma carries magnetic frozen in field
        self.e_x.spectral[0] = ex0;
        self.e_y.spectral[0] = ey0;

        // push on wrkspace magnetic field with updated electric field. advance it to t + 1/2
        // timestep
        for (b_x, k_y, e_z) in izip!(&mut self.b_x_wrk, &self.k_y, &self.e_z.spectral) {
            *b_x -= Complex::new(0.0, cdt) * k_y * e_z;
        }
        for (b_y, k_x, e_z) in izip!(&mut self.b_y_wrk, &self.k_x, &self.e_z.spectral) {
            *b_y += Complex::new(0.0, cdt) * k_x * e_z;
        }
        for (b_z, k_x, k_y, e_x, e_y) in izip!(
            &mut self.b_z_wrk,
            &self.k_x,
            &self.k_y,
            &self.e_x.spectral,
            &self.e_y.spectral
        ) {
            *b_z += Complex::new(0.0, cdt) * (k_y * e_x - k_x * e_y);
        }

        // Filter out the Nyquist frequency component, because it can cause
        // spurious imaginary quantities to show up in real space.

        let ny_col_start = sim.size_x / 2;
        let ny_end = sim.size_x * sim.size_y;
        for fld in &mut [
            &mut self.b_x_wrk,
            &mut self.b_y_wrk,
            &mut self.b_z_wrk,
            &mut self.e_x.spectral,
            &mut self.e_y.spectral,
            &mut self.e_z.spectral,
        ] {
            // gonna do some unsafe code so need to have assert! here
            if !cfg!(feature = "unchecked") {
                assert!(ny_end == fld.len());
            }

            for i in (ny_col_start..ny_end).step_by(sim.size_x) {
                unsafe {
                    // safe because the assert above
                    *fld.get_unchecked_mut(i) = Complex::zero();
                }
            }
        }
        // now filter out nyquist row
        let ny_row_start = sim.size_y * sim.size_x / 2;
        let ny_row_end = ny_row_start + sim.size_x;
        for fld in &mut [
            &mut self.b_x_wrk,
            &mut self.b_y_wrk,
            &mut self.b_z_wrk,
            &mut self.e_x.spectral,
            &mut self.e_y.spectral,
            &mut self.e_z.spectral,
        ] {
            for v in fld[ny_row_start..ny_row_end].iter_mut() {
                *v = Complex::zero();
            }
        }

        // advance B.spectral by half timestep using averaging
        // so that it is at time t+1
        for (b_minus_half, b_plus_half) in &mut [
            (&mut self.b_x.spectral, &self.b_x_wrk),
            (&mut self.b_y.spectral, &self.b_y_wrk),
            (&mut self.b_z.spectral, &self.b_z_wrk),
        ] {
            for (v1, v2) in b_minus_half.iter_mut().zip(b_plus_half.iter()) {
                *v1 = 0.5 * (*v1 + *v2);
            }
        }

        // take the inverse fft to return to spatial domain
        for fld in &mut [
            &mut self.b_x,
            &mut self.b_y,
            &mut self.b_z,
            &mut self.e_x,
            &mut self.e_y,
            &mut self.e_z,
        ] {
            self.fft_2d.inv_fft(fld);
        }
        // copy that fft to real array
        // let mut im_sum = 0.0;
        for fld in &mut [
            &mut self.b_x,
            &mut self.b_y,
            &mut self.b_z,
            &mut self.e_x,
            &mut self.e_y,
            &mut self.e_z,
        ] {
            /* im_sum += fld.spectral.iter().map(|o| o.im.abs()).sum::<Float>()
            / (fld.spectral.len() as Float);
            */
            fld.copy_to_spatial(&sim);
        }
        // println!("{}", im_sum);
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::build_test_sim;
    #[test]
    fn flds_init() {
        // checks that all fields are intialized to the correct
        // size and to zero
        let expected_spatial_val: Vec<Float> = vec![0.; (24 + 2) * (12 + 2)];
        let expected_complex_val: Vec<Complex<Float>> = vec![Complex::zero(); 24 * 12];

        let sim = build_test_sim();
        let flds = Flds::new(&sim);
        for fld in &[
            flds.j_x, flds.j_y, flds.j_z, flds.dsty, flds.b_x, flds.b_y, flds.b_z, flds.e_x,
            flds.e_y, flds.e_z,
        ] {
            assert_eq!(fld.spatial.len(), expected_spatial_val.len());
            assert_eq!(fld.spatial.len(), (sim.size_x + 2) * (sim.size_y + 2));
            assert_eq!(fld.spectral.len(), expected_complex_val.len());
            assert_eq!(fld.spectral.len(), sim.size_x * sim.size_y);
            for (v, expected_v) in fld.spatial.iter().zip(expected_spatial_val.iter()) {
                assert_eq!(v, expected_v);
            }
            for (v, expected_v) in fld.spectral.iter().zip(expected_complex_val.iter()) {
                assert_eq!(v, expected_v);
            }
        }
    }
}
