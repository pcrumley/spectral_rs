use crate::{Float, Sim};
const PI: Float = std::f64::consts::PI as Float;

use itertools::izip;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;
pub mod field;
pub mod ghosts;
use crate::flds::field::Field;
use crate::flds::ghosts::update_ghosts;

pub struct WaveVectors {
    k_x: Vec<Float>,
    k_y: Vec<Float>,
    k_norm: Vec<Float>,
}

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
    fft_x: std::sync::Arc<dyn rustfft::Fft<Float>>,
    ifft_x: std::sync::Arc<dyn rustfft::Fft<Float>>,
    fft_y: std::sync::Arc<dyn rustfft::Fft<Float>>,
    ifft_y: std::sync::Arc<dyn rustfft::Fft<Float>>,
    fft_x_buf: Vec<Complex<Float>>,
    fft_y_buf: Vec<Complex<Float>>,
    real_wrkspace_ghosts: Vec<Float>,
    cmp_wrkspace: Vec<Complex<Float>>,
    pub dsty: Field,
}

#[inline(always)]
fn binomial_filter_2_d(sim: &Sim, in_vec: &mut Vec<Float>, wrkspace: &mut Vec<Float>) {
    // wrkspace should be same size as fld
    if !cfg!(feature = "unchecked") {
        assert!(in_vec.len() == wrkspace.len());
        assert!(in_vec.len() == (sim.size_x + 2) * (sim.size_y + 2));
    }

    let weights: [Float; 3] = [0.25, 0.5, 0.25];
    // account for ghost zones
    // FIRST FILTER IN X-DIRECTION

    for _ in 0..sim.n_pass {
        for i in ((sim.size_x + 2)..(sim.size_y + 1) * (sim.size_x + 2)).step_by(sim.size_x + 2) {
            for j in 1..sim.size_x + 1 {
                wrkspace[i] = weights
                    .iter()
                    .zip(&in_vec[i + j - 1..i + j + 1])
                    .map(|(&w, &f)| w * f)
                    .sum::<Float>();
            }
        }

        update_ghosts(&sim, wrkspace);

        // NOW FILTER IN Y-DIRECTION AND PUT VALS IN in_vec
        for i in ((sim.size_x + 2)..(sim.size_y + 1) * (sim.size_x + 2)).step_by(sim.size_x + 2) {
            for j in 1..sim.size_x + 1 {
                in_vec[i] = weights
                    .iter()
                    .zip(
                        wrkspace[i + j - (sim.size_x + 2)..i + j + (sim.size_x + 2)]
                            .iter()
                            .step_by(sim.size_x + 2),
                    )
                    .map(|(&w, &f)| w * f)
                    .sum::<Float>();
            }
        }

        update_ghosts(&sim, in_vec);
    }
}

impl Flds {
    pub fn new(sim: &Sim) -> Flds {
        // let Bnorm = 0Float;
        let mut planner = FftPlanner::new();
        let mut inv_planner = FftPlanner::new();

        let fft_x = planner.plan_fft_forward(sim.size_x);
        let ifft_x = inv_planner.plan_fft_inverse(sim.size_x);
        let fft_y = planner.plan_fft_forward(sim.size_y);
        let ifft_y = planner.plan_fft_inverse(sim.size_y);
        let xscratch = vec![Complex::zero(); fft_x.get_outofplace_scratch_len()];
        let yscratch = vec![Complex::zero(); fft_y.get_outofplace_scratch_len()];

        let mut f = Flds {
            e_x: Field {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            e_y: Field {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            e_z: Field {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            b_x: Field {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            b_y: Field {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            b_z: Field {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            j_x: Field {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            j_y: Field {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            j_z: Field {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            dsty: Field {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            k_x: vec![0.0; sim.size_x * sim.size_y],
            k_y: vec![0.0; sim.size_y * sim.size_x],
            k_norm: vec![0.0; sim.size_y * sim.size_x],
            fft_x,
            ifft_x,
            fft_y,
            ifft_y,
            fft_x_buf: xscratch,
            fft_y_buf: yscratch,
            real_wrkspace_ghosts: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            cmp_wrkspace: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_x_wrk: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_y_wrk: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_z_wrk: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
        };

        // Build the k basis of FFT
        for i in 0..sim.size_y {
            for j in 0..sim.size_x {
                let ind = i * sim.size_x + j;
                // FIRST DO K_X
                f.k_x[ind] = j as Float;
                if j >= sim.size_x / 2 + 1 {
                    f.k_x[ind] -= sim.size_x as Float;
                }
                f.k_x[ind] *= 2.0 * PI / (sim.size_x as Float);
                // NOW DO K_Y
                f.k_y[ind] = i as Float;
                if i >= sim.size_y / 2 + 1 {
                    f.k_y[ind] -= sim.size_y as Float;
                }
                f.k_y[ind] *= 2.0 * PI / (sim.size_y as Float);
            }
        }
        // Make the norm:
        for (norm, kx, ky) in izip!(&mut f.k_norm, &f.k_x, &f.k_y) {
            *norm = 1. / (kx * kx + ky * ky);
        }
        /* I HAVE NO IDEA WHY THIS IS HERE??? WHAT DOES IT MEAN??
         * IGNORE BUT LEAVING IN CASE I REMEMBER
        if false {
            for i in 0..(sim.size_y + 2) {
                //let tmp_b = 2.0 * Bnorm * (sim.size_y/2 - (i - 1))/(sim.size_y as Float);
                let tmp_b = 0.0;
                for j in 0..(sim.size_x + 2) {
                    if i > (sim.size_y) / 6 && i < 2 * sim.size_y / 6 {
                        f.e_y[i * (sim.size_x + 2) + j] = -0.9 * tmp_b;
                        f.b_z[i * (sim.size_x + 2) + j] = tmp_b;
                    }
                    if i > 2 * (sim.size_y) / 3 && i < 5 * sim.size_y / 6 {
                        f.e_y[i * (sim.size_x + 2) + j] = 0.9 * tmp_b;
                        f.b_z[i * (sim.size_x + 2) + j] = -tmp_b;
                    }
                }
            }
        }
        */
        f
    }

    pub fn transpose(sim: &Sim, in_fld: &Vec<Complex<Float>>, out_fld: &mut Vec<Complex<Float>>) {
        // check to make sure the two vecs are the same size
        if !cfg!(feature = "unchecked") {
            assert!(in_fld.len() == out_fld.len());
            assert!(sim.size_y * sim.size_x == in_fld.len());
        }
        for i in 0..sim.size_y {
            for j in 0..sim.size_x {
                unsafe {
                    // If you don't trust this unsafe section,
                    // run the code with the checked feature
                    // len(out_fld) == len(in_fld)
                    // && size_y * size_x == len(out_fld)
                    *out_fld.get_unchecked_mut(i * sim.size_x + j) =
                        *in_fld.get_unchecked(j * sim.size_y + i);
                }
                // bounds checked version
                // out_fld[i * sim.size_x + j] = in_fld[j * sim.size_y + i];
            }
        }
    }

    fn fft2d(
        fft_x: std::sync::Arc<dyn rustfft::Fft<Float>>,
        fft_y: std::sync::Arc<dyn rustfft::Fft<Float>>,
        sim: &Sim,
        fld: &mut Vec<Complex<Float>>,
        wrk_space: &mut Vec<Complex<Float>>,
        xscratch: &mut Vec<Complex<Float>>,
        yscratch: &mut Vec<Complex<Float>>,
    ) {
        for iy in (0..sim.size_y * sim.size_x).step_by(sim.size_x) {
            fft_x.process_outofplace_with_scratch(
                &mut fld[iy..iy + sim.size_x],
                &mut wrk_space[iy..iy + sim.size_x],
                xscratch,
            );
        }
        Flds::transpose(sim, wrk_space, fld);
        for iy in (0..sim.size_x * sim.size_y).step_by(sim.size_y) {
            fft_y.process_outofplace_with_scratch(
                &mut fld[iy..iy + sim.size_y],
                &mut wrk_space[iy..iy + sim.size_y],
                yscratch,
            );
        }
        Flds::transpose(sim, wrk_space, fld);
    }

    fn copy_spatial_to_spectral(&mut self, sim: &Sim) {
        // copy j_x, j_y, j_z, dsty into complex vector
        self.j_x.copy_to_spectral(sim);
        self.j_y.copy_to_spectral(sim);
        self.j_z.copy_to_spectral(sim);
        // self.b_x.copy_to_spectral(sim);
        // self.b_y.copy_to_spectral(sim);
        // self.b_z.copy_to_spectral(sim);
        self.e_x.copy_to_spectral(sim);
        self.e_y.copy_to_spectral(sim);
        self.e_z.copy_to_spectral(sim);
        self.dsty.copy_to_spectral(sim);
        // need to normalize self.dsty.spectral by 1/ sim.dens;
        let norm = 1.0 / (sim.dens as Float);
        for v in self.dsty.spectral.iter_mut() {
            v.re *= norm;
            v.im *= norm;
        }
    }

    pub fn update(&mut self, sim: &Sim) {
        // Filter currents and density fields
        binomial_filter_2_d(sim, &mut self.j_x.spatial, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.j_y.spatial, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.j_z.spatial, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.dsty.spatial, &mut self.real_wrkspace_ghosts);

        // copy the flds to the complex arrays to perform ffts;
        self.copy_spatial_to_spectral(sim);

        // Take fft of currents
        for current in &mut [
            &mut self.j_x.spectral,
            &mut self.j_y.spectral,
            &mut self.j_z.spectral,
            &mut self.dsty.spectral,
        ] {
            Flds::fft2d(
                self.fft_x.clone(),
                self.fft_y.clone(),
                sim,
                current,
                &mut self.cmp_wrkspace,
                &mut self.fft_x_buf,
                &mut self.fft_y_buf,
            );
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
        for e_fld in &mut [
            &mut self.e_x.spectral,
            &mut self.e_y.spectral,
            &mut self.e_z.spectral,
        ] {
            Flds::fft2d(
                self.fft_x.clone(),
                self.fft_y.clone(),
                sim,
                e_fld,
                &mut self.cmp_wrkspace,
                &mut self.fft_x_buf,
                &mut self.fft_y_buf,
            );
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

        // gonna do some unsafe code so need to have assert! here
        if !cfg!(feature = "unchecked") {
            let tot_cells = sim.size_x * sim.size_y;
            for b_fld in &[&self.b_x_wrk, &self.b_y_wrk, &self.b_z_wrk] {
                assert_eq!(tot_cells, b_fld.len());
            }
            for fld in &[&self.e_x, &self.e_y, &self.e_z] {
                assert_eq!(tot_cells, fld.spectral.len());
            }
        }
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
            for i in (ny_col_start..ny_end).step_by(sim.size_x) {
                unsafe {
                    // safe because the iterator returns values between 0 and
                    // sim.size_x * sim.size_y exclusive and size of all these fields
                    // was asserted to be sim_size_x * sim.size_y
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
            &mut self.b_x.spectral,
            &mut self.b_y.spectral,
            &mut self.b_z.spectral,
            &mut self.e_x.spectral,
            &mut self.e_y.spectral,
            &mut self.e_z.spectral,
        ] {
            Flds::fft2d(
                self.ifft_x.clone(),
                self.ifft_y.clone(),
                sim,
                fld,
                &mut self.cmp_wrkspace,
                &mut self.fft_x_buf,
                &mut self.fft_y_buf,
            );
        }
        // copy that fft to real array
        let mut im_sum = 0.0;
        for fld in &mut [
            &mut self.b_x,
            &mut self.b_y,
            &mut self.b_z,
            &mut self.e_x,
            &mut self.e_y,
            &mut self.e_z,
        ] {
            im_sum += fld.spectral.iter().map(|o| o.im.abs()).sum::<Float>()
                / (fld.spectral.len() as Float);
            fld.copy_to_spatial(&sim);
        }
        println!("{}", im_sum);
    }
}
