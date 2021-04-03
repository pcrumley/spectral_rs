use rand::prelude::*;
use rand_distr::Standard;
use rand_distr::StandardNormal;
use serde::Deserialize;
use std::fs;

use anyhow::{Context, Result};
use itertools::izip;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

// We use a type alias for f64/f32 to easily support
// double and single precision.
#[cfg(feature = "dprec")]
type Float = f64;

#[cfg(not(feature = "dprec"))]
type Float = f32;

const PI: Float = std::f64::consts::PI as Float;

#[derive(Deserialize)]
pub struct Config {
    pub params: Params,
    pub setup: Setup,
    pub output: Output,
}

#[derive(Deserialize)]
pub struct Setup {
    pub t_final: u32,
}

#[derive(Deserialize)]
pub struct Output {
    pub _track_prtls: bool,
    pub write_output: bool,
    pub _track_interval: u32,
    pub output_interval: u32,
    pub stride: usize,
}

#[derive(Deserialize)]
pub struct Params {
    pub size_x: usize,
    pub size_y: usize,
    pub delta: usize,
    pub dt: Float,
    pub c: Float,
    pub dens: u32,
    pub gamma_inj: Float,
    pub n_pass: u8,
}

impl Config {
    pub fn new() -> Result<Config> {
        let contents =
            fs::read_to_string("config.toml").context("Could not open the config.toml file")?;
        toml::from_str(&contents).with_context(|| "Could not parse Config file")
    }
}

pub fn run(cfg: Config) -> Result<()> {
    let sim = Sim::new(&cfg);
    let mut prtls = Vec::<Prtl>::new();
    // Add ions to prtls list
    println!("initialzing  prtls");
    prtls.push(Prtl::new(&sim, 1.0, 1.0, 1E-3));
    // Add lecs to prtls list
    prtls.push(Prtl::new(&sim, -1.0, 1.0, 1E-3));

    let mut flds = Flds::new(&sim);
    /* TODO add better particle tracking
    let mut x_track =
        Vec::<f32>::with_capacity((sim.t_final / cfg.output.output_interval) as usize);
    let mut y_track =
        Vec::<f32>::with_capacity((sim.t_final / cfg.output.output_interval) as usize);
    let mut gam_track =
        Vec::<f32>::with_capacity((sim.t_final / cfg.output.output_interval) as usize);
    */
    for t in 0..=sim.t_final {
        if cfg.output.write_output {
            if t % cfg.output.output_interval == 0 {
                let output_prefix = format!("output/dat_{:05}", t / cfg.output.output_interval);
                fs::create_dir_all(&output_prefix).context("Unable to create output directory")?;
                println!("saving prtls");
                let x: Vec<Float> = prtls[0]
                    .ix
                    .iter()
                    .zip(prtls[0].dx.iter())
                    .step_by(cfg.output.stride)
                    .map(|(&ix, &dx)| ix as Float + dx)
                    .collect();

                npy::to_file(format!("{}/x.npy", output_prefix), x)
                    .context("Could not save x data to file")?;
                let y: Vec<Float> = prtls[0]
                    .iy
                    .iter()
                    .zip(prtls[0].dy.iter())
                    .step_by(cfg.output.stride)
                    .map(|(&iy, &dy)| iy as Float + dy)
                    .collect();
                npy::to_file(format!("{}/y.npy", output_prefix), y)
                    .context("Could not save y prtl data")?;

                let u: Vec<_> = prtls[0]
                    .px
                    .iter()
                    .step_by(cfg.output.stride)
                    .map(|&x| x / sim.c)
                    .collect();

                npy::to_file(format!("{}/u.npy", output_prefix), u)
                    .context("Could not save u data to file")?;

                let gam: Vec<_> = prtls[0]
                    .psa
                    .iter()
                    .step_by(cfg.output.stride)
                    .map(|&psa| psa)
                    .collect();

                npy::to_file(format!("{}/gam.npy", output_prefix), gam)
                    .context("Error saving writing lorentz factor to file")?;
            }
        }
        /* TODO Add better way of tracking particles
        if cfg.output.track_prtls {
            if t % cfg.output.track_interval == 0 {
                for (ix, iy, dx, dy, track, psa) in izip!(
                    &prtls[0].ix,
                    &prtls[0].iy,
                    &prtls[0].dx,
                    &prtls[0].dy,
                    &prtls[0].track,
                    &prtls[0].psa
                ) {
                    if *track {
                        x_track.push((*ix as f32 + *dx) / sim.c);
                        y_track.push((*iy as f32 + *dy) / sim.c);
                        gam_track.push(*psa);
                    }
                }
            }
        }
        */
        // Zero out currents
        println!("{}", t);
        for (jx, jy, jz, dsty) in izip!(&mut flds.j_x, &mut flds.j_y, &mut flds.j_z, &mut flds.dsty)
        {
            *jx = 0.;
            *jy = 0.;
            *jz = 0.;
            *dsty = 0.;
        }
        println!("moving & dep prtl");
        // deposit currents
        for prtl in prtls.iter_mut() {
            sim.move_and_deposit(prtl, &mut flds);
        }

        // solve field
        println!("solving fields");
        flds.update(&sim);

        // push prtls
        println!("pushing prtl");
        for prtl in prtls.iter_mut() {
            prtl.boris_push(&sim, &flds);
        }

        //calc Density
        for prtl in prtls.iter_mut() {
            sim.calc_density(prtl, &mut flds);
        }

        // let sim.t = t;
    }
    /*
    if cfg.output.track_prtls {
        fs::create_dir_all("output/trckd_prtl/")?;
        npy::to_file("output/trckd_prtl/x.npy", x_track)?;
        npy::to_file("output/trckd_prtl/y.npy", y_track)?;
        npy::to_file("output/trckd_prtl/psa.npy", gam_track)?;
    }
    */
    Ok(())
}

fn binomial_filter_2_d(sim: &Sim, in_vec: &mut Vec<Float>, wrkspace: &mut Vec<Float>) {
    // wrkspace should be same size as fld
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
            // handle the ghost zones in x direction
            wrkspace[i - 1] = wrkspace[i + sim.size_x];
            wrkspace[i + sim.size_x + 1] = wrkspace[i];
        }
        // handle the ghost zones in y direction
        // I COULD DO THIS WITH MEMCPY AND I KNOW IT IS POSSIBLE WITH SAFE
        // RUST BUT I DON'T KNOW HOW :(

        for j in 0..sim.size_x + 2 {
            wrkspace[j] = wrkspace[sim.size_y * sim.size_x + j];
            wrkspace[(sim.size_y + 1) * sim.size_x + j] = wrkspace[sim.size_x + j];
        }
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
            // handle the ghost zones in x direction
            in_vec[i - 1] = in_vec[i + sim.size_x];
            in_vec[i + sim.size_x + 1] = in_vec[i];
        }
        // handle the ghost zones in y direction
        // I COULD DO THIS WITH MEMCPY AND I KNOW IT IS POSSIBLE WITH SAFE
        // RUST BUT I DON'T KNOW HOW :(

        for j in 0..sim.size_x + 2 {
            in_vec[j] = in_vec[sim.size_y * sim.size_x + j];
            in_vec[(sim.size_y + 1) * sim.size_x + j] = in_vec[sim.size_x + j];
        }
    }
}
struct Flds {
    e_x: Vec<Float>,
    e_y: Vec<Float>,
    e_z: Vec<Float>,
    b_x: Vec<Float>,
    b_y: Vec<Float>,
    b_z: Vec<Float>,
    j_x: Vec<Float>,
    j_y: Vec<Float>,
    j_z: Vec<Float>,
    k_x: Vec<Float>,
    k_y: Vec<Float>,
    k_norm: Vec<Float>,
    b_xr: Vec<Complex<Float>>,
    b_yr: Vec<Complex<Float>>,
    b_zr: Vec<Complex<Float>>,
    b_x2: Vec<Complex<Float>>,
    b_y2: Vec<Complex<Float>>,
    b_z2: Vec<Complex<Float>>,
    fft_x: std::sync::Arc<dyn rustfft::FFT<Float>>,
    ifft_x: std::sync::Arc<dyn rustfft::FFT<Float>>,
    fft_y: std::sync::Arc<dyn rustfft::FFT<Float>>,
    ifft_y: std::sync::Arc<dyn rustfft::FFT<Float>>,
    real_wrkspace_ghosts: Vec<Float>,
    real_wrkspace: Vec<Float>,
    cmp_wrkspace: Vec<Complex<Float>>,
    c_x: Vec<Complex<Float>>,
    c_y: Vec<Complex<Float>>,
    c_z: Vec<Complex<Float>>,
    dsty_cmp: Vec<Complex<Float>>,
    dsty: Vec<Float>,
}
impl Flds {
    fn new(sim: &Sim) -> Flds {
        //let Bnorm = 0f32;
        let mut planner = FFTplanner::new(false);
        //let () = planner;
        let mut inv_planner = FFTplanner::new(true);
        //let mut input:  Vec<Complex<f32>> = vec![Complex::zero(); sim.size_x];
        //let mut output: Vec<Complex<f32>> = vec![Complex::zero();  sim.size_x];

        let fft_x = planner.plan_fft(sim.size_x);
        let ifft_x = inv_planner.plan_fft(sim.size_x);
        let fft_y = planner.plan_fft(sim.size_y);
        let ifft_y = planner.plan_fft(sim.size_y);
        let mut input: Vec<Complex<Float>> = vec![Complex::zero(); sim.size_x];
        let mut output: Vec<Complex<Float>> = vec![Complex::zero(); sim.size_x];
        //let ifft_x = inv_planner.plan_fft(sim.size_x);
        //let fft_y = planner.plan_fft(sim.size_y);
        //let ifft_y = inv_planner.plan_fft(sim.size_y);
        fft_x.process(&mut input, &mut output);

        let mut f = Flds {
            e_x: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)], // 2 Ghost zones. 1 at 0, 1 at SIZE_X
            e_y: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            e_z: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            b_x: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            b_y: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            b_z: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            j_x: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            j_y: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            j_z: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            k_x: vec![0.0; sim.size_x],
            k_y: vec![0.0; sim.size_y],
            k_norm: vec![0.0; sim.size_y * sim.size_x],
            fft_x: fft_x,
            ifft_x: ifft_x,
            fft_y: fft_y,
            ifft_y: ifft_y,
            real_wrkspace_ghosts: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            real_wrkspace: vec![0.0; (sim.size_y) * (sim.size_x)],
            cmp_wrkspace: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            c_x: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            c_y: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            c_z: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_xr: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_yr: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_zr: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_x2: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_y2: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_z2: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            dsty: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            dsty_cmp: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
        };
        // Build the k basis of FFT
        for i in 0..f.k_x.len() {
            f.k_x[i] = i as Float;
            if i > sim.size_x / 2 + 1 {
                f.k_x[i] -= sim.size_x as Float;
            }
            f.k_x[i] *= 2.0 * PI / (sim.size_x as Float);
        }
        for i in 0..f.k_y.len() {
            f.k_y[i] = i as Float;
            if i > sim.size_y / 2 + 1 {
                f.k_y[i] -= sim.size_y as Float;
            }
            f.k_y[i] *= 2.0 * PI / (sim.size_y as Float);
        }
        // Make the norm:
        for i in 0..f.k_y.len() {
            for j in 0..f.k_x.len() {
                f.k_norm[i * sim.size_x + j] = 1. / (f.k_x[j] * f.k_x[j] + f.k_y[i] * f.k_y[i]);
            }
        }

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
        f
    }
    pub fn transpose(sim: &Sim, in_fld: &Vec<Complex<Float>>, out_fld: &mut Vec<Complex<Float>>) {
        for i in 0..sim.size_y {
            for j in 0..sim.size_x {
                if cfg!(feature = "unsafe") {
                    unsafe {
                        *out_fld.get_unchecked_mut(i * sim.size_x + j) =
                            *in_fld.get_unchecked(j * sim.size_y + i);
                    }
                } else {
                    out_fld[i * sim.size_x + j] = in_fld[j * sim.size_y + i];
                }
            }
        }
    }
    pub fn fft2d(
        fft_x: std::sync::Arc<dyn rustfft::FFT<Float>>,
        fft_y: std::sync::Arc<dyn rustfft::FFT<Float>>,
        sim: &Sim,
        fld: &mut Vec<Complex<Float>>,
        wrk_space: &mut Vec<Complex<Float>>,
    ) {
        for iy in (0..sim.size_y * sim.size_x).step_by(sim.size_x) {
            fft_x.process(
                &mut fld[iy..iy + sim.size_x],
                &mut wrk_space[iy..iy + sim.size_x],
            );
        }
        Flds::transpose(sim, wrk_space, fld);
        for iy in (0..sim.size_x * sim.size_y).step_by(sim.size_y) {
            fft_y.process(
                &mut fld[iy..iy + sim.size_y],
                &mut wrk_space[iy..iy + sim.size_y],
            );
        }
        Flds::transpose(sim, wrk_space, fld);
    }
    pub fn update(&mut self, sim: &Sim) {
        let ckc = sim.dt / sim.dens as Float;
        let cdt = sim.dt * sim.c as Float;
        // Filter fields
        binomial_filter_2_d(sim, &mut self.j_x, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.j_y, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.j_z, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.dsty, &mut self.real_wrkspace_ghosts);
        // copy j_x, j_y, j_z, dsty into complex vector
        let mut ij: usize;
        let mut ij_ghosts: usize;
        for iy in 0..sim.size_y {
            ij = iy * (sim.size_x);
            ij_ghosts = (iy + 1) * (sim.size_x + 2);
            for ix in 0..sim.size_x {
                self.c_x[ij + ix].re = self.j_x[ij_ghosts + ix + 1];
                self.c_x[ij + ix].im = 0.0;
                self.c_y[ij + ix].re = self.j_y[ij_ghosts + ix + 1];
                self.c_y[ij + ix].im = 0.0;
                self.c_z[ij + ix].re = self.j_z[ij_ghosts + ix + 1];
                self.c_z[ij + ix].im = 0.0;
                self.dsty_cmp[ij + ix].re = self.dsty[ij_ghosts + ix + 1] / (sim.dens as Float);
                self.dsty_cmp[ij + ix].im = 0.0;
            }
        }
        // Take fft of currents
        Flds::fft2d(
            self.fft_x.clone(),
            self.fft_y.clone(),
            sim,
            &mut self.c_x,
            &mut self.cmp_wrkspace,
        );
        Flds::fft2d(
            self.fft_x.clone(),
            self.fft_y.clone(),
            sim,
            &mut self.c_y,
            &mut self.cmp_wrkspace,
        );
        Flds::fft2d(
            self.fft_x.clone(),
            self.fft_y.clone(),
            sim,
            &mut self.c_z,
            &mut self.cmp_wrkspace,
        );
        Flds::fft2d(
            self.fft_x.clone(),
            self.fft_y.clone(),
            sim,
            &mut self.dsty_cmp,
            &mut self.cmp_wrkspace,
        );

        // copy previous timestep should maybe use memcopy
        for (b2, br) in self.b_x2.iter_mut().zip(self.b_xr.iter()) {
            *b2 = *br;
        }
        for (b2, br) in self.b_y2.iter_mut().zip(self.b_yr.iter()) {
            *b2 = *br;
        }
        for (b2, br) in self.b_z2.iter_mut().zip(self.b_zr.iter()) {
            *b2 = *br;
        }
    }
}
struct Sim {
    // flds: Flds,
    // prtls: Vec<Prtl>,
    t: u32,
    t_final: u32,
    size_x: usize,
    size_y: usize,
    delta: usize,
    dt: Float,
    c: Float,
    dens: u32,
    gamma_inj: Float, // Speed of upstream flow
    prtl_num: usize,  // = *DENS * ( *SIZE_X - 2* *DELTA) * *SIZE_Y;
    n_pass: u8,       // = 4; //Number of filter passes
}

impl Sim {
    fn new(cfg: &Config) -> Sim {
        Sim {
            t: 0,
            t_final: cfg.setup.t_final,
            size_x: cfg.params.size_x,
            size_y: cfg.params.size_y,
            delta: cfg.params.delta,
            dt: cfg.params.dt,
            c: cfg.params.c,
            dens: cfg.params.dens,
            gamma_inj: cfg.params.gamma_inj,
            prtl_num: cfg.params.dens as usize
                * (cfg.params.size_x - 2 * cfg.params.delta)
                * cfg.params.size_y,
            n_pass: cfg.params.n_pass,
        }
    }

    fn deposit_current(&self, prtl: &Prtl, flds: &mut Flds) {
        // local vars we will use
        let mut ij: usize;
        let mut ijm1: usize;
        let mut ijp1: usize;

        // for the weights
        let mut w00: Float;
        let mut w01: Float;
        let mut w02: Float;
        let mut w10: Float;
        let mut w11: Float;
        let mut w12: Float;
        let mut w20: Float;
        let mut w21: Float;
        let mut w22: Float;

        let mut vx: Float;
        let mut vy: Float;
        let mut vz: Float;
        let mut psa_inv: Float;

        for (ix, iy, dx, dy, px, py, pz, psa) in
            izip!(&prtl.ix, &prtl.iy, &prtl.dx, &prtl.dy, &prtl.px, &prtl.py, &prtl.pz, &prtl.psa)
        {
            ijm1 = iy - 1;
            ijp1 = iy + 1;
            //if ix1 >= *SIZE_X {
            //    ix1 -= *SIZE_X;
            //    ix2 -= *SIZE_X;
            //} else if ix2 >= *SIZE_X {
            //    ix2 -= *SIZE_X;
            //}
            ij = iy * (2 + self.size_x);
            ijm1 *= 2 + self.size_x;
            ijp1 *= 2 + self.size_x;
            psa_inv = psa.powi(-1);
            vx = prtl.charge * px * psa_inv;
            vy = prtl.charge * py * psa_inv;
            vz = prtl.charge * pz * psa_inv;
            // CALC WEIGHTS
            // 2nd order
            // The weighting scheme prtl is in middle
            // # ----------------------
            // # | w0,0 | w0,1 | w0,2 |
            // # ----------------------
            // # | w1,0 | w1,1 | w1,2 |
            // # ----------------------
            // # | w2,0 | w2,1 | w2,2 |
            // # ----------------------
            w00 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w01 = 0.5 * (0.5 - dy) * (0.5 - dy) * (0.75 - dx * dx); // y0
            w02 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0
            w10 = (0.75 - dy * dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w11 = (0.75 - dy * dy) * (0.75 - dx * dx); // y0
            w12 = (0.75 - dy * dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0
            w20 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w21 = 0.5 * (0.5 + dy) * (0.5 + dy) * (0.75 - dx * dx); // y0
            w22 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0

            // Deposit the CURRENT
            if cfg!(feature = "unsafe") {
                unsafe {
                    *flds.j_x.get_unchecked_mut(ijm1 + ix - 1) += w00 * vx;
                    *flds.j_x.get_unchecked_mut(ijm1 + ix) += w01 * vx;
                    *flds.j_x.get_unchecked_mut(ijm1 + ix + 1) += w02 * vx;
                    *flds.j_x.get_unchecked_mut(ij + ix - 1) += w10 * vx;
                    *flds.j_x.get_unchecked_mut(ij + ix) += w11 * vx;
                    *flds.j_x.get_unchecked_mut(ij + ix + 1) += w12 * vx;
                    *flds.j_x.get_unchecked_mut(ijp1 + ix - 1) += w20 * vx;
                    *flds.j_x.get_unchecked_mut(ijp1 + ix) += w21 * vx;
                    *flds.j_x.get_unchecked_mut(ijp1 + ix + 1) += w22 * vx;

                    *flds.j_y.get_unchecked_mut(ijm1 + ix - 1) += w00 * vy;
                    *flds.j_y.get_unchecked_mut(ijm1 + ix) += w01 * vy;
                    *flds.j_y.get_unchecked_mut(ijm1 + ix + 1) += w02 * vy;
                    *flds.j_y.get_unchecked_mut(ij + ix - 1) += w10 * vy;
                    *flds.j_y.get_unchecked_mut(ij + ix) += w11 * vy;
                    *flds.j_y.get_unchecked_mut(ij + ix + 1) += w12 * vy;
                    *flds.j_y.get_unchecked_mut(ijp1 + ix - 1) += w20 * vy;
                    *flds.j_y.get_unchecked_mut(ijp1 + ix) += w21 * vy;
                    *flds.j_y.get_unchecked_mut(ijp1 + ix + 1) += w22 * vy;

                    *flds.j_z.get_unchecked_mut(ijm1 + ix - 1) += w00 * vz;
                    *flds.j_z.get_unchecked_mut(ijm1 + ix) += w01 * vz;
                    *flds.j_z.get_unchecked_mut(ijm1 + ix + 1) += w02 * vz;
                    *flds.j_z.get_unchecked_mut(ij + ix - 1) += w10 * vz;
                    *flds.j_z.get_unchecked_mut(ij + ix) += w11 * vz;
                    *flds.j_z.get_unchecked_mut(ij + ix + 1) += w12 * vz;
                    *flds.j_z.get_unchecked_mut(ijp1 + ix - 1) += w20 * vz;
                    *flds.j_z.get_unchecked_mut(ijp1 + ix) += w21 * vz;
                    *flds.j_z.get_unchecked_mut(ijp1 + ix + 1) += w22 * vz;
                }
            } else {
                flds.j_x[ijm1 + ix - 1] += w00 * vx;
                flds.j_x[ijm1 + ix] += w01 * vx;
                flds.j_x[ijm1 + ix + 1] += w02 * vx;
                flds.j_x[ij + ix - 1] += w10 * vx;
                flds.j_x[ij + ix] += w11 * vx;
                flds.j_x[ij + ix + 1] += w12 * vx;
                flds.j_x[ijp1 + ix - 1] += w20 * vx;
                flds.j_x[ijp1 + ix] += w21 * vx;
                flds.j_x[ijp1 + ix + 1] += w22 * vx;

                flds.j_y[ijm1 + ix - 1] += w00 * vy;
                flds.j_y[ijm1 + ix] += w01 * vy;
                flds.j_y[ijm1 + ix + 1] += w02 * vy;
                flds.j_y[ij + ix - 1] += w10 * vy;
                flds.j_y[ij + ix] += w11 * vy;
                flds.j_y[ij + ix + 1] += w12 * vy;
                flds.j_y[ijp1 + ix - 1] += w20 * vy;
                flds.j_y[ijp1 + ix] += w21 * vy;
                flds.j_y[ijp1 + ix + 1] += w22 * vy;

                flds.j_z[ijm1 + ix - 1] += w00 * vz;
                flds.j_z[ijm1 + ix] += w01 * vz;
                flds.j_z[ijm1 + ix + 1] += w02 * vz;
                flds.j_z[ij + ix - 1] += w10 * vz;
                flds.j_z[ij + ix] += w11 * vz;
                flds.j_z[ij + ix + 1] += w12 * vz;
                flds.j_z[ijp1 + ix - 1] += w20 * vz;
                flds.j_z[ijp1 + ix] += w21 * vz;
                flds.j_z[ijp1 + ix + 1] += w22 * vz;
            }
        }
    }

    fn calc_density(&self, prtl: &Prtl, flds: &mut Flds) {
        // local vars we will use
        let mut ij: usize;
        let mut ijm1: usize;
        let mut ijp1: usize;

        // for the weights
        let mut w00: Float;
        let mut w01: Float;
        let mut w02: Float;
        let mut w10: Float;
        let mut w11: Float;
        let mut w12: Float;
        let mut w20: Float;
        let mut w21: Float;
        let mut w22: Float;

        for (ix, iy, dx, dy) in izip!(&prtl.ix, &prtl.iy, &prtl.dx, &prtl.dy) {
            ijm1 = iy - 1;
            ijp1 = iy + 1;
            //if ix1 >= *SIZE_X {
            //    ix1 -= *SIZE_X;
            //    ix2 -= *SIZE_X;
            //} else if ix2 >= *SIZE_X {
            //    ix2 -= *SIZE_X;
            //}
            ij = iy * (2 + self.size_x);
            ijm1 *= 2 + self.size_x;
            ijp1 *= 2 + self.size_x;

            // CALC WEIGHTS
            // 2nd order
            // The weighting scheme prtl is in middle
            // # ----------------------
            // # | w0,0 | w0,1 | w0,2 |
            // # ----------------------
            // # | w1,0 | w1,1 | w1,2 |
            // # ----------------------
            // # | w2,0 | w2,1 | w2,2 |
            // # ----------------------
            w00 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w01 = 0.5 * (0.5 - dy) * (0.5 - dy) * (0.75 - dx * dx); // y0
            w02 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0
            w10 = (0.75 - dy * dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w11 = (0.75 - dy * dy) * (0.75 - dx * dx); // y0
            w12 = (0.75 - dy * dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0
            w20 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w21 = 0.5 * (0.5 + dy) * (0.5 + dy) * (0.75 - dx * dx); // y0
            w22 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0

            // Deposit the CURRENT
            if cfg!(feature = "unsafe") {
                unsafe {
                    *flds.dsty.get_unchecked_mut(ijm1 + ix - 1) += w00 * prtl.charge;
                    *flds.dsty.get_unchecked_mut(ijm1 + ix) += w01 * prtl.charge;
                    *flds.dsty.get_unchecked_mut(ijm1 + ix + 1) += w02 * prtl.charge;
                    *flds.dsty.get_unchecked_mut(ij + ix - 1) += w10 * prtl.charge;
                    *flds.dsty.get_unchecked_mut(ij + ix) += w11 * prtl.charge;
                    *flds.dsty.get_unchecked_mut(ij + ix + 1) += w12 * prtl.charge;
                    *flds.dsty.get_unchecked_mut(ijp1 + ix - 1) += w20 * prtl.charge;
                    *flds.dsty.get_unchecked_mut(ijp1 + ix) += w21 * prtl.charge;
                    *flds.dsty.get_unchecked_mut(ijp1 + ix + 1) += w22 * prtl.charge;
                }
            } else {
                flds.dsty[ijm1 + ix - 1] += w00 * prtl.charge;
                flds.dsty[ijm1 + ix] += w01 * prtl.charge;
                flds.dsty[ijm1 + ix + 1] += w02 * prtl.charge;
                flds.dsty[ij + ix - 1] += w10 * prtl.charge;
                flds.dsty[ij + ix] += w11 * prtl.charge;
                flds.dsty[ij + ix + 1] += w12 * prtl.charge;
                flds.dsty[ijp1 + ix - 1] += w20 * prtl.charge;
                flds.dsty[ijp1 + ix] += w21 * prtl.charge;
                flds.dsty[ijp1 + ix + 1] += w22 * prtl.charge;
            }
        }
    }
    fn move_and_deposit(&self, prtl: &mut Prtl, flds: &mut Flds) {
        // FIRST we update positions of particles
        let mut c1: Float;
        for (ix, iy, dx, dy, px, py, psa) in izip!(
            &mut prtl.ix,
            &mut prtl.iy,
            &mut prtl.dx,
            &mut prtl.dy,
            &prtl.px,
            &prtl.py,
            &prtl.psa
        ) {
            c1 = 0.5 * self.dt * psa.powi(-1);
            *dx += c1 * px;
            if *dx >= 0.5 {
                *dx -= 1.0;
                *ix += 1;
            } else if *dx < -0.5 {
                *dx += 1.0;
                *ix -= 1;
            }
            *dy += c1 * py;
            if *dy >= 0.5 {
                *dy -= 1.0;
                *iy += 1;
            } else if *dy < -0.5 {
                *dy += 1.0;
                *iy -= 1;
            }
        }
        //self.dsty *=0
        prtl.apply_bc(self);

        // Deposit currents
        self.deposit_current(prtl, flds);

        // UPDATE POS AGAIN!
        for (ix, iy, dx, dy, px, py, psa) in izip!(
            &mut prtl.ix,
            &mut prtl.iy,
            &mut prtl.dx,
            &mut prtl.dy,
            &prtl.px,
            &prtl.py,
            &prtl.psa
        ) {
            c1 = 0.5 * self.dt * psa.powi(-1);
            *dx += c1 * px;
            if *dx >= 0.5 {
                *dx -= 1.0;
                *ix += 1;
            } else if *dx < -0.5 {
                *dx += 1.0;
                *ix -= 1;
            }
            *dy += c1 * py;
            if *dy >= 0.5 {
                *dy -= 1.0;
                *iy += 1;
            } else if *dy < -0.5 {
                *dy += 1.0;
                *iy -= 1;
            }
        }
        prtl.apply_bc(self);

        // # CALCULATE DENSITY
        //calculateDens(self.x, self.y, self.dsty)#, self.charge)
        //self.sim.dsty += self.charge*self.dsty
    }
}

struct Prtl {
    ix: Vec<usize>,
    iy: Vec<usize>,
    dx: Vec<Float>,
    dy: Vec<Float>,
    px: Vec<Float>,
    py: Vec<Float>,
    pz: Vec<Float>,
    psa: Vec<Float>, // Lorentz Factors
    charge: Float,
    alpha: Float,
    beta: Float,
    vth: Float,
    tag: Vec<u64>,
    track: Vec<bool>,
}
fn fld2prtl(sim: &Sim, ix: usize, iy: usize, dx: Float, dy: Float, fld: &Vec<Float>) -> Float {
    let ijm1 = (iy - 1) * (sim.size_x + 2) + ix;
    let ij = iy * (sim.size_x + 2) + ix;
    let ijp1 = (iy + 1) * (sim.size_x + 2) + ix;

    let weights = [
        0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 - dx) * (0.5 - dx),
        0.5 * (0.5 - dy) * (0.5 - dy) * (0.75 - dx * dx),
        0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5 + dx),
        (0.75 - dy * dy) * 0.5 * (0.5 - dx) * (0.5 - dx),
        (0.75 - dy * dy) * (0.75 - dx * dx),
        (0.75 - dy * dy) * 0.5 * (0.5 + dx) * (0.5 + dx),
        0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 - dx) * (0.5 - dx),
        0.5 * (0.5 + dy) * (0.5 + dy) * (0.75 - dx * dx),
        0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 + dx) * (0.5 + dx),
    ];
    weights
        .iter()
        .zip(
            fld[ijm1 - 1..ijm1 + 1]
                .into_iter()
                .chain(fld[ij - 1..ij + 1].into_iter())
                .chain(fld[ijp1 - 1..ijp1 + 1].into_iter()),
        )
        .map(|(&w, &f)| w * f)
        .sum::<Float>()
}

impl Prtl {
    fn new(sim: &Sim, charge: Float, mass: Float, vth: Float) -> Prtl {
        let beta = (charge / mass) * 0.5 * sim.dt;
        let alpha = (charge / mass) * 0.5 * sim.dt / sim.c;
        let mut prtl = Prtl {
            ix: vec![0; sim.prtl_num],
            dx: vec![0.0; sim.prtl_num],
            iy: vec![0; sim.prtl_num],
            dy: vec![0.0; sim.prtl_num],
            px: vec![0.0; sim.prtl_num],
            py: vec![0.0; sim.prtl_num],
            pz: vec![0.0; sim.prtl_num],
            psa: vec![0.0; sim.prtl_num],
            track: vec![false; sim.prtl_num],
            tag: vec![0u64; sim.prtl_num],
            charge: charge,
            vth: vth,
            alpha: alpha,
            beta: beta,
        };
        prtl.track[40] = true;
        prtl.initialize_positions(sim);
        prtl.initialize_velocities(sim);
        prtl.apply_bc(sim);
        prtl
    }
    fn apply_bc(&mut self, sim: &Sim) {
        // PERIODIC BOUNDARIES IN Y
        // First iterate over y array and apply BC
        for iy in self.iy.iter_mut() {
            if *iy < 1 {
                *iy += sim.size_y;
            } else if *iy > sim.size_y {
                *iy -= sim.size_y;
            }
        }

        // Now iterate over x array
        for ix in self.ix.iter_mut() {
            if *ix < 1 {
                *ix += sim.size_x;
            } else if *ix > sim.size_x {
                *ix -= sim.size_x;
            }
        }

        // x boundary conditions are incorrect
        // let c1 = sim.size_x - sim.delta;
        // let c2 = 2 * c1;

        //for (ix, px) in self.ix.iter_mut().zip(self.px.iter_mut()) {
        //     if *ix >= c1 {
        //         *ix = c2 - *ix;
        //         *px *= -1.0;
        //     }
        //}
    }
    fn initialize_positions(&mut self, sim: &Sim) {
        // A method to calculate the initial, non-random
        // position of the particles
        let mut c1 = 0;
        // let mut rng = thread_rng();
        if true {
            for i in 0..sim.size_y {
                for j in sim.delta..sim.size_x - sim.delta {
                    for k in 0..sim.dens as usize {
                        // RANDOM OPT
                        // let r1: f32 = rng.sample(Standard);
                        // let r2: f32 = rng.sample(Standard);
                        // self.x[c1+k]= r1 + (j as f32);
                        // self.y[c1+k]= r2 + (i as f32);

                        // UNIFORM OPT
                        self.iy[c1 + k] = i + 1;
                        self.ix[c1 + k] = j + 1;

                        let mut r1 = 1.0 / (2.0 * (sim.dens as Float));
                        r1 = (2. * (k as Float) + 1.) * r1;
                        self.dx[c1 + k] = r1 - 0.5;
                        self.dy[c1 + k] = r1 - 0.5;
                        self.tag[c1 + k] = (c1 + k) as u64;
                    }
                    c1 += sim.dens as usize;
                    // helper_arr = -.5+0.25+np.arange(dens)*.5
                }
            }
        } else {
            for i in 0..self.iy.len() {
                self.iy[i] = 1 + i * ((sim.size_y as Float) / (self.iy.len() as Float)) as usize;
                self.ix[i] = sim.size_x - 4;
                self.dx[i] = 0.0;
                self.dy[i] = 0.0;
            }
        }
        //    for j in range(delta, Lx-delta):
        //#for i in range(Ly//2, Ly//2+10):
        //    for j in range(delta, delta+10):
        //        xArr[c1:c1+dens] = helper_arr + j
        //        yArr[c1:c1+dens] = helper_arr + i
        //        c1+=dens
    }
    fn initialize_velocities(&mut self, sim: &Sim) {
        //placeholder
        let csqinv = 1. / (sim.c * sim.c);
        let beta_inj = -Float::sqrt(1. - sim.gamma_inj.powi(-2));
        // println!("{}", beta_inj);
        if true {
            let mut rng = thread_rng();

            for (px, py, pz, psa) in izip!(&mut self.px, &mut self.py, &mut self.pz, &mut self.psa)
            {
                *px = rng.sample(StandardNormal);
                *px *= self.vth * sim.c;
                *py = rng.sample(StandardNormal);
                *py *= self.vth * sim.c;
                *pz = rng.sample(StandardNormal);
                *pz *= self.vth * sim.c;
                *psa = 1.0 + (*px * *px + *py * *py + *pz * *pz) * csqinv;
                *psa = psa.sqrt();

                // Flip the px according to zenitani 2015
                let mut ux = *px / sim.c;
                let rand: Float = rng.sample(Standard);
                if -beta_inj * ux > rand * *psa {
                    ux *= -1.
                }
                *px = sim.gamma_inj * (ux + beta_inj * *psa); // not p yet... really ux-prime
                *px *= sim.c;
                *psa = 1.0 + (*px * *px + *py * *py + *pz * *pz) * csqinv;
                *psa = psa.sqrt();
            }
        } else {
            for (px, psa) in izip!(&mut self.px, &mut self.psa) {
                *px = -sim.c * sim.gamma_inj * beta_inj;
                *psa = 1.0 + (*px * *px) * csqinv;
                *psa = psa.sqrt();
            }
        }
    }
    fn boris_push(&mut self, sim: &Sim, flds: &Flds) {
        // local vars we will use
        let mut ijm1: usize;
        let mut ijp1: usize;
        let mut ij: usize;

        let csqinv = 1. / (sim.c * sim.c);
        // for the weights
        let mut w00: Float;
        let mut w01: Float;
        let mut w02: Float;
        let mut w10: Float;
        let mut w11: Float;
        let mut w12: Float;
        let mut w20: Float;
        let mut w21: Float;
        let mut w22: Float;

        let mut ext: Float;
        let mut eyt: Float;
        let mut ezt: Float;
        let mut bxt: Float;
        let mut byt: Float;
        let mut bzt: Float;
        let mut ux: Float;
        let mut uy: Float;
        let mut uz: Float;
        let mut uxt: Float;
        let mut uyt: Float;
        let mut uzt: Float;
        let mut pt: Float;
        let mut gt: Float;
        let mut boris: Float;

        for (ix, iy, dx, dy, px, py, pz, psa) in izip!(
            &self.ix,
            &self.iy,
            &self.dx,
            &self.dy,
            &mut self.px,
            &mut self.py,
            &mut self.pz,
            &mut self.psa
        ) {
            ijm1 = iy - 1;
            ijp1 = iy + 1;
            ij = iy * (2 + sim.size_x);
            ijm1 *= 2 + sim.size_x;
            ijp1 *= 2 + sim.size_x;
            // CALC WEIGHTS
            // 2nd order
            // The weighting scheme prtl is in middle
            // # ----------------------
            // # | w0,0 | w0,1 | w0,2 |
            // # ----------------------
            // # | w1,0 | w1,1 | w1,2 |
            // # ----------------------
            // # | w2,0 | w2,1 | w2,2 |
            // # ----------------------
            w00 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w01 = 0.5 * (0.5 - dy) * (0.5 - dy) * (0.75 - dx * dx); // y0
            w02 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0
            w10 = (0.75 - dy * dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w11 = (0.75 - dy * dy) * (0.75 - dx * dx); // y0
            w12 = (0.75 - dy * dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0
            w20 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w21 = 0.5 * (0.5 + dy) * (0.5 + dy) * (0.75 - dx * dx); // y0
            w22 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0

            // INTERPOLATE ALL THE FIELDS
            if cfg!(feature = "unsafe") {
                unsafe {
                    ext = w00 * flds.e_x.get_unchecked(ijm1 + ix - 1);
                    ext += w01 * flds.e_x.get_unchecked(ijm1 + ix);
                    ext += w02 * flds.e_x.get_unchecked(ijm1 + ix + 1);
                    ext += w10 * flds.e_x.get_unchecked(ij + ix - 1);
                    ext += w11 * flds.e_x.get_unchecked(ij + ix);
                    ext += w12 * flds.e_x.get_unchecked(ij + ix + 1);
                    ext += w20 * flds.e_x.get_unchecked(ijp1 + ix - 1);
                    ext += w21 * flds.e_x.get_unchecked(ijp1 + ix);
                    ext += w22 * flds.e_x.get_unchecked(ijp1 + ix + 1);

                    eyt = w00 * flds.e_y.get_unchecked(ijm1 + ix - 1);
                    eyt += w01 * flds.e_y.get_unchecked(ijm1 + ix);
                    eyt += w02 * flds.e_y.get_unchecked(ijm1 + ix + 1);
                    eyt += w10 * flds.e_y.get_unchecked(ij + ix - 1);
                    eyt += w11 * flds.e_y.get_unchecked(ij + ix);
                    eyt += w12 * flds.e_y.get_unchecked(ij + ix + 1);
                    eyt += w20 * flds.e_y.get_unchecked(ijp1 + ix - 1);
                    eyt += w21 * flds.e_y.get_unchecked(ijp1 + ix);
                    eyt += w22 * flds.e_y.get_unchecked(ijp1 + ix + 1);

                    ezt = w00 * flds.e_z.get_unchecked(ijm1 + ix - 1);
                    ezt += w01 * flds.e_z.get_unchecked(ijm1 + ix);
                    ezt += w02 * flds.e_z.get_unchecked(ijm1 + ix + 1);
                    ezt += w10 * flds.e_z.get_unchecked(ij + ix - 1);
                    ezt += w11 * flds.e_z.get_unchecked(ij + ix);
                    ezt += w12 * flds.e_z.get_unchecked(ij + ix + 1);
                    ezt += w20 * flds.e_z.get_unchecked(ijp1 + ix - 1);
                    ezt += w21 * flds.e_z.get_unchecked(ijp1 + ix);
                    ezt += w22 * flds.e_z.get_unchecked(ijp1 + ix + 1);

                    bxt = w00 * flds.b_x.get_unchecked(ijm1 + ix - 1);
                    bxt += w01 * flds.b_x.get_unchecked(ijm1 + ix);
                    bxt += w02 * flds.b_x.get_unchecked(ijm1 + ix + 1);
                    bxt += w10 * flds.b_x.get_unchecked(ij + ix - 1);
                    bxt += w11 * flds.b_x.get_unchecked(ij + ix);
                    bxt += w12 * flds.b_x.get_unchecked(ij + ix + 1);
                    bxt += w20 * flds.b_x.get_unchecked(ijp1 + ix - 1);
                    bxt += w21 * flds.b_x.get_unchecked(ijp1 + ix);
                    bxt += w22 * flds.b_x.get_unchecked(ijp1 + ix + 1);

                    byt = w00 * flds.b_y.get_unchecked(ijm1 + ix - 1);
                    byt += w01 * flds.b_y.get_unchecked(ijm1 + ix);
                    byt += w02 * flds.b_y.get_unchecked(ijm1 + ix + 1);
                    byt += w10 * flds.b_y.get_unchecked(ij + ix - 1);
                    byt += w11 * flds.b_y.get_unchecked(ij + ix);
                    byt += w12 * flds.b_y.get_unchecked(ij + ix + 1);
                    byt += w20 * flds.b_y.get_unchecked(ijp1 + ix - 1);
                    byt += w21 * flds.b_y.get_unchecked(ijp1 + ix);
                    byt += w22 * flds.b_y.get_unchecked(ijp1 + ix + 1);

                    bzt = w00 * flds.b_z.get_unchecked(ijm1 + ix - 1);
                    bzt += w01 * flds.b_z.get_unchecked(ijm1 + ix);
                    bzt += w02 * flds.b_z.get_unchecked(ijm1 + ix + 1);
                    bzt += w10 * flds.b_z.get_unchecked(ij + ix - 1);
                    bzt += w11 * flds.b_z.get_unchecked(ij + ix);
                    bzt += w12 * flds.b_z.get_unchecked(ij + ix + 1);
                    bzt += w20 * flds.b_z.get_unchecked(ijp1 + ix - 1);
                    bzt += w21 * flds.b_z.get_unchecked(ijp1 + ix);
                    bzt += w22 * flds.b_z.get_unchecked(ijp1 + ix + 1);
                }
            } else {
                ext = w00 * flds.e_x[ijm1 + ix - 1];
                ext += w01 * flds.e_x[ijm1 + ix];
                ext += w02 * flds.e_x[ijm1 + ix + 1];
                ext += w10 * flds.e_x[ij + ix - 1];
                ext += w11 * flds.e_x[ij + ix];
                ext += w12 * flds.e_x[ij + ix + 1];
                ext += w20 * flds.e_x[ijp1 + ix - 1];
                ext += w21 * flds.e_x[ijp1 + ix];
                ext += w22 * flds.e_x[ijp1 + ix + 1];

                eyt = w00 * flds.e_y[ijm1 + ix - 1];
                eyt += w01 * flds.e_y[ijm1 + ix];
                eyt += w02 * flds.e_y[ijm1 + ix + 1];
                eyt += w10 * flds.e_y[ij + ix - 1];
                eyt += w11 * flds.e_y[ij + ix];
                eyt += w12 * flds.e_y[ij + ix + 1];
                eyt += w20 * flds.e_y[ijp1 + ix - 1];
                eyt += w21 * flds.e_y[ijp1 + ix];
                eyt += w22 * flds.e_y[ijp1 + ix + 1];

                ezt = w00 * flds.e_z[ijm1 + ix - 1];
                ezt += w01 * flds.e_z[ijm1 + ix];
                ezt += w02 * flds.e_z[ijm1 + ix + 1];
                ezt += w10 * flds.e_z[ij + ix - 1];
                ezt += w11 * flds.e_z[ij + ix];
                ezt += w12 * flds.e_z[ij + ix + 1];
                ezt += w20 * flds.e_z[ijp1 + ix - 1];
                ezt += w21 * flds.e_z[ijp1 + ix];
                ezt += w22 * flds.e_z[ijp1 + ix + 1];

                bxt = w00 * flds.b_x[ijm1 + ix - 1];
                bxt += w01 * flds.b_x[ijm1 + ix];
                bxt += w02 * flds.b_x[ijm1 + ix + 1];
                bxt += w10 * flds.b_x[ij + ix - 1];
                bxt += w11 * flds.b_x[ij + ix];
                bxt += w12 * flds.b_x[ij + ix + 1];
                bxt += w20 * flds.b_x[ijp1 + ix - 1];
                bxt += w21 * flds.b_x[ijp1 + ix];
                bxt += w22 * flds.b_x[ijp1 + ix + 1];

                byt = w00 * flds.b_y[ijm1 + ix - 1];
                byt += w01 * flds.b_y[ijm1 + ix];
                byt += w02 * flds.b_y[ijm1 + ix + 1];
                byt += w10 * flds.b_y[ij + ix - 1];
                byt += w11 * flds.b_y[ij + ix];
                byt += w12 * flds.b_y[ij + ix + 1];
                byt += w20 * flds.b_y[ijp1 + ix - 1];
                byt += w21 * flds.b_y[ijp1 + ix];
                byt += w22 * flds.b_y[ijp1 + ix + 1];

                bzt = w00 * flds.b_z[ijm1 + ix - 1];
                bzt += w01 * flds.b_z[ijm1 + ix];
                bzt += w02 * flds.b_z[ijm1 + ix + 1];
                bzt += w10 * flds.b_z[ij + ix - 1];
                bzt += w11 * flds.b_z[ij + ix];
                bzt += w12 * flds.b_z[ij + ix + 1];
                bzt += w20 * flds.b_z[ijp1 + ix - 1];
                bzt += w21 * flds.b_z[ijp1 + ix];
                bzt += w22 * flds.b_z[ijp1 + ix + 1];
            }

            ext *= self.beta;
            eyt *= self.beta;
            ezt *= self.beta;
            bxt *= self.alpha;
            byt *= self.alpha;
            bzt *= self.alpha;
            //  Now, the Boris push:
            ux = *px + ext;
            uy = *py + eyt;
            uz = *pz + ezt;
            gt = (1. + (ux * ux + uy * uy + uz * uz) * csqinv)
                .sqrt()
                .powi(-1);

            bxt *= gt;
            byt *= gt;
            bzt *= gt;

            boris = 2.0 * (1.0 + bxt * bxt + byt * byt + bzt * bzt).powi(-1);

            uxt = ux + uy * bzt - uz * byt;
            uyt = uy + uz * bxt - ux * bzt;
            uzt = uz + ux * byt - uy * bxt;

            *px = ux + boris * (uyt * bzt - uzt * byt) + ext;
            *py = uy + boris * (uzt * bxt - uxt * bzt) + eyt;
            *pz = uz + boris * (uxt * byt - uyt * bxt) + ezt;

            *psa = (1.0 + (*px * *px + *py * *py + *pz * *pz) * csqinv).sqrt()
        }
    }
}
