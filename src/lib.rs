use rand::prelude::*;
use rand_distr::Standard;
use rand_distr::StandardNormal;
use serde::Deserialize;
use std::fs;

use anyhow::{Context, Result};
use itertools::izip;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;

// We use a type alias for f64/Float to easily support
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
    pub track_prtls: bool,
    pub write_output: bool,
    pub track_interval: u32,
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

pub struct Fld {
    pub spatial: Vec<Float>,
    pub spectral: Vec<Complex<Float>>,
}

impl Fld {
    #[inline(always)]
    fn copy_to_spectral(&mut self, sim: &Sim) -> () {
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
}

impl Config {
    pub fn new() -> Result<Config> {
        let contents =
            fs::read_to_string("config.toml").context("Could not open the config.toml file")?;
        toml::from_str(&contents).with_context(|| "Could not parse Config file")
    }
}

pub fn run(cfg: Config) -> Result<()> {
    // the number of cells must be even for the fft algorithm to work.
    if cfg.params.size_x % 2 != 0 {
        return Err(anyhow::Error::msg(
            "Number of cells in x direction must be even",
        ));
    }
    if cfg.params.size_y % 2 != 0 {
        return Err(anyhow::Error::msg(
            "Number of cells in y direction must be even",
        ));
    }

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
        Vec::<Float>::with_capacity((sim.t_final / cfg.output.output_interval) as usize);
    let mut y_track =
        Vec::<Float>::with_capacity((sim.t_final / cfg.output.output_interval) as usize);
    let mut gam_track =
        Vec::<Float>::with_capacity((sim.t_final / cfg.output.output_interval) as usize);
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
                        x_track.push((*ix as Float + *dx) / sim.c);
                        y_track.push((*iy as Float + *dy) / sim.c);
                        gam_track.push(*psa);
                    }
                }
            }
        }
        */

        // Zero out currents and density
        println!("{}", t);
        for fld in &mut [
            &mut flds.j_x.spatial,
            &mut flds.j_y.spatial,
            &mut flds.j_z.spatial,
            &mut flds.dsty.spatial,
        ] {
            for v in fld.iter_mut() {
                *v = 0.0;
            }
        }
        println!("moving & dep prtl");

        // deposit current. This part is finished.
        for prtl in prtls.iter_mut() {
            sim.move_and_deposit(prtl, &mut flds);
        }

        // solve field. This part is NOT finished
        println!("solving fields");
        flds.update(&sim);

        // push prtls finished
        println!("pushing prtl");
        for prtl in prtls.iter_mut() {
            prtl.boris_push(&sim, &flds)
        }

        //calc Density. This part is finished
        for prtl in prtls.iter_mut() {
            sim.calc_density(prtl, &mut flds);
        }

        sim.t.set(t);
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

#[inline(always)]
fn binomial_filter_2_d(sim: &Sim, in_vec: &mut Vec<Float>, wrkspace: &mut Vec<Float>) {
    // wrkspace should be same size as fld
    if !cfg!(feature = "unchecked") {
        assert!(in_vec.len() == wrkspace.len());
        assert!(in_vec.len() == (sim.size_x + 2) * (sim.size_y + 2));
    }
    let sim_size = in_vec.len();

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
        for j in 1..sim.size_x + 1 {
            wrkspace[j] = wrkspace[sim.size_y * sim.size_x + j];
            wrkspace[(sim.size_y + 1) * sim.size_x + j] = wrkspace[sim.size_x + j];
        }
        // handle ghost zones in the corners
        wrkspace[0] = wrkspace[sim.size_y * (sim.size_x + 2) - 1];
        wrkspace[sim_size - 1] = wrkspace[sim.size_x + 3];
        wrkspace[sim_size - sim.size_x - 2] = wrkspace[2 * (sim.size_x + 2) - 1];
        wrkspace[sim.size_x + 1] = wrkspace[(sim.size_y - 1) * (sim.size_x + 2)];

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
        for j in 1..sim.size_x + 1 {
            in_vec[j] = in_vec[sim.size_y * sim.size_x + j];
            in_vec[(sim.size_y + 1) * sim.size_x + j] = in_vec[sim.size_x + j];
        }
        // handle ghost zones in the corners
        in_vec[0] = in_vec[sim.size_y * (sim.size_x + 2) - 1];
        in_vec[sim_size - 1] = in_vec[sim.size_x + 3];
        in_vec[sim_size - sim.size_x - 2] = in_vec[2 * (sim.size_x + 2) - 1];
        in_vec[sim.size_x + 1] = in_vec[(sim.size_y - 1) * (sim.size_x + 2)];
    }
}

struct Flds {
    // The struct that holds all the fields.
    // First off is all the regular fields.
    e_x: Fld,
    e_y: Fld,
    e_z: Fld,
    b_x: Fld,
    b_y: Fld,
    b_z: Fld,
    j_x: Fld,
    j_y: Fld,
    j_z: Fld,
    k_x: Vec<Float>,
    k_y: Vec<Float>,
    k_norm: Vec<Float>,
    b_x2: Vec<Complex<Float>>,
    b_y2: Vec<Complex<Float>>,
    b_z2: Vec<Complex<Float>>,
    fft_x: std::sync::Arc<dyn rustfft::Fft<Float>>,
    ifft_x: std::sync::Arc<dyn rustfft::Fft<Float>>,
    fft_y: std::sync::Arc<dyn rustfft::Fft<Float>>,
    ifft_y: std::sync::Arc<dyn rustfft::Fft<Float>>,
    fft_x_buf: Vec<Complex<Float>>,
    fft_y_buf: Vec<Complex<Float>>,
    real_wrkspace_ghosts: Vec<Float>,
    real_wrkspace: Vec<Float>,
    cmp_wrkspace: Vec<Complex<Float>>,
    dsty: Fld,
}
impl Flds {
    fn new(sim: &Sim) -> Flds {
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
            e_x: Fld {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            e_y: Fld {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            e_z: Fld {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            b_x: Fld {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            b_y: Fld {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            b_z: Fld {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            j_x: Fld {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            j_y: Fld {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            j_z: Fld {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            dsty: Fld {
                spatial: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
                spectral: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            },
            k_x: vec![0.0; sim.size_x],
            k_y: vec![0.0; sim.size_y],
            k_norm: vec![0.0; sim.size_y * sim.size_x],
            fft_x,
            ifft_x,
            fft_y,
            ifft_y,
            fft_x_buf: xscratch,
            fft_y_buf: yscratch,
            real_wrkspace_ghosts: vec![0.0; (sim.size_y + 2) * (sim.size_x + 2)],
            real_wrkspace: vec![0.0; (sim.size_y) * (sim.size_x)],
            cmp_wrkspace: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_x2: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_y2: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            b_z2: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
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

    pub fn fft2d(
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
        let ckc = sim.dt / sim.dens as Float;
        let cdt = sim.dt * sim.c as Float;

        // Filter currents and density fields
        binomial_filter_2_d(sim, &mut self.j_x.spatial, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.j_y.spatial, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.j_z.spatial, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.dsty.spatial, &mut self.real_wrkspace_ghosts);

        // copy the flds to the complex arrays to perform ffts;
        self.copy_spatial_to_spectral(sim);

        // Take fft of currents
        for current in &[
            &mut self.j_x.spectral,
            &mut self.j_y.spectral,
            &mut self.j_z.spectral,
            &mut self.dsty.spectral,
        ] {
            Flds::fft2d(
                self.fft_x.clone(),
                self.fft_y.clone(),
                sim,
                &mut self.j_x.spectral,
                &mut self.cmp_wrkspace,
                &mut self.fft_x_buf,
                &mut self.fft_y_buf,
            );
        }

        // copy previous timestep
        for (b_copy, b_prev) in &[
            (&mut self.b_x2, &self.b_x.spectral),
            (&mut self.b_y2, &self.b_y.spectral),
            (&mut self.b_z2, &self.b_z.spectral),
        ] {
            for (b2, br) in b_copy.iter_mut().zip(b_prev.iter()) {
                *b2 = *br;
            }
        }
        // Take fft of electric fields
    }
}
struct Sim {
    t: std::cell::Cell<u32>,
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
            t: std::cell::Cell::new(0),
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

        // The [i,j] position in the array. Slightly complicated because
        // Using a 1d vec to represent 2D array for speed.
        // Here is the layout if it were a 2d array
        // --------------------------------
        // | ijm1 - 1 |  ijm1  | ijm1 + 1 |
        // --------------------------------
        // |  ij - 1  |   ij   |  ij + 1  |
        // --------------------------------
        // | ijp1 - 1 |  ijp1  | ijp1 + 1 |
        // --------------------------------

        let mut ij: usize;
        let mut ijm1: usize;
        let mut ijp1: usize;

        // Similarly for the weights
        // -------------------
        // | w00 | w01 | w02 |
        // -------------------
        // | w10 | w11 | w12 |
        // -------------------
        // | w20 | w21 | w22 |
        // -------------------

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
            // to ensure ijm1 doesn't underflow
            if !cfg!(feature = "uchecked") {
                assert!(*iy > 0);
                assert!(*ix > 0);
            }
            ijm1 = iy - 1;
            ijp1 = iy + 1;

            ij = iy * (2 + self.size_x); // 2 because 1 ghost zone on each side
            ijm1 *= 2 + self.size_x;
            ijp1 *= 2 + self.size_x;
            psa_inv = psa.powi(-1);
            vx = prtl.charge * px * psa_inv;
            vy = prtl.charge * py * psa_inv;
            vz = prtl.charge * pz * psa_inv;
            // CALC WEIGHTS
            // 2nd order
            w00 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 - dx) * (0.5 - dx);
            w01 = 0.5 * (0.5 - dy) * (0.5 - dy) * (0.75 - dx * dx);
            w02 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5 + dx);
            w10 = (0.75 - dy * dy) * 0.5 * (0.5 - dx) * (0.5 - dx);
            w11 = (0.75 - dy * dy) * (0.75 - dx * dx);
            w12 = (0.75 - dy * dy) * 0.5 * (0.5 + dx) * (0.5 + dx);
            w20 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 - dx) * (0.5 - dx);
            w21 = 0.5 * (0.5 + dy) * (0.5 + dy) * (0.75 - dx * dx);
            w22 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 + dx) * (0.5 + dx);

            // Deposit the CURRENT
            let j_x = &mut flds.j_x.spatial;
            let j_y = &mut flds.j_y.spatial;
            let j_z = &mut flds.j_z.spatial;
            if !cfg!(feature = "unchecked") {
                // Safe because we will assert that ijp1 + ix + 1 < len(flds.j_x)

                assert!(ijp1 + ix + 1 < j_x.len());
            }
            unsafe {
                *j_x.get_unchecked_mut(ijm1 + ix - 1) += w00 * vx;
                *j_x.get_unchecked_mut(ijm1 + ix) += w01 * vx;
                *j_x.get_unchecked_mut(ijm1 + ix + 1) += w02 * vx;
                *j_x.get_unchecked_mut(ij + ix - 1) += w10 * vx;
                *j_x.get_unchecked_mut(ij + ix) += w11 * vx;
                *j_x.get_unchecked_mut(ij + ix + 1) += w12 * vx;
                *j_x.get_unchecked_mut(ijp1 + ix - 1) += w20 * vx;
                *j_x.get_unchecked_mut(ijp1 + ix) += w21 * vx;
                *j_x.get_unchecked_mut(ijp1 + ix + 1) += w22 * vx;

                *j_y.get_unchecked_mut(ijm1 + ix - 1) += w00 * vy;
                *j_y.get_unchecked_mut(ijm1 + ix) += w01 * vy;
                *j_y.get_unchecked_mut(ijm1 + ix + 1) += w02 * vy;
                *j_y.get_unchecked_mut(ij + ix - 1) += w10 * vy;
                *j_y.get_unchecked_mut(ij + ix) += w11 * vy;
                *j_y.get_unchecked_mut(ij + ix + 1) += w12 * vy;
                *j_y.get_unchecked_mut(ijp1 + ix - 1) += w20 * vy;
                *j_y.get_unchecked_mut(ijp1 + ix) += w21 * vy;
                *j_y.get_unchecked_mut(ijp1 + ix + 1) += w22 * vy;

                *j_z.get_unchecked_mut(ijm1 + ix - 1) += w00 * vz;
                *j_z.get_unchecked_mut(ijm1 + ix) += w01 * vz;
                *j_z.get_unchecked_mut(ijm1 + ix + 1) += w02 * vz;
                *j_z.get_unchecked_mut(ij + ix - 1) += w10 * vz;
                *j_z.get_unchecked_mut(ij + ix) += w11 * vz;
                *j_z.get_unchecked_mut(ij + ix + 1) += w12 * vz;
                *j_z.get_unchecked_mut(ijp1 + ix - 1) += w20 * vz;
                *j_z.get_unchecked_mut(ijp1 + ix) += w21 * vz;
                *j_z.get_unchecked_mut(ijp1 + ix + 1) += w22 * vz;
            }
            /* Bounds checked version
             *} else {
                j_x[ijm1 + ix - 1] += w00 * vx;
                j_x[ijm1 + ix] += w01 * vx;
                j_x[ijm1 + ix + 1] += w02 * vx;
                j_x[ij + ix - 1] += w10 * vx;
                j_x[ij + ix] += w11 * vx;
                j_x[ij + ix + 1] += w12 * vx;
                j_x[ijp1 + ix - 1] += w20 * vx;
                j_x[ijp1 + ix] += w21 * vx;
                j_x[ijp1 + ix + 1] += w22 * vx;

                j_y[ijm1 + ix - 1] += w00 * vy;
                j_y[ijm1 + ix] += w01 * vy;
                j_y[ijm1 + ix + 1] += w02 * vy;
                j_y[ij + ix - 1] += w10 * vy;
                j_y[ij + ix] += w11 * vy;
                j_y[ij + ix + 1] += w12 * vy;
                j_y[ijp1 + ix - 1] += w20 * vy;
                j_y[ijp1 + ix] += w21 * vy;
                j_y[ijp1 + ix + 1] += w22 * vy;

                j_z[ijm1 + ix - 1] += w00 * vz;
                j_z[ijm1 + ix] += w01 * vz;
                j_z[ijm1 + ix + 1] += w02 * vz;
                j_z[ij + ix - 1] += w10 * vz;
                j_z[ij + ix] += w11 * vz;
                j_z[ij + ix + 1] += w12 * vz;
                j_z[ijp1 + ix - 1] += w20 * vz;
                j_z[ijp1 + ix] += w21 * vz;
                j_z[ijp1 + ix + 1] += w22 * vz;
            }
            */
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
            if !cfg!(feature = "unchecked") {
                assert!(*iy > 0);
                assert!(*ix > 0);
            }
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

            let dsty = &mut flds.dsty.spatial;
            // Deposit the density
            if !cfg!(feature = "unchecked") {
                // safe because following assertion
                assert!(ijp1 + ix + 1 < dsty.len());
            }
            unsafe {
                *dsty.get_unchecked_mut(ijm1 + ix - 1) += w00 * prtl.charge;
                *dsty.get_unchecked_mut(ijm1 + ix) += w01 * prtl.charge;
                *dsty.get_unchecked_mut(ijm1 + ix + 1) += w02 * prtl.charge;
                *dsty.get_unchecked_mut(ij + ix - 1) += w10 * prtl.charge;
                *dsty.get_unchecked_mut(ij + ix) += w11 * prtl.charge;
                *dsty.get_unchecked_mut(ij + ix + 1) += w12 * prtl.charge;
                *dsty.get_unchecked_mut(ijp1 + ix - 1) += w20 * prtl.charge;
                *dsty.get_unchecked_mut(ijp1 + ix) += w21 * prtl.charge;
                *dsty.get_unchecked_mut(ijp1 + ix + 1) += w22 * prtl.charge;
            }
            /*
             * bounds checked version. not needed because of asssert above
                dsty[ijm1 + ix - 1] += w00 * prtl.charge;
                dsty[ijm1 + ix] += w01 * prtl.charge;
                dsty[ijm1 + ix + 1] += w02 * prtl.charge;
                dsty[ij + ix - 1] += w10 * prtl.charge;
                dsty[ij + ix] += w11 * prtl.charge;
                dsty[ij + ix + 1] += w12 * prtl.charge;
                dsty[ijp1 + ix - 1] += w20 * prtl.charge;
                dsty[ijp1 + ix] += w21 * prtl.charge;
                dsty[ijp1 + ix + 1] += w22 * prtl.charge;
            */
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
            if !cfg!(feature = "unchecked") {
                // check that we don't underflow
                assert!(*ix > 0);
                assert!(*iy > 0);
            }
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
            if !cfg!(feature = "unchecked") {
                // check that we don't underflow
                assert!(*ix > 0);
                assert!(*iy > 0);
            }
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
fn _fld2prtl(sim: &Sim, ix: usize, iy: usize, dx: Float, dy: Float, fld: &Vec<Float>) -> Float {
    // this function has been replaced with a fully inline one for
    // performance reasons. Leaving here for historical reasons.
    if !cfg!(feature = "unchecked") {
        //avoid underflow
        assert!(iy > 0);
    }
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
            charge,
            vth,
            alpha,
            beta,
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
                        // let r1: Float = rng.sample(Standard);
                        // let r2: Float = rng.sample(Standard);
                        // self.x[c1+k]= r1 + (j as Float);
                        // self.y[c1+k]= r2 + (i as Float);

                        // UNIFORM OPT
                        self.iy[c1 + k] = i + 1; // +1 for ghost zone
                        self.ix[c1 + k] = j + 1; // +1 for ghost zone

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
        // let mut pt: Float;
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
            if !cfg!(feature = "unchecked") {
                assert!(*iy > 0);
                assert!(*ix > 0);
            }
            ijm1 = iy - 1;
            ijp1 = iy + 1;
            ij = iy * (2 + sim.size_x);
            ijm1 *= 2 + sim.size_x;
            ijp1 *= 2 + sim.size_x;
            // CALC WEIGHTS
            // 2nd order
            // The weighting scheme prtl is in middle
            // -------------------
            // | w00 | w01 | w02 |
            // -------------------
            // | w10 | w11 | w12 |
            // -------------------
            // | w20 | w21 | w22 |
            // -------------------
            w00 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 - dx) * (0.5 - dx);
            w01 = 0.5 * (0.5 - dy) * (0.5 - dy) * (0.75 - dx * dx);
            w02 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5 + dx);
            w10 = (0.75 - dy * dy) * 0.5 * (0.5 - dx) * (0.5 - dx);
            w11 = (0.75 - dy * dy) * (0.75 - dx * dx);
            w12 = (0.75 - dy * dy) * 0.5 * (0.5 + dx) * (0.5 + dx);
            w20 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 - dx) * (0.5 - dx);
            w21 = 0.5 * (0.5 + dy) * (0.5 + dy) * (0.75 - dx * dx);
            w22 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 + dx) * (0.5 + dx);

            // get direct pointers to avoid unnecessary lookups
            let e_x = &flds.e_x.spatial;
            let e_y = &flds.e_y.spatial;
            let e_z = &flds.e_z.spatial;

            let b_x = &flds.b_x.spatial;
            let b_y = &flds.b_y.spatial;
            let b_z = &flds.b_z.spatial;

            // INTERPOLATE ALL THE FIELDS
            if !cfg!(feature = "unchecked") {
                assert!(ijp1 + ix + 1 < e_x.len());
            }
            // safe because of following assertion

            unsafe {
                ext = w00 * e_x.get_unchecked(ijm1 + ix - 1);
                ext += w01 * e_x.get_unchecked(ijm1 + ix);
                ext += w02 * e_x.get_unchecked(ijm1 + ix + 1);
                ext += w10 * e_x.get_unchecked(ij + ix - 1);
                ext += w11 * e_x.get_unchecked(ij + ix);
                ext += w12 * e_x.get_unchecked(ij + ix + 1);
                ext += w20 * e_x.get_unchecked(ijp1 + ix - 1);
                ext += w21 * e_x.get_unchecked(ijp1 + ix);
                ext += w22 * e_x.get_unchecked(ijp1 + ix + 1);

                eyt = w00 * e_y.get_unchecked(ijm1 + ix - 1);
                eyt += w01 * e_y.get_unchecked(ijm1 + ix);
                eyt += w02 * e_y.get_unchecked(ijm1 + ix + 1);
                eyt += w10 * e_y.get_unchecked(ij + ix - 1);
                eyt += w11 * e_y.get_unchecked(ij + ix);
                eyt += w12 * e_y.get_unchecked(ij + ix + 1);
                eyt += w20 * e_y.get_unchecked(ijp1 + ix - 1);
                eyt += w21 * e_y.get_unchecked(ijp1 + ix);
                eyt += w22 * e_y.get_unchecked(ijp1 + ix + 1);

                ezt = w00 * e_z.get_unchecked(ijm1 + ix - 1);
                ezt += w01 * e_z.get_unchecked(ijm1 + ix);
                ezt += w02 * e_z.get_unchecked(ijm1 + ix + 1);
                ezt += w10 * e_z.get_unchecked(ij + ix - 1);
                ezt += w11 * e_z.get_unchecked(ij + ix);
                ezt += w12 * e_z.get_unchecked(ij + ix + 1);
                ezt += w20 * e_z.get_unchecked(ijp1 + ix - 1);
                ezt += w21 * e_z.get_unchecked(ijp1 + ix);
                ezt += w22 * e_z.get_unchecked(ijp1 + ix + 1);

                bxt = w00 * b_x.get_unchecked(ijm1 + ix - 1);
                bxt += w01 * b_x.get_unchecked(ijm1 + ix);
                bxt += w02 * b_x.get_unchecked(ijm1 + ix + 1);
                bxt += w10 * b_x.get_unchecked(ij + ix - 1);
                bxt += w11 * b_x.get_unchecked(ij + ix);
                bxt += w12 * b_x.get_unchecked(ij + ix + 1);
                bxt += w20 * b_x.get_unchecked(ijp1 + ix - 1);
                bxt += w21 * b_x.get_unchecked(ijp1 + ix);
                bxt += w22 * b_x.get_unchecked(ijp1 + ix + 1);

                byt = w00 * b_y.get_unchecked(ijm1 + ix - 1);
                byt += w01 * b_y.get_unchecked(ijm1 + ix);
                byt += w02 * b_y.get_unchecked(ijm1 + ix + 1);
                byt += w10 * b_y.get_unchecked(ij + ix - 1);
                byt += w11 * b_y.get_unchecked(ij + ix);
                byt += w12 * b_y.get_unchecked(ij + ix + 1);
                byt += w20 * b_y.get_unchecked(ijp1 + ix - 1);
                byt += w21 * b_y.get_unchecked(ijp1 + ix);
                byt += w22 * b_y.get_unchecked(ijp1 + ix + 1);

                bzt = w00 * b_z.get_unchecked(ijm1 + ix - 1);
                bzt += w01 * b_z.get_unchecked(ijm1 + ix);
                bzt += w02 * b_z.get_unchecked(ijm1 + ix + 1);
                bzt += w10 * b_z.get_unchecked(ij + ix - 1);
                bzt += w11 * b_z.get_unchecked(ij + ix);
                bzt += w12 * b_z.get_unchecked(ij + ix + 1);
                bzt += w20 * b_z.get_unchecked(ijp1 + ix - 1);
                bzt += w21 * b_z.get_unchecked(ijp1 + ix);
                bzt += w22 * b_z.get_unchecked(ijp1 + ix + 1);
            }
            /* bounds checked verion. leaving for posterity

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
            */
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
