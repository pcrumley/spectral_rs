use serde::{Deserialize};
use std::fs;
use std::error::Error;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_distr::Standard;
const PI:f32 = std::f32::consts::PI;
use std::sync::Arc;
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

#[macro_use]
extern crate npy_derive;
extern crate npy;

#[macro_use] extern crate itertools;

#[derive(Deserialize)]
pub struct Config {
    pub params: Params,
    pub setup: Setup,
    pub output: Output
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
    pub dt: f32,
    pub c: f32,
    pub dens: u32,
    pub gamma_inj: f32,
    pub n_pass: u32,
}
impl Config {
    pub fn new() ->  Result<Config, &'static str> {
        let contents = fs::read_to_string("config.toml")
           .expect("Something went wrong reading the config.toml file");
        let config: Config = toml::from_str(&contents).unwrap();
        Ok( config )
    }
}
pub fn run(cfg: Config) -> Result<(), Box<dyn Error>> {
    //let contents = fs::read_to_string(Sconfig.params.n_pass)?;
    let sim = Sim::new(&cfg);
    let mut prtls = Vec::<Prtl>::new();
    // Add ions to prtls list
    println!("initialzing  prtls");
    prtls.push(Prtl::new(&sim, 1.0, 1.0, 1E-3));
    // Add lecs to prtls list
    prtls.push(Prtl::new(&sim, -1.0, 1.0, 1E-3));

    let mut flds = Flds::new(&sim);
    let mut x_track = Vec::<f32>::with_capacity((sim.t_final/cfg.output.output_interval) as usize);
    let mut y_track = Vec::<f32>::with_capacity((sim.t_final/cfg.output.output_interval) as usize);
    let mut gam_track = Vec::<f32>::with_capacity((sim.t_final/cfg.output.output_interval) as usize);
    for t in 0 .. sim.t_final + 1 {
        if cfg.output.write_output {
            if t % cfg.output.output_interval == 0 {
                fs::create_dir_all(format!("output/dat_{:04}", t/cfg.output.output_interval))?;
                println!("saving prtls");
                let x: Vec::<f32> = prtls[0].ix.iter()
                        .zip(prtls[0].dx.iter())
                        .step_by(cfg.output.stride)
                        .map(|(&ix, &dx)| ix as f32 + dx)
                        .collect();

                npy::to_file(format!("output/dat_{:04}/x.npy", t/cfg.output.output_interval), x).unwrap();
                let y: Vec::<f32> = prtls[0].iy.iter()
                            .zip(prtls[0].dy.iter())
                            .step_by(cfg.output.stride)
                            .map(|(&iy, &dy)| iy as f32 + dy)
                            .collect();
                npy::to_file(format!("output/dat_{:04}/y.npy", t/cfg.output.output_interval), y).unwrap();
                npy::to_file(format!("output/dat_{:04}/u.npy", t/cfg.output.output_interval),
                    prtls[0].px.clone().iter()
                    .step_by(cfg.output.stride)
                    .map(|&x| x/sim.c)).unwrap();
                npy::to_file(format!("output/dat_{:04}/gam.npy", t/cfg.output.output_interval),
                        prtls[0].psa.clone()).unwrap();
            }
        }
        if cfg.output.track_prtls {
            if t % cfg.output.track_interval == 0 {
                for (ix, iy, dx, dy, track, psa) in izip!(&prtls[0].ix, &prtls[0].iy, &prtls[0].dx, &prtls[0].dy, &prtls[0].track, &prtls[0].psa){
                    if *track {

                        x_track.push((*ix as f32 + *dx)/sim.c);
                        y_track.push((*iy as f32 + *dy)/sim.c);
                        gam_track.push(*psa);
                    }
                }

            }
        }
        // Zero out currents
        println!("{}", t);
        for (jx, jy, jz, dsty) in izip!(&mut flds.j_x, &mut flds.j_y, &mut flds.j_z, &mut flds.dsty) {
            *jx = 0.; *jy = 0.; *jz = 0.; *dsty =0.;
        }
        println!("moving & dep prtl");
        // deposit currents
        for prtl in prtls.iter_mut(){
            sim.move_and_deposit(prtl, &mut flds);
        }

        // solve field
        println!("solving fields");
        flds.update(&sim);

        // push prtls
        println!("pushing prtl");
        for prtl in prtls.iter_mut(){
            prtl.boris_push(&sim, &flds);
        }

        //calc Density
        for prtl in prtls.iter_mut(){
            sim.calc_density(prtl, &mut flds);
        }

        // let sim.t = t;

    }
    if cfg.output.track_prtls {
        fs::create_dir_all("output/trckd_prtl/")?;
        npy::to_file("output/trckd_prtl/x.npy", x_track)?;
        npy::to_file("output/trckd_prtl/y.npy", y_track)?;
        npy::to_file("output/trckd_prtl/psa.npy", gam_track)?;
    }
    Ok(())
}


fn binomial_filter_2_d(sim: &Sim,in_vec: &mut Vec::<f32>, wrkspace: &mut Vec::<f32>) {
    // wrkspace should be same size as fld
    let weights: [f32; 3] = [0.25, 0.5, 0.25];
    // account for ghost zones
    // FIRST FILTER IN X-DIRECTION
    for _ in 0 .. sim.n_pass {
        for i in ((sim.size_x + 2) .. (sim.size_y + 1)*(sim.size_x + 2)).step_by(sim.size_x+ 2) {
            for j in 1 .. sim.size_x + 1 {
                wrkspace[i] = weights.iter()
                                .zip(&in_vec[i + j - 1 .. i + j + 1])
                                .map(|(&w, &f)| w * f)
                                .sum::<f32>();
            }
        // handle the ghost zones in x direction
        wrkspace[i - 1] = wrkspace[i + sim.size_x];
        wrkspace[i + sim.size_x + 1 ] = wrkspace[i];
    }
    // handle the ghost zones in y direction
    // I COULD DO THIS WITH MEMCPY AND I KNOW IT IS POSSIBLE WITH SAFE
    // RUST BUT I DON'T KNOW HOW :(

        for j in 0 .. sim.size_x + 2 {
            wrkspace[j] = wrkspace[sim.size_y * sim.size_x  + j];
            wrkspace[(sim.size_y + 1) * sim.size_x  + j] = wrkspace[sim.size_x  + j];
        }
        // NOW FILTER IN Y-DIRECTION AND PUT VALS IN in_vec
        for i in ((sim.size_x + 2) .. (sim.size_y + 1)*(sim.size_x + 2)).step_by(sim.size_x + 2) {
            for j in 1 .. sim.size_x + 1 {
                in_vec[i] = weights.iter()
                                .zip(wrkspace[i + j - (sim.size_x + 2) .. i + j + (sim.size_x + 2)].iter().step_by(sim.size_x + 2))
                                .map(|(&w, &f)| w * f)
                                .sum::<f32>();
            }
            // handle the ghost zones in x direction
            in_vec[i - 1] = in_vec[i + sim.size_x];
            in_vec[i + sim.size_x + 1 ] = in_vec[i];
        }
        // handle the ghost zones in y direction
        // I COULD DO THIS WITH MEMCPY AND I KNOW IT IS POSSIBLE WITH SAFE
        // RUST BUT I DON'T KNOW HOW :(

        for j in 0 .. sim.size_x + 2 {
            in_vec[j] = in_vec[sim.size_y * sim.size_x  + j];
            in_vec[(sim.size_y + 1) * sim.size_x  + j] = in_vec[sim.size_x  + j];
        }
    }
}
struct Flds {
    e_x: Vec<f32>,
    e_y: Vec<f32>,
    e_z: Vec<f32>,
    b_x: Vec<f32>,
    b_y: Vec<f32>,
    b_z: Vec<f32>,
    j_x: Vec<f32>,
    j_y: Vec<f32>,
    j_z: Vec<f32>,
    k_x: Vec<f32>,
    k_y: Vec<f32>,
    k_norm: Vec<f32>,
    fft_x: std::sync::Arc<dyn rustfft::FFT<f32>>,
    ifft_x: std::sync::Arc<dyn rustfft::FFT<f32>>,
    fft_y: std::sync::Arc<dyn rustfft::FFT<f32>>,
    ifft_y: std::sync::Arc<dyn rustfft::FFT<f32>>,
    real_wrkspace_ghosts: Vec<f32>,
    real_wrkspace: Vec<f32>,
    cmp_wrkspace: Vec<Complex<f32>>,
    c_x: Vec<Complex<f32>>,
    c_y: Vec<Complex<f32>>,
    c_z: Vec<Complex<f32>>,
    dsty_cmp: Vec<Complex<f32>>,
    dsty: Vec<f32>
}
impl Flds {
    fn new(sim: &Sim) ->  Flds {
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
        let mut input:  Vec<Complex<f32>> = vec![Complex::zero(); sim.size_x];
        let mut output: Vec<Complex<f32>> = vec![Complex::zero(); sim.size_x];
        //let ifft_x = inv_planner.plan_fft(sim.size_x);
        //let fft_y = planner.plan_fft(sim.size_y);
        //let ifft_y = inv_planner.plan_fft(sim.size_y);
        fft_x.process(&mut input, &mut output);

        let mut f = Flds {
            e_x: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)], // 2 Ghost zones. 1 at 0, 1 at SIZE_X
            e_y: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            e_z: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            b_x: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            b_y: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            b_z: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            j_x: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            j_y: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            j_z: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            k_x: vec![0f32; sim.size_x],
            k_y: vec![0f32; sim.size_y],
            k_norm: vec![0f32; sim.size_y * sim.size_x],
            fft_x: fft_x,
            ifft_x: ifft_x,
            fft_y: fft_y,
            ifft_y: ifft_y,
            real_wrkspace_ghosts: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            real_wrkspace: vec![0f32; (sim.size_y) * (sim.size_x)],
            cmp_wrkspace: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            c_x: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            c_y: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            c_z: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
            dsty: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            dsty_cmp: vec![Complex::zero(); (sim.size_y) * (sim.size_x)],
        };
        // Build the k basis of FFT
        for i in 0 .. f.k_x.len() {
            f.k_x[i] = i as f32;
            if i > sim.size_x/2 +1 {
                f.k_x[i] -= sim.size_x as f32;
            }
            f.k_x[i] *= 2f32 * PI / (sim.size_x as f32);
        }
        for i in 0 .. f.k_y.len() {
            f.k_y[i] = i as f32;
            if i > sim.size_y/2 +1 {
                f.k_y[i] -= sim.size_y as f32;
            }
            f.k_y[i] *= 2f32 * PI / (sim.size_y as f32);
        }
        // Make the norm:
        for i in 0 .. f.k_y.len() {
            for j in 0 .. f.k_x.len() {
                f.k_norm[i*sim.size_x + j] = 1./(f.k_x[j] * f.k_x[j] + f.k_y[i] * f.k_y[i]);
            }
        }

        if false {
            for i in 0 .. (sim.size_y + 2) {
                //let tmp_b = 2f32 * Bnorm * (sim.size_y/2 - (i - 1))/(sim.size_y as f32);
                let tmp_b = 0f32;
                for j in 0.. (sim.size_x + 2) {

                    if i > (sim.size_y)/6  && i < 2 * sim.size_y/6 {

                        f.e_y[i*(sim.size_x+2)+j] = -0.9 * tmp_b;
                        f.b_z[i*(sim.size_x+2)+j] = tmp_b;
                    }
                    if i > 2*(sim.size_y)/3  && i < 5 * sim.size_y/6 {
                        f.e_y[i*(sim.size_x+2)+j] = 0.9 * tmp_b;
                        f.b_z[i*(sim.size_x+2)+j] = -tmp_b;
                    }
                }
            }
        }
        f
    }
    pub fn transpose(sim: &Sim, in_fld: &Vec<Complex<f32>>, out_fld: &mut Vec<Complex<f32>>) {
        for i in 0..sim.size_y {
            for j in 0..sim.size_x {
                if cfg!(feature = "unsafe") {
                    unsafe {
                        *out_fld.get_unchecked_mut(i*sim.size_x+j) = *in_fld.get_unchecked(j*sim.size_y+i);
                    }
                } else {
                    out_fld[i*sim.size_x+j] = in_fld[j*sim.size_y+i];
                }
            }
        }
    }
    pub fn fft2d(fft_x: std::sync::Arc<dyn rustfft::FFT<f32>>, fft_y: std::sync::Arc<dyn rustfft::FFT<f32>>, sim: &Sim, fld: &mut Vec<Complex<f32>>, wrk_space: &mut Vec<Complex<f32>>) {
        for iy in (0 .. sim.size_y*sim.size_x).step_by(sim.size_x) {
            fft_x.process(&mut fld[iy..iy + sim.size_x], &mut wrk_space[iy..iy + sim.size_x]);
        }
        Flds::transpose(sim, wrk_space, fld);
        for iy in (0 .. sim.size_x*sim.size_y).step_by(sim.size_y) {
            fft_y.process(&mut fld[iy..iy + sim.size_y], &mut wrk_space[iy..iy +sim.size_y]);
        }
        Flds::transpose(sim, wrk_space, fld);

    }
    pub fn update(&mut self, sim: &Sim) {
        let ckc = sim.dt / sim.dens as f32;
        let cdt = sim.dt * sim.c as f32;
        // Filter fields
        binomial_filter_2_d(sim, &mut self.j_x, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.j_y, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.j_z, &mut self.real_wrkspace_ghosts);
        binomial_filter_2_d(sim, &mut self.dsty, &mut self.real_wrkspace_ghosts);
        // copy j_x, j_y, j_z, dsty into complex vector
        let mut ij: usize; let mut ij_ghosts: usize;
        for iy in 0 .. sim.size_y {
            ij = iy * (sim.size_x);
            ij_ghosts = (iy + 1) * (sim.size_x + 2);
            for ix in 0 .. sim.size_x {
                self.c_x[ij+ix].re = self.j_x[ij_ghosts + ix + 1];
                self.c_x[ij+ix].im = 0.0;
                self.c_y[ij+ix].re = self.j_y[ij_ghosts + ix + 1];
                self.c_y[ij+ix].im = 0.0;
                self.c_z[ij+ix].re = self.j_z[ij_ghosts + ix + 1];
                self.c_z[ij+ix].im = 0.0;
                self.dsty_cmp[ij+ix].re = self.dsty[ij_ghosts + ix + 1]/(sim.dens as f32);
                self.dsty_cmp[ij+ix].im = 0.0;
            }
        }
        // Take fft of currents
        Flds::fft2d(self.fft_x.clone(), self.fft_y.clone(), sim, &mut self.c_x, &mut self.cmp_wrkspace);
        Flds::fft2d(self.fft_x.clone(), self.fft_y.clone(), sim, &mut self.c_y, &mut self.cmp_wrkspace);
        Flds::fft2d(self.fft_x.clone(), self.fft_y.clone(), sim, &mut self.c_z, &mut self.cmp_wrkspace);
        Flds::fft2d(self.fft_x.clone(), self.fft_y.clone(), sim, &mut self.dsty_cmp, &mut self.cmp_wrkspace);


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
    dt: f32,
    c: f32,
    dens: u32,
    gamma_inj: f32, // Speed of upstream flow
    prtl_num: usize, // = *DENS * ( *SIZE_X - 2* *DELTA) * *SIZE_Y;
    n_pass: u32 // = 4; //Number of filter passes
}

impl Sim {
    fn new(cfg: &Config) ->  Sim {
        Sim {
            t: 0,
            t_final: cfg.setup.t_final,
            size_x: cfg.params.size_x,
            size_y: cfg.params.size_y,
            delta: cfg.params.delta,
            dt: cfg.params.dt,
            c: cfg.params.c,
            dens: cfg.params.dens,
            gamma_inj: cfg.params.gamma_inj, // Speed of upstream flow
            prtl_num: cfg.params.dens as usize * (cfg.params.size_x - 2 * cfg.params.delta) * cfg.params.size_y,
            //prtl_num: 100,
            n_pass: cfg.params.n_pass
        }
    }

    fn deposit_current (&self, prtl: &Prtl, flds: &mut Flds) {
        // local vars we will use
        let mut ij: usize; let mut ijm1: usize; let mut ijp1: usize;

        // for the weights
        let mut w00: f32; let mut w01: f32; let mut w02: f32;
        let mut w10: f32; let mut w11: f32; let mut w12: f32;
        let mut w20: f32; let mut w21: f32; let mut w22: f32;

        let mut vx: f32; let mut vy: f32; let mut vz: f32;
        let mut psa_inv: f32;

        for (ix, iy, dx, dy, px, py, pz, psa) in izip!(&prtl.ix, &prtl.iy, &prtl.dx, &prtl.dy, &prtl.px, &prtl.py, &prtl.pz, &prtl.psa) {
            ijm1 = iy - 1;
            ijp1 = iy + 1;
            //if ix1 >= *SIZE_X {
            //    ix1 -= *SIZE_X;
            //    ix2 -= *SIZE_X;
            //} else if ix2 >= *SIZE_X {
            //    ix2 -= *SIZE_X;
            //}
            ij = iy *  (2 + self.size_x); ijm1 *= 2 + self.size_x; ijp1 *= 2 + self.size_x;
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
            w10 = (0.75 - dy * dy) * 0.5 * (0.5 - dx) * (0.5-dx); // y0
            w11 = (0.75 - dy * dy) * (0.75 - dx * dx); // y0
            w12 = (0.75 - dy * dy) * 0.5 * (0.5 + dx) * (0.5+dx); // y0
            w20 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w21 = 0.5 * (0.5 + dy) * (0.5 + dy) * (0.75 - dx * dx); // y0
            w22 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0

            // Deposit the CURRENT
            if cfg!(feature = "unsafe") {
                unsafe {
                    *flds.j_x.get_unchecked_mut(ijm1 + ix -1) += w00 * vx;
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
                flds.j_x[ijm1 + ix -1] += w00 * vx;
                flds.j_x[ijm1 + ix] += w01 * vx;
                flds.j_x[ijm1 + ix + 1] += w02 * vx;
                flds.j_x[ij + ix - 1] += w10 * vx;
                flds.j_x[ij + ix] += w11 * vx;
                flds.j_x[ij + ix + 1] += w12 * vx;
                flds.j_x[ijp1 + ix - 1] += w20 * vx;
                flds.j_x[ijp1 + ix] += w21 * vx;
                flds.j_x[ijp1 + ix + 1] += w22 * vx;

                flds.j_y[ijm1 + ix -1] += w00 * vy;
                flds.j_y[ijm1 + ix] += w01 * vy;
                flds.j_y[ijm1 + ix + 1] += w02 * vy;
                flds.j_y[ij + ix - 1] += w10 * vy;
                flds.j_y[ij + ix] += w11 * vy;
                flds.j_y[ij + ix + 1] += w12 * vy;
                flds.j_y[ijp1 + ix - 1] += w20 * vy;
                flds.j_y[ijp1 + ix] += w21 * vy;
                flds.j_y[ijp1 + ix + 1] += w22 * vy;

                flds.j_z[ijm1 + ix -1] += w00 * vz;
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

        fn calc_density (&self, prtl: &Prtl, flds: &mut Flds) {
            // local vars we will use
            let mut ij: usize; let mut ijm1: usize; let mut ijp1: usize;

            // for the weights
            let mut w00: f32; let mut w01: f32; let mut w02: f32;
            let mut w10: f32; let mut w11: f32; let mut w12: f32;
            let mut w20: f32; let mut w21: f32; let mut w22: f32;


            for (ix, iy, dx, dy) in izip!(&prtl.ix, &prtl.iy, &prtl.dx, &prtl.dy) {
                ijm1 = iy - 1;
                ijp1 = iy + 1;
                //if ix1 >= *SIZE_X {
                //    ix1 -= *SIZE_X;
                //    ix2 -= *SIZE_X;
                //} else if ix2 >= *SIZE_X {
                //    ix2 -= *SIZE_X;
                //}
                ij = iy *  (2 + self.size_x); ijm1 *= 2 + self.size_x; ijp1 *= 2 + self.size_x;

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
                w10 = (0.75 - dy * dy) * 0.5 * (0.5 - dx) * (0.5-dx); // y0
                w11 = (0.75 - dy * dy) * (0.75 - dx * dx); // y0
                w12 = (0.75 - dy * dy) * 0.5 * (0.5 + dx) * (0.5+dx); // y0
                w20 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
                w21 = 0.5 * (0.5 + dy) * (0.5 + dy) * (0.75 - dx * dx); // y0
                w22 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0

                // Deposit the CURRENT
                if cfg!(feature = "unsafe") {
                    unsafe {
                        *flds.dsty.get_unchecked_mut(ijm1 + ix -1) += w00 * prtl.charge;
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
                    flds.dsty[ijm1 + ix -1] += w00 * prtl.charge;
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
        let mut c1: f32;
        for (ix, iy, dx, dy, px, py, psa) in izip!(&mut prtl.ix, &mut prtl.iy, &mut prtl.dx, &mut prtl.dy, & prtl.px, & prtl.py, & prtl.psa) {
            c1 =  0.5 * self.dt * psa.powi(-1);
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
        for (ix, iy, dx, dy, px, py, psa) in izip!(&mut prtl.ix, &mut prtl.iy, &mut prtl.dx, &mut prtl.dy, & prtl.px, & prtl.py, & prtl.psa) {
            c1 =  0.5 * self.dt * psa.powi(-1);
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
    dx: Vec<f32>,
    dy: Vec<f32>,
    px: Vec<f32>,
    py: Vec<f32>,
    pz: Vec<f32>,
    psa: Vec<f32>, // Lorentz Factors
    charge: f32,
    alpha: f32,
    beta: f32,
    vth: f32,
    tag: Vec<u64>,
    track: Vec<bool>
}
fn fld2prtl(sim: &Sim, ix: usize, iy: usize, dx: f32, dy: f32, fld: &Vec<f32>) -> f32 {
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
        0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 + dx) * (0.5 + dx)
    ];
    weights.iter()
        .zip(fld[ijm1 - 1 .. ijm1 + 1].into_iter()
            .chain(fld[ij - 1 .. ij + 1].into_iter())
            .chain(fld[ijp1 - 1 .. ijp1 + 1].into_iter()))
        .map(|(&w, &f)| w * f)
        .sum::<f32>()

}

impl Prtl {
    fn new (sim: &Sim, charge: f32, mass: f32, vth: f32) -> Prtl {
        let beta = (charge / mass) * 0.5 * sim.dt;
        let alpha = (charge / mass) * 0.5 * sim.dt / sim.c;
        let mut prtl = Prtl {
            ix: vec![0; sim.prtl_num],
            dx: vec![0f32; sim.prtl_num],
            iy: vec![0; sim.prtl_num],
            dy: vec![0f32; sim.prtl_num],
            px: vec![0f32; sim.prtl_num],
            py: vec![0f32; sim.prtl_num],
            pz: vec![0f32; sim.prtl_num],
            psa: vec![0f32; sim.prtl_num],
            track: vec![false; sim.prtl_num],
            tag: vec![0u64; sim.prtl_num],
            charge: charge,
            vth: vth,
            alpha: alpha,
            beta: beta
        };
        prtl.track[40]=true;
        prtl.initialize_positions(sim);
        prtl.initialize_velocities(sim);
        prtl.apply_bc(sim);
        prtl
    }
    fn apply_bc(&mut self, sim: &Sim){
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
        for i in 0 .. sim.size_y {
            for j in sim.delta .. sim.size_x - sim.delta {
                for k in 0 .. sim.dens as usize {
                    // RANDOM OPT
                    // let r1: f32 = rng.sample(Standard);
                    // let r2: f32 = rng.sample(Standard);
                    // self.x[c1+k]= r1 + (j as f32);
                    // self.y[c1+k]= r2 + (i as f32);

                    // UNIFORM OPT
                    self.iy[c1 + k] = i + 1;
                    self.ix[c1 + k] = j + 1;

                    let mut r1 = 1.0/(2.0 * (sim.dens as f32));
                    r1 = (2.*(k as f32) +1.) * r1;
                    self.dx[c1+k] = r1 - 0.5;
                    self.dy[c1+k] = r1 - 0.5;
                    self.tag[c1+k] = (c1 + k) as u64;

                }
                c1 += sim.dens as usize;
                // helper_arr = -.5+0.25+np.arange(dens)*.5
            }

        }
        } else {
            for i in 0 .. self.iy.len() {
                self.iy[i] = 1 + i * ((sim.size_y as f32 ) / (self.iy.len() as f32)) as usize;
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
        let csqinv = 1./(sim.c * sim.c);
        let beta_inj = -f32::sqrt(1.-sim.gamma_inj.powi(-2));
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
                let rand: f32 = rng.sample(Standard);
                if - beta_inj * ux > rand * *psa {
                    ux *= -1.
                }
                *px = sim.gamma_inj * (ux + beta_inj * *psa); // not p yet... really ux-prime
                *px *= sim.c;
                *psa = 1.0 + (*px * *px + *py * *py + *pz * *pz) * csqinv;
                *psa = psa.sqrt();
            }
        } else {
            for (px, psa) in izip!(&mut self.px, &mut self.psa) {
                *px = -sim.c*sim.gamma_inj*beta_inj;
                *psa = 1.0 + (*px * *px) * csqinv;
                *psa = psa.sqrt();
            }
        }
    }
    fn boris_push(&mut self, sim: &Sim, flds: &Flds) {
        // local vars we will use
        let mut ijm1: usize; let mut ijp1: usize; let mut ij: usize;

        let csqinv = 1./(sim.c * sim.c);
        // for the weights
        let mut w00: f32; let mut w01: f32; let mut w02: f32;
        let mut w10: f32; let mut w11: f32; let mut w12: f32;
        let mut w20: f32; let mut w21: f32; let mut w22: f32;

        let mut ext: f32; let mut eyt: f32; let mut ezt: f32;
        let mut bxt: f32; let mut byt: f32; let mut bzt: f32;
        let mut ux: f32;  let mut uy: f32;  let mut uz: f32;
        let mut uxt: f32;  let mut uyt: f32;  let mut uzt: f32;
        let mut pt: f32; let mut gt: f32; let mut boris: f32;

        for (ix, iy, dx, dy, px, py, pz, psa) in izip!(&self.ix, &self.iy, &self.dx, &self.dy, &mut self.px, &mut self.py, &mut self.pz, &mut self.psa) {
            ijm1 = iy - 1;
            ijp1 = iy + 1;
            ij = iy * (2 + sim.size_x); ijm1 *= 2 + sim.size_x; ijp1 *= 2 + sim.size_x;
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

            ext *= self.beta; eyt *= self.beta; ezt *= self.beta;
            bxt *= self.alpha; byt *= self.alpha; bzt *= self.alpha;
            //  Now, the Boris push:
            ux = *px + ext;
            uy = *py + eyt;
            uz = *pz + ezt;
            gt = (1. + (ux * ux + uy * uy + uz * uz) * csqinv).sqrt().powi(-1);

            bxt *= gt;
            byt *= gt;
            bzt *= gt;

            boris = 2.0 * (1.0 + bxt * bxt + byt * byt + bzt * bzt).powi(-1);

            uxt = ux + uy*bzt - uz*byt;
            uyt = uy + uz*bxt - ux*bzt;
            uzt = uz + ux*byt - uy*bxt;

            *px = ux + boris * (uyt * bzt - uzt * byt) + ext;
            *py = uy + boris * (uzt * bxt - uxt * bzt) + eyt;
            *pz = uz + boris * (uxt * byt - uyt * bxt) + ezt;

            *psa = (1.0 + (*px * *px + *py * *py + *pz * *pz) * csqinv).sqrt()
        }
    }


}
