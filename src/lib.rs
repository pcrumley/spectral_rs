use serde::{Deserialize};
use std::fs;
use std::error::Error;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_distr::Standard;

#[macro_use] extern crate itertools;
//#[macro_use(array)] extern crate ndarray;

#[derive(Deserialize)]
pub struct Config {
    pub params: Params,
    pub setup: Setup
}

#[derive(Deserialize)]
pub struct Setup {
    pub t_final: u32,
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
pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    //let contents = fs::read_to_string(Sconfig.params.n_pass)?;

    let sim = Sim::new(config);
    let mut prtls = Vec::<Prtl>::new();
    // Add ions to prtls list
    prtls.push(Prtl::new(&sim, 1.0, 1.0, 1E-3));
    // Add lecs to prtls list
    prtls.push(Prtl::new(&sim, 1.0, 1.0, 1E-3));
    let mut flds = Flds::new(&sim);
    for t in 1 .. sim.t_final + 1 {
        // Zero out currents
        for (jx, jy, jz) in izip!(&mut flds.j_x, &mut flds.j_y, &mut flds.j_z) {
            *jx = 0.; *jy = 0.; *jz = 0.;
        }

        // deposit currents
        for prtl in prtls.iter_mut(){
            sim.move_and_deposit(prtl, &mut flds);
        }

        // solve field
        // self.fieldSolver()

        // push prtls

        for prtl in prtls.iter_mut(){
            prtl.boris_push(&sim, &flds);
        }

        // let sim.t = t;
    }


    Ok(())
}

fn binomial_filter_2_d(sim: &Sim,in_vec: &mut Vec::<f32>, wrkspace: &mut Vec::<f32>) {
    // wrkspace should be same size as fld
    let weights: [f32; 3] = [0.25, 0.5, 0.25];
    // account for ghost zones
    // FIRST FILTER IN X-DIRECTION
    for _ in 0 .. sim.n_pass {
        for i in (1 .. (sim.size_y + 1)*(sim.size_x + 2)).step_by(sim.size_x+ 2) {
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
        for i in (1 .. (sim.size_y + 1)*(sim.size_x + 2)).step_by(sim.size_x + 2) {
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
}
impl Flds {
    fn new(sim: &Sim) ->  Flds {
        Flds {
            e_x: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)], // 2 Ghost zones. 1 at 0, 1 at SIZE_X
            e_y: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            e_z: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            b_x: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            b_y: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            b_z: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            j_x: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            j_y: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)],
            j_z: vec![0f32; (sim.size_y + 2) * (sim.size_x + 2)]
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
    dt: f32,
    c: f32,
    dens: u32,
    gamma_inj: f32, // Speed of upstream flow
    prtl_num: usize, // = *DENS * ( *SIZE_X - 2* *DELTA) * *SIZE_Y;
    n_pass: u32 // = 4; //Number of filter passes
}

impl Sim {
    fn new(cfg: Config) ->  Sim {
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
            psa_inv = psa.powf(-1.0);
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
            w00 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 - dx) * (0.5-dx); // y0
            w01 = 0.5 * (0.5 - dy) * (0.5 - dy) * (0.75 - dx * dx); // y0
            w02 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5+dx); // y0
            w10 = (0.75 - dy * dy) * 0.5 * (0.5 - dx) * (0.5-dx); // y0
            w11 = (0.75 - dy * dy) * (0.75 - dx * dx); // y0
            w12 = (0.75 - dy * dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5+dx); // y0
            w20 = 0.5 * (0.5 + dy) * (0.5 - dy) * 0.5 * (0.5 - dx) * (0.5-dx); // y0
            w21 = 0.5 * (0.5 + dy) * (0.5 - dy) * (0.75 - dx * dx); // y0
            w22 = 0.5 * (0.5 + dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5+dx); // y0

            // Deposit the CURRENT
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
}

impl Prtl {
    fn new (sim: &Sim, charge: f32, mass: f32, vth: f32) -> Prtl {
        let beta = charge * 0.5 * mass * sim.dt;
        let alpha = charge * 0.5 * mass * sim.dt / sim.c;
        let mut prtl = Prtl {
            ix: vec![0; sim.prtl_num],
            dx: vec![0f32; sim.prtl_num],
            iy: vec![0; sim.prtl_num],
            dy: vec![0f32; sim.prtl_num],
            px: vec![0f32; sim.prtl_num],
            py: vec![0f32; sim.prtl_num],
            pz: vec![0f32; sim.prtl_num],
            psa: vec![0f32; sim.prtl_num],
            charge: charge,
            vth: vth,
            alpha: alpha,
            beta: beta
        };
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
        // x boundary conditions are incorrect
        let c1 = sim.size_x - sim.delta;
        let c2 = 2 * c1;
        // Let len = std::cmp::min(xs.len(), pxs.len());
        for (ix, px) in self.ix.iter_mut().zip(self.px.iter_mut()) {
             if *ix >= c1 {
                 *ix = c2 - *ix;
                 *px *= -1.0;
             }
         }
    }
    fn initialize_positions(&mut self, sim: &Sim) {
        // A method to calculate the initial, non-random
        // position of the particles
        let mut c1 = 0;
        // let mut rng = thread_rng();
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


                }
                c1 += sim.dens as usize;
                // helper_arr = -.5+0.25+np.arange(dens)*.5
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
        let mut rng = thread_rng();
        let beta_inj = f32::sqrt(1.-sim.gamma_inj.powi(-2));
        let csqinv = 1./(sim.c * sim.c);
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
            w12 = (0.75 - dy * dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0
            w20 = 0.5 * (0.5 + dy) * (0.5 - dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w21 = 0.5 * (0.5 + dy) * (0.5 - dy) * (0.75 - dx * dx); // y0
            w22 = 0.5 * (0.5 + dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0

            // INTERPOLATE ALL THE FIELDS
            if cfg!(feature = "unsafe") {
                unsafe {
                    ext = w00 * flds.e_x.get_unchecked(ij + ix - 1);
                    ext += w01 * flds.e_x.get_unchecked(ij + ix);
                    ext += w02 * flds.e_x.get_unchecked(ij + ix + 1);
                    ext += w10 * flds.e_x.get_unchecked(ijm1 + ix - 1);
                    ext += w11 * flds.e_x.get_unchecked(ijm1 + ix);
                    ext += w12 * flds.e_x.get_unchecked(ijm1 + ix + 1);
                    ext += w20 * flds.e_x.get_unchecked(ijp1 + ix - 1);
                    ext += w21 * flds.e_x.get_unchecked(ijp1 + ix);
                    ext += w22 * flds.e_x.get_unchecked(ijp1 + ix + 1);

                    eyt = w00 * flds.e_y.get_unchecked(ij + ix - 1);
                    eyt += w01 * flds.e_y.get_unchecked(ij + ix);
                    eyt += w02 * flds.e_y.get_unchecked(ij + ix + 1);
                    eyt += w10 * flds.e_y.get_unchecked(ijm1 + ix - 1);
                    eyt += w11 * flds.e_y.get_unchecked(ijm1 + ix);
                    eyt += w12 * flds.e_y.get_unchecked(ijm1 + ix + 1);
                    eyt += w20 * flds.e_y.get_unchecked(ijp1 + ix - 1);
                    eyt += w21 * flds.e_y.get_unchecked(ijp1 + ix);
                    eyt += w22 * flds.e_y.get_unchecked(ijp1 + ix + 1);

                    ezt = w00 * flds.e_z.get_unchecked(ij + ix - 1);
                    ezt += w01 * flds.e_z.get_unchecked(ij + ix);
                    ezt += w02 * flds.e_z.get_unchecked(ij + ix + 1);
                    ezt += w10 * flds.e_z.get_unchecked(ijm1 + ix - 1);
                    ezt += w11 * flds.e_z.get_unchecked(ijm1 + ix);
                    ezt += w12 * flds.e_z.get_unchecked(ijm1 + ix + 1);
                    ezt += w20 * flds.e_z.get_unchecked(ijp1 + ix - 1);
                    ezt += w21 * flds.e_z.get_unchecked(ijp1 + ix);
                    ezt += w22 * flds.e_z.get_unchecked(ijp1 + ix + 1);

                    bxt = w00 * flds.b_x.get_unchecked(ij + ix - 1);
                    bxt += w01 * flds.b_x.get_unchecked(ij + ix);
                    bxt += w02 * flds.b_x.get_unchecked(ij + ix + 1);
                    bxt += w10 * flds.b_x.get_unchecked(ijm1 + ix - 1);
                    bxt += w11 * flds.b_x.get_unchecked(ijm1 + ix);
                    bxt += w12 * flds.b_x.get_unchecked(ijm1 + ix + 1);
                    bxt += w20 * flds.b_x.get_unchecked(ijp1 + ix - 1);
                    bxt += w21 * flds.b_x.get_unchecked(ijp1 + ix);
                    bxt += w22 * flds.b_x.get_unchecked(ijp1 + ix + 1);

                    byt = w00 * flds.b_y.get_unchecked(ij + ix - 1);
                    byt += w01 * flds.b_y.get_unchecked(ij + ix);
                    byt += w02 * flds.b_y.get_unchecked(ij + ix + 1);
                    byt += w10 * flds.b_y.get_unchecked(ijm1 + ix - 1);
                    byt += w11 * flds.b_y.get_unchecked(ijm1 + ix);
                    byt += w12 * flds.b_y.get_unchecked(ijm1 + ix + 1);
                    byt += w20 * flds.b_y.get_unchecked(ijp1 + ix - 1);
                    byt += w21 * flds.b_y.get_unchecked(ijp1 + ix);
                    byt += w22 * flds.b_y.get_unchecked(ijp1 + ix + 1);

                    bzt = w00 * flds.b_z.get_unchecked(ij + ix - 1);
                    bzt += w01 * flds.b_z.get_unchecked(ij + ix);
                    bzt += w02 * flds.b_z.get_unchecked(ij + ix + 1);
                    bzt += w10 * flds.b_z.get_unchecked(ijm1 + ix - 1);
                    bzt += w11 * flds.b_z.get_unchecked(ijm1 + ix);
                    bzt += w12 * flds.b_z.get_unchecked(ijm1 + ix + 1);
                    bzt += w20 * flds.b_z.get_unchecked(ijp1 + ix - 1);
                    bzt += w21 * flds.b_z.get_unchecked(ijp1 + ix);
                    bzt += w22 * flds.b_z.get_unchecked(ijp1 + ix + 1);
                }
            } else {
                ext = w00 * flds.e_x[ij + ix - 1];
                ext += w01 * flds.e_x[ij + ix];
                ext += w02 * flds.e_x[ij + ix + 1];
                ext += w10 * flds.e_x[ijm1 + ix - 1];
                ext += w11 * flds.e_x[ijm1 + ix];
                ext += w12 * flds.e_x[ijm1 + ix + 1];
                ext += w20 * flds.e_x[ijp1 + ix - 1];
                ext += w21 * flds.e_x[ijp1 + ix];
                ext += w22 * flds.e_x[ijp1 + ix + 1];

                eyt = w00 * flds.e_y[ij + ix - 1];
                eyt += w01 * flds.e_y[ij + ix];
                eyt += w02 * flds.e_y[ij + ix + 1];
                eyt += w10 * flds.e_y[ijm1 + ix - 1];
                eyt += w11 * flds.e_y[ijm1 + ix];
                eyt += w12 * flds.e_y[ijm1 + ix + 1];
                eyt += w20 * flds.e_y[ijp1 + ix - 1];
                eyt += w21 * flds.e_y[ijp1 + ix];
                eyt += w22 * flds.e_y[ijp1 + ix + 1];

                ezt = w00 * flds.e_z[ij + ix - 1];
                ezt += w01 * flds.e_z[ij + ix];
                ezt += w02 * flds.e_z[ij + ix + 1];
                ezt += w10 * flds.e_z[ijm1 + ix - 1];
                ezt += w11 * flds.e_z[ijm1 + ix];
                ezt += w12 * flds.e_z[ijm1 + ix + 1];
                ezt += w20 * flds.e_z[ijp1 + ix - 1];
                ezt += w21 * flds.e_z[ijp1 + ix];
                ezt += w22 * flds.e_z[ijp1 + ix + 1];

                bxt = w00 * flds.b_x[ij + ix - 1];
                bxt += w01 * flds.b_x[ij + ix];
                bxt += w02 * flds.b_x[ij + ix + 1];
                bxt += w10 * flds.b_x[ijm1 + ix - 1];
                bxt += w11 * flds.b_x[ijm1 + ix];
                bxt += w12 * flds.b_x[ijm1 + ix + 1];
                bxt += w20 * flds.b_x[ijp1 + ix - 1];
                bxt += w21 * flds.b_x[ijp1 + ix];
                bxt += w22 * flds.b_x[ijp1 + ix + 1];

                byt = w00 * flds.b_y[ij + ix - 1];
                byt += w01 * flds.b_y[ij + ix];
                byt += w02 * flds.b_y[ij + ix + 1];
                byt += w10 * flds.b_y[ijm1 + ix - 1];
                byt += w11 * flds.b_y[ijm1 + ix];
                byt += w12 * flds.b_y[ijm1 + ix + 1];
                byt += w20 * flds.b_y[ijp1 + ix - 1];
                byt += w21 * flds.b_y[ijp1 + ix];
                byt += w22 * flds.b_y[ijp1 + ix + 1];

                bzt = w00 * flds.b_z[ij + ix - 1];
                bzt += w01 * flds.b_z[ij + ix];
                bzt += w02 * flds.b_z[ij + ix + 1];
                bzt += w10 * flds.b_z[ijm1 + ix - 1];
                bzt += w11 * flds.b_z[ijm1 + ix];
                bzt += w12 * flds.b_z[ijm1 + ix + 1];
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
            pt = ux * ux + uy * uy + uz * uz;
            gt = (1. + pt * csqinv).sqrt().powi(-1);

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
