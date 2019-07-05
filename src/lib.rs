use serde::{Deserialize};
use std::fs;
use std::error::Error;
#[macro_use] extern crate itertools;
#[macro_use] extern crate lazy_static;
//#[macro_use(array)] extern crate ndarray;

lazy_static! {
    static ref SIZE_X: usize = 100;
    static ref SIZE_Y: usize = 100;
    static ref DELTA: usize = 15;
    static ref DT: f32 = 0.1;
    static ref C:  f32 = 3.0; // Don't touch this.
    static ref CSQINV: f32 = 1.0/ (*C * *C);
    static ref DENS: usize = 2; // # of prtls per species per cell
    static ref GAMMA_INJ: f32 = 15.0; // Speed of upstream flow
    static ref BETA_INJ: f32 = f32::sqrt(1.-f32::powi(*GAMMA_INJ, -2));
    static ref PRTL_NUM: usize = *DENS * ( *SIZE_X - 2* *DELTA) * *SIZE_Y;
    static ref N_PASS: usize = 4; //Number of filter passes.
}

#[derive(Deserialize)]
pub struct Config {
    pub params: Params
}

#[derive(Deserialize)]
pub struct Params {
    pub size_x: u32,
    pub size_y: u32,
    pub delta: u32,
    pub dt: f32,
    pub c: f32,
    pub dens: f32,
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
    let mut sim = Sim::new(config);
      sim.add_species(1.0, 1.0, 1E-3);
      sim.add_species(-1.0, 1.0, 1E-3);
      println!["hi"];
      sim.run();

    Ok(())
}



use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_distr::Standard;

fn binomial_filter_2_d(in_vec: &mut Vec::<f32>, wrkspace: &mut Vec::<f32>) {
    // wrkspace should be same size as fld
    let weights: [f32; 3] = [0.25, 0.5, 0.25];
    // account for ghost zones
    // FIRST FILTER IN X-DIRECTION
    for _ in 0 .. *N_PASS {
        for i in (1 .. (*SIZE_Y + 1)*(*SIZE_X + 3)).step_by(*SIZE_X + 3) {
            for j in 1 .. *SIZE_X + 1 {
                wrkspace[i] = weights.iter()
                                .zip(&in_vec[i + j - 1 .. i + j + 1])
                                .map(|(&w, &f)| w * f)
                                .sum::<f32>();
            }
        // handle the ghost zones in x direction
        wrkspace[i - 1] = wrkspace[i + *SIZE_X];
        wrkspace[i + *SIZE_X + 1 ] = wrkspace[i];
        wrkspace[i + *SIZE_X + 1 ] = wrkspace[i + 1];
    }
    // handle the ghost zones in y direction
    // I COULD DO THIS WITH MEMCPY AND I KNOW IT IS POSSIBLE WITH SAFE
    // RUST BUT I DON'T KNOW HOW :(

        for j in 0 .. *SIZE_X + 3 {
            wrkspace[j] = wrkspace[*SIZE_Y * *SIZE_X  + j];
            wrkspace[(*SIZE_Y + 1) * *SIZE_X  + j] =wrkspace[*SIZE_X  + j];
            wrkspace[(*SIZE_Y + 1) * *SIZE_X  + j] =wrkspace[*SIZE_X  + j];
        }
        // NOW FILTER IN Y-DIRECTION AND PUT VALS IN in_vec
        for i in (1 .. (*SIZE_Y + 1)*(*SIZE_X + 3)).step_by(*SIZE_X + 3) {
            for j in 1 .. *SIZE_X + 1 {
                in_vec[i] = weights.iter()
                                .zip(wrkspace[i + j - (*SIZE_X + 3) .. i + j + (*SIZE_X + 3)].iter().step_by(*SIZE_X + 3))
                                .map(|(&w, &f)| w * f)
                                .sum::<f32>();
            }
            // handle the ghost zones in x direction
            in_vec[i - 1] = in_vec[i + *SIZE_X];
            in_vec[i + *SIZE_X + 1 ] = in_vec[i];
            in_vec[i + *SIZE_X + 1 ] = in_vec[i + 1];
        }
        // handle the ghost zones in y direction
        // I COULD DO THIS WITH MEMCPY AND I KNOW IT IS POSSIBLE WITH SAFE
        // RUST BUT I DON'T KNOW HOW :(

        for j in 0 .. *SIZE_X + 3 {
            in_vec[j] = in_vec[*SIZE_Y * *SIZE_X  + j];
            in_vec[(*SIZE_Y + 1) * *SIZE_X  + j] = in_vec[*SIZE_X  + j];
            in_vec[(*SIZE_Y + 1) * *SIZE_X  + j] = in_vec[*SIZE_X  + j];
        }
    }
}

struct Sim {
    e_x: Vec::<f32>,
    e_y: Vec::<f32>,
    e_z: Vec::<f32>,
    b_x: Vec::<f32>,
    b_y: Vec::<f32>,
    b_z: Vec::<f32>,
    j_x: Vec::<f32>,
    j_y: Vec::<f32>,
    j_z: Vec::<f32>,
    prtls: Vec<Prtl>,
    t: u32,
}

impl Sim {
    fn new(_cfg: Config) ->  Sim {
        Sim {
            e_x: vec![0f32; (*SIZE_Y + 3) * (3 + *SIZE_X)], // 3 Ghost zones. 1 at 0, 2 at SIZE_X
            e_y: vec![0f32; (*SIZE_Y + 3) * (3 + *SIZE_X)],
            e_z: vec![0f32; (*SIZE_Y + 3) * (3 + *SIZE_X)],
            b_x: vec![0f32; (*SIZE_Y + 3) * (3 + *SIZE_X)],
            b_y: vec![0f32; (*SIZE_Y + 3) * (3 + *SIZE_X)],
            b_z: vec![0f32; (*SIZE_Y + 3) * (3 + *SIZE_X)],
            j_x: vec![0f32; (*SIZE_Y + 3) * (3 + *SIZE_X)],
            j_y: vec![0f32; (*SIZE_Y + 3) * (3 + *SIZE_X)],
            j_z: vec![0f32; (*SIZE_Y + 3) * (3 + *SIZE_X)],
            prtls: Vec::<Prtl>::new(),
            t: 0,
        }
    }

    fn add_species (&mut self, charge: f32, mass: f32, vth: f32) {
        let beta = charge * 0.5 * mass * *DT;
        let alpha = charge * 0.5 * mass * *DT / *C;
        let mut prtl = Prtl {
            x: vec![0f32; *PRTL_NUM],
            y: vec![0f32; *PRTL_NUM],
            px: vec![0f32; *PRTL_NUM],
            py: vec![0f32; *PRTL_NUM],
            pz: vec![0f32; *PRTL_NUM],
            psa: vec![0f32; *PRTL_NUM],
            charge: charge,
            // mass: mass,
            vth: vth,
            alpha: alpha,
            beta: beta
        };
        prtl.initialize_positions();
        prtl.initialize_velocities();
        prtl.apply_bc();

        self.prtls.push(prtl);
    }
    fn run (&mut self) {
        for _ in 0..10 {
            // Zero out currents
            for (jx, jy, jz) in izip!(&mut self.j_x, &mut self.j_y, &mut self.j_z) {
                *jx = 0.; *jy = 0.; *jz = 0.;
            }

            // deposit currents
            for prtl in self.prtls.iter_mut(){
                prtl.move_and_deposit(&mut self.j_x, &mut self.j_y, &mut self.j_z);
            }


            // solve field
            // self.fieldSolver()

            // push prtls

            for prtl in self.prtls.iter_mut(){
                prtl.boris_push(&self.e_x, &self.e_y, &self.e_z,
                    &self.b_x, &self.b_y, &self.b_z);
            }

            self.t += 1
        }
    }
}

pub struct Prtl {
    x: Vec<f32>,
    y: Vec<f32>,
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
    fn apply_bc(&mut self){
        // PERIODIC BOUNDARIES IN Y
        // First iterate over y array and apply BC
        for y in self.y.iter_mut() {
            if *y < -0.5 {
                *y += *SIZE_Y as f32;
            } else if *y > *SIZE_Y as f32 - 0.5 {
                *y -= *SIZE_Y as f32;
            }
        }
        // Now iterate over x array
        let c1 = (*SIZE_X - *DELTA) as f32 - 0.5;
        let c2 = 2. * c1;
        // Let len = std::cmp::min(xs.len(), pxs.len());
        for (x, px) in self.x.iter_mut().zip(self.px.iter_mut()) {
             if *x > c1 {
                 *x *= -1.0;
                 *x += c2;
                 *px *= -1.0;
             }
         }
    }
    fn initialize_positions(&mut self) {
        // A method to calculate the initial, non-random
        // position of the particles
        let mut c1 = 0;
        // let mut rng = thread_rng();
        for i in 0 .. *SIZE_Y {
            for j in *DELTA .. *SIZE_X - *DELTA {
                for k in 0 .. *DENS {
                    // RANDOM OPT
                    // let r1: f32 = rng.sample(Standard);
                    // let r2: f32 = rng.sample(Standard);
                    // self.x[c1+k]= r1 + (j as f32);
                    // self.y[c1+k]= r2 + (i as f32);

                    // UNIFORM OPT
                    let mut r1 = 1.0/(2.0 * (*DENS as f32));
                    r1 = (2.*(k as f32) +1.) * r1;
                    self.x[c1+k]= r1 + (j as f32);
                    self.y[c1+k]= r1 + (i as f32);


                }
                c1 += *DENS;
            }

        }
    }
    fn initialize_velocities(&mut self) {
        //placeholder
        let mut rng = thread_rng();
        for (px, py, pz, psa) in izip!(&mut self.px, &mut self.py, &mut self.pz, &mut self.psa)
             {
            *px = rng.sample(StandardNormal);
            *px *= self.vth * *C;
            *py = rng.sample(StandardNormal);
            *py *= self.vth * *C;
            *pz = rng.sample(StandardNormal);
            *pz *= self.vth * *C;
            *psa = 1.0 + (*px * *px + *py * *py + *pz * *pz) * *CSQINV;
            *psa = psa.sqrt();

            // Flip the px according to zenitani 2015
            let mut ux = *px / *C;
            let rand: f32 = rng.sample(Standard);
            if - *BETA_INJ * ux > rand * *psa {
                ux *= -1.
            }
            *px = *GAMMA_INJ * (ux + *BETA_INJ * *psa); // not p yet... really ux-prime
            *px *= *C;
            *psa = 1.0 + (*px * *px + *py * *py + *pz * *pz)/(*C * *C);
            *psa = psa.sqrt();
        }

    }
    fn boris_push(&mut self, ex: &Vec::<f32>, ey: &Vec::<f32>, ez: &Vec::<f32>,
        bx: &Vec::<f32>, by: &Vec::<f32>, bz: &Vec::<f32>) {
        // local vars we will use
        let mut ix: usize; let mut dx: f32; let mut iy: usize; let mut dy: f32;
        let mut iy1: usize; let mut iy2: usize;

        // for the weights
        let mut w00: f32; let mut w01: f32; let mut w02: f32;
        let mut w10: f32; let mut w11: f32; let mut w12: f32;
        let mut w20: f32; let mut w21: f32; let mut w22: f32;

        let mut ext: f32; let mut eyt: f32; let mut ezt: f32;
        let mut bxt: f32; let mut byt: f32; let mut bzt: f32;
        let mut ux: f32;  let mut uy: f32;  let mut uz: f32;
        let mut uxt: f32;  let mut uyt: f32;  let mut uzt: f32;
        let mut pt: f32; let mut gt: f32; let mut boris: f32;

        for (x, y, px, py, pz, psa) in izip!(&mut self.x, &mut self.y, &mut self.px, &mut self.py, &mut self.pz, &mut self.psa) {
            dx = *x - x.round();
            ix = x.round() as usize;
            ix += 1;  // +1 to account for ghost cell on left
            dy = *y - y.round();
            iy = y.round() as usize;
            iy += 1;  // +1 to account for ghost cell @ 0
            iy1 = iy + 1;
            iy2 = iy + 2;
            iy *= 3 + *SIZE_X; iy1 *= 3 + *SIZE_X; iy2 *= 3 + *SIZE_X;
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

            // INTERPOLATE ALL THE FIELDS
            if cfg!(feature = "unsafe") {
                unsafe {
                    ext = w00 * ex.get_unchecked(iy + ix);
                    ext += w01 * ex.get_unchecked(iy + ix + 1);
                    ext += w02 * ex.get_unchecked(iy + ix + 2);
                    ext += w10 * ex.get_unchecked(iy1 + ix);
                    ext += w11 * ex.get_unchecked(iy1 + ix + 1);
                    ext += w12 * ex.get_unchecked(iy1 + ix + 2);
                    ext += w20 * ex.get_unchecked(iy2 + ix);
                    ext += w21 * ex.get_unchecked(iy2 + ix + 1);
                    ext += w22 * ex.get_unchecked(iy2 + ix + 2);
                    ext *= self.beta;

                    eyt = w00 * ey.get_unchecked(iy + ix);
                    eyt += w01 * ey.get_unchecked(iy + ix + 1);
                    eyt += w02 * ey.get_unchecked(iy + ix + 2);
                    eyt += w10 * ey.get_unchecked(iy1 + ix);
                    eyt += w11 * ey.get_unchecked(iy1 + ix + 1);
                    eyt += w12 * ey.get_unchecked(iy1 + ix + 2);
                    eyt += w20 * ey.get_unchecked(iy2 + ix);
                    eyt += w21 * ey.get_unchecked(iy2 + ix + 1);
                    eyt += w22 * ey.get_unchecked(iy2 + ix + 2);
                    eyt *= self.beta;

                    ezt = w00 * ez.get_unchecked(iy + ix);
                    ezt += w01 * ez.get_unchecked(iy + ix + 1);
                    ezt += w02 * ez.get_unchecked(iy + ix + 2);
                    ezt += w10 * ez.get_unchecked(iy1 + ix);
                    ezt += w11 * ez.get_unchecked(iy1 + ix + 1);
                    ezt += w12 * ez.get_unchecked(iy1 + ix + 2);
                    ezt += w20 * ez.get_unchecked(iy2 + ix);
                    ezt += w21 * ez.get_unchecked(iy2 + ix + 1);
                    ezt += w22 * ez.get_unchecked(iy2 + ix + 2);
                    ezt *= self.beta;

                    bxt = w00 * bx.get_unchecked(iy + ix);
                    bxt += w01 * bx.get_unchecked(iy + ix + 1);
                    bxt += w02 * bx.get_unchecked(iy + ix + 2);
                    bxt += w10 * bx.get_unchecked(iy1 + ix);
                    bxt += w11 * bx.get_unchecked(iy1 + ix + 1);
                    bxt += w12 * bx.get_unchecked(iy1 + ix + 2);
                    bxt += w20 * bx.get_unchecked(iy2 + ix);
                    bxt += w21 * bx.get_unchecked(iy2 + ix + 1);
                    bxt += w22 * bx.get_unchecked(iy2 + ix + 2);
                    bxt *= self.alpha;

                    byt = w00 * by.get_unchecked(iy + ix);
                    byt += w01 * by.get_unchecked(iy + ix + 1);
                    byt += w02 * by.get_unchecked(iy + ix + 2);
                    byt += w10 * by.get_unchecked(iy1 + ix);
                    byt += w11 * by.get_unchecked(iy1 + ix + 1);
                    byt += w12 * by.get_unchecked(iy1 + ix + 2);
                    byt += w20 * by.get_unchecked(iy2 + ix);
                    byt += w21 * by.get_unchecked(iy2 + ix + 1);
                    byt += w22 * by.get_unchecked(iy2 + ix + 2);
                    byt *= self.alpha;


                    bzt = w00 * bz.get_unchecked(iy + ix);
                    bzt += w01 * bz.get_unchecked(iy + ix + 1);
                    bzt += w02 * bz.get_unchecked(iy + ix + 2);
                    bzt += w10 * bz.get_unchecked(iy1 + ix);
                    bzt += w11 * bz.get_unchecked(iy1 + ix + 1);
                    bzt += w12 * bz.get_unchecked(iy1 + ix + 2);
                    bzt += w20 * bz.get_unchecked(iy2 + ix);
                    bzt += w21 * bz.get_unchecked(iy2 + ix + 1);
                    bzt += w22 * bz.get_unchecked(iy2 + ix + 2);
                    bzt *= self.alpha;
                }
            } else {
                // INTERPOLATE ALL THE FIELDS
                ext = w00 * ex[iy + ix];
                ext += w01 * ex[iy + ix + 1];
                ext += w02 * ex[iy + ix + 2];
                ext += w10 * ex[iy1 + ix];
                ext += w11 * ex[iy1 + ix + 1];
                ext += w12 * ex[iy1 + ix + 2];
                ext += w20 * ex[iy2 + ix];
                ext += w21 * ex[iy2 + ix + 1];
                ext += w22 * ex[iy2 + ix + 2];
                ext *= self.beta;

                eyt = w00 * ey[iy + ix];
                eyt += w01 * ey[iy + ix + 1];
                eyt += w02 * ey[iy + ix + 2];
                eyt += w10 * ey[iy1 + ix];
                eyt += w11 * ey[iy1 + ix + 1];
                eyt += w12 * ey[iy1 + ix + 2];
                eyt += w20 * ey[iy2 + ix];
                eyt += w21 * ey[iy2 + ix + 1];
                eyt += w22 * ey[iy2 + ix + 2];
                eyt *= self.beta;

                ezt = w00 * ez[iy + ix];
                ezt += w01 * ez[iy + ix + 1];
                ezt += w02 * ez[iy + ix + 2];
                ezt += w10 * ez[iy1 + ix];
                ezt += w11 * ez[iy1 + ix + 1];
                ezt += w12 * ez[iy1 + ix + 2];
                ezt += w20 * ez[iy2 + ix];
                ezt += w21 * ez[iy2 + ix + 1];
                ezt += w22 * ez[iy2 + ix + 2];
                ezt *= self.beta;

                bxt = w00 * bx[iy + ix];
                bxt += w01 * bx[iy + ix + 1];
                bxt += w02 * bx[iy + ix + 2];
                bxt += w10 * bx[iy1 + ix];
                bxt += w11 * bx[iy1 + ix + 1];
                bxt += w12 * bx[iy1 + ix + 2];
                bxt += w20 * bx[iy2 + ix];
                bxt += w21 * bx[iy2 + ix + 1];
                bxt += w22 * bx[iy2 + ix + 2];
                bxt *= self.alpha;

                byt = w00 * by[iy + ix];
                byt += w01 * by[iy + ix + 1];
                byt += w02 * by[iy + ix + 2];
                byt += w10 * by[iy1 + ix];
                byt += w11 * by[iy1 + ix + 1];
                byt += w12 * by[iy1 + ix + 2];
                byt += w20 * by[iy2 + ix];
                byt += w21 * by[iy2 + ix + 1];
                byt += w22 * by[iy2 + ix + 2];
                byt *= self.alpha;


                bzt = w00 * bz[iy + ix];
                bzt += w01 * bz[iy + ix + 1];
                bzt += w02 * bz[iy + ix + 2];
                bzt += w10 * bz[iy1 + ix];
                bzt += w11 * bz[iy1 + ix + 1];
                bzt += w12 * bz[iy1 + ix + 2];
                bzt += w20 * bz[iy2 + ix];
                bzt += w21 * bz[iy2 + ix + 1];
                bzt += w22 * bz[iy2 + ix + 2];
                bzt *= self.alpha;
            }
            //  Now, the Boris push:
            ux = *px + ext;
            uy = *py + eyt;
            uz = *pz + ezt;
            pt = ux * ux + uy * uy + uz * uz;
            gt = 1.0 / (1. + pt * *CSQINV ).sqrt();

            bxt *= gt;
            byt *= gt;
            bzt *= gt;

            boris = 2.0 / (1.0 + bxt * bxt + byt * byt + bzt * bzt);

            uxt = ux + uy*bzt - uz*byt;
            uyt = uy + uz*bxt - ux*bzt;
            uzt = uz + ux*byt - uy*bxt;

            *px = ux + boris * (uyt * bzt - uzt * byt) + ext;
            *py = uy + boris * (uzt * bxt - uxt * bzt) + eyt;
            *pz = uz + boris * (uxt * byt - uyt * bxt) + ezt;

            *psa = (1.0 + (*px * *px + *py * *py + *pz * *pz) * *CSQINV).sqrt()
        }
    }
    fn deposit_current (&self, jx: &mut Vec::<f32>, jy: &mut Vec::<f32>, jz: &mut Vec::<f32>) {
        // local vars we will use
        let mut ix: usize; let mut dx: f32; let mut iy: usize; let mut dy: f32;
        let mut iy1: usize; let mut iy2: usize;

        // for the weights
        let mut w00: f32; let mut w01: f32; let mut w02: f32;
        let mut w10: f32; let mut w11: f32; let mut w12: f32;
        let mut w20: f32; let mut w21: f32; let mut w22: f32;

        let mut vx: f32; let mut vy: f32; let mut vz: f32;
        let mut psa_inv: f32;

        for (x, y, px, py, pz, psa) in izip!(&self.x, &self.y, &self.px, &self.py, &self.pz, &self.psa) {
            dx = x - x.round();
            ix = x.round() as usize;
            ix += 1; // +1 to account for ghost cell on left
            dy = y - y.round();
            iy = y.round() as usize;
            iy += 1; // +1 to account for ghost cell at y = 0
            iy1 = iy + 1;
            iy2 = iy + 2;
            //if ix1 >= *SIZE_X {
            //    ix1 -= *SIZE_X;
            //    ix2 -= *SIZE_X;
            //} else if ix2 >= *SIZE_X {
            //    ix2 -= *SIZE_X;
            //}
            iy *= 3 + *SIZE_X; iy1 *= 3 + *SIZE_X; iy2 *= 3 + *SIZE_X;
            psa_inv = psa.powi(-1);
            vx = self.charge * px * psa_inv;
            vy = self.charge * py * psa_inv;
            vz = self.charge * pz * psa_inv;
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
            if cfg!(feature = "unsafe") {
                unsafe {
                    *jx.get_unchecked_mut(iy + ix) += w00 * vx;
                    *jx.get_unchecked_mut(iy + ix + 1) += w01 * vx;
                    *jx.get_unchecked_mut(iy + ix + 2) += w02 * vx;
                    *jx.get_unchecked_mut(iy1 + ix) += w10 * vx;
                    *jx.get_unchecked_mut(iy1 + ix + 1) += w11 * vx;
                    *jx.get_unchecked_mut(iy1 + ix + 2) += w12 * vx;
                    *jx.get_unchecked_mut(iy2 + ix) += w20 * vx;
                    *jx.get_unchecked_mut(iy2 + ix + 1) += w21 * vx;
                    *jx.get_unchecked_mut(iy2 + ix + 2) += w22 * vx;

                    *jy.get_unchecked_mut(iy + ix) += w00 * vy;
                    *jy.get_unchecked_mut(iy + ix + 1) += w01 * vy;
                    *jy.get_unchecked_mut(iy + ix + 2) += w02 * vy;
                    *jy.get_unchecked_mut(iy1 + ix) += w10 * vy;
                    *jy.get_unchecked_mut(iy1 + ix + 1) += w11 * vy;
                    *jy.get_unchecked_mut(iy1 + ix + 2) += w12 * vy;
                    *jy.get_unchecked_mut(iy2 + ix) += w20 * vy;
                    *jy.get_unchecked_mut(iy2 + ix + 1) += w21 * vy;
                    *jy.get_unchecked_mut(iy2 + ix + 2) += w22 * vy;

                    *jz.get_unchecked_mut(iy + ix) += w00 * vz;
                    *jz.get_unchecked_mut(iy + ix + 1) += w01 * vz;
                    *jz.get_unchecked_mut(iy + ix + 2) += w02 * vz;
                    *jz.get_unchecked_mut(iy1 + ix) += w10 * vz;
                    *jz.get_unchecked_mut(iy1 + ix + 1) += w11 * vz;
                    *jz.get_unchecked_mut(iy1 + ix + 2) += w12 * vz;
                    *jz.get_unchecked_mut(iy2 + ix) += w20 * vz;
                    *jz.get_unchecked_mut(iy2 + ix + 1) += w21 * vz;
                    *jz.get_unchecked_mut(iy2 + ix + 2) += w22 * vz;
                }
            } else {
                jx[iy + ix] += w00 * vx;
                jx[iy + ix + 1] += w01 * vx;
                jx[iy + ix + 2] += w02 * vx;
                jx[iy1 + ix] += w10 * vx;
                jx[iy1 + ix + 1] += w11 * vx;
                jx[iy1 + ix + 2] += w12 * vx;
                jx[iy2 + ix] += w20 * vx;
                jx[iy2 + ix + 1] += w21 * vx;
                jx[iy2 + ix + 2] += w22 * vx;

                jy[iy + ix] += w00 * vy;
                jy[iy + ix + 1] += w01 * vy;
                jy[iy + ix + 2] += w02 * vy;
                jy[iy1 + ix] += w10 * vy;
                jy[iy1 + ix + 1] += w11 * vy;
                jy[iy1 + ix + 2] += w12 * vy;
                jy[iy2 + ix] += w20 * vy;
                jy[iy2 + ix + 1] += w21 * vy;
                jy[iy2 + ix + 2] += w22 * vy;

                jz[iy + ix] += w00 * vz;
                jz[iy + ix + 1] += w01 * vz;
                jz[iy + ix + 2] += w02 * vz;
                jz[iy1 + ix] += w10 * vz;
                jz[iy1 + ix + 1] += w11 * vz;
                jz[iy1 + ix + 2] += w12 * vz;
                jz[iy2 + ix] += w20 * vz;
                jz[iy2 + ix + 1] += w21 * vz;
                jz[iy2 + ix + 2] += w22 * vz;
            }
        }
    }
    fn move_and_deposit(&mut self,  jx: &mut Vec::<f32>, jy: &mut Vec::<f32>, jz: &mut Vec::<f32>) {
        // FIRST we update positions of particles
        let mut c1: f32;
        for (x, y, px, py, psa) in izip!(&mut self.x, &mut self.y, & self.px, & self.py, & self.psa) {
            c1 =  0.5 * *DT / psa;
            *x += c1 * px;
            *y += c1 * py;
        }
        //self.dsty *=0
        self.apply_bc();


        // Deposit currents
        self.deposit_current(jx, jy, jz);

        // UPDATE POS AGAIN!
        for (x, y, px, py, psa) in izip!(&mut self.x, &mut self.y, & self.px, & self.py, & self.psa) {
            c1 =  0.5 * *DT / psa;
            *x += c1 * px;
            *y += c1 * py;
        }
        //self.dsty *=0
        self.apply_bc();

        // # CALCULATE DENSITY
        //calculateDens(self.x, self.y, self.dsty)#, self.charge)
        //self.sim.dsty += self.charge*self.dsty
    }
}
