#[macro_use]
extern crate criterion;

use criterion::Criterion;
//use criterion::black_box;
//use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_distr::Standard;
//use ndarray::prelude::*;
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
    static ref DENS: usize = 2; // # of prlts per species per cell
    static ref GAMMA_INJ: f32 = 15.0; // Speed of upstream flow
    static ref BETA_INJ: f32 = f32::sqrt(1.-f32::powf(*GAMMA_INJ,-2.));
    static ref PRTL_NUM: usize = *DENS * ( *SIZE_X - 2* *DELTA) * *SIZE_Y;
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
    fn add_species (&mut self, charge: f32, mass: f32, vth: f32) {
        let beta = charge * 0.5 * mass * *DT;
        let alpha = charge * 0.5 * mass * *DT / *C;
        let mut prtl = Prtl {
            ix: vec![0; *PRTL_NUM],
            dx: vec![0f32; *PRTL_NUM],
            iy: vec![0; *PRTL_NUM],
            dy: vec![0f32; *PRTL_NUM],
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
fn new_sim() -> Sim {
    let sim = Sim {
            e_x: vec![0f32; (*SIZE_Y + 2) * (2 + *SIZE_X)], // 3 Ghost zones. 1 at 0, 2 at SIZE_X
            e_y: vec![0f32; (*SIZE_Y + 2) * (2 + *SIZE_X)],
            e_z: vec![0f32; (*SIZE_Y + 2) * (2 + *SIZE_X)],
            b_x: vec![0f32; (*SIZE_Y + 2) * (2 + *SIZE_X)],
            b_y: vec![0f32; (*SIZE_Y + 2) * (2 + *SIZE_X)],
            b_z: vec![0f32; (*SIZE_Y + 2) * (2 + *SIZE_X)],
            j_x: vec![0f32; (*SIZE_Y + 2) * (2 + *SIZE_X)],
            j_y: vec![0f32; (*SIZE_Y + 2) * (2 + *SIZE_X)],
            j_z: vec![0f32; (*SIZE_Y + 2) * (2 + *SIZE_X)],
            prtls: Vec::<Prtl>::new(),
            t: 0,
        };
    sim
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
    fn apply_bc(&mut self){
        // PERIODIC BOUNDARIES IN Y
        // First iterate over y array and apply BC
        for iy in self.iy.iter_mut() {
            if *iy < 1 {
                *iy += *SIZE_Y;
            } else if *iy >= *SIZE_Y {
                *iy -= *SIZE_Y;
            }
        }
        // Now iterate over x array
        let c1 = *SIZE_X - *DELTA;
        let c2 = 2 * c1;
        // Let len = std::cmp::min(xs.len(), pxs.len());
        for (ix, px) in self.ix.iter_mut().zip(self.px.iter_mut()) {
             if *ix >= c1 {
                 *ix = c2 - *ix;
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
                    self.dx[c1+k]= r1 + (j as f32);
                    self.dy[c1+k]= r1 + (i as f32);


                }
                c1 += *DENS;
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
        let mut ijm1: usize; let mut ijp1: usize; let mut ij: usize;

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
            ij = iy * (2 + *SIZE_X); ijm1 *= 2 + *SIZE_X; ijp1 *= 2 + *SIZE_X;
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
            unsafe {
                ext = w00 * ex.get_unchecked(ij + ix - 1);
                ext += w01 * ex.get_unchecked(ij + ix);
                ext += w02 * ex.get_unchecked(ij + ix + 1);
                ext += w10 * ex.get_unchecked(ijm1 + ix - 1);
                ext += w11 * ex.get_unchecked(ijm1 + ix);
                ext += w12 * ex.get_unchecked(ijm1 + ix + 1);
                ext += w20 * ex.get_unchecked(ijp1 + ix - 1);
                ext += w21 * ex.get_unchecked(ijp1 + ix);
                ext += w22 * ex.get_unchecked(ijp1 + ix + 1);

                eyt = w00 * ey.get_unchecked(ij + ix - 1);
                eyt += w01 * ey.get_unchecked(ij + ix);
                eyt += w02 * ey.get_unchecked(ij + ix + 1);
                eyt += w10 * ey.get_unchecked(ijm1 + ix - 1);
                eyt += w11 * ey.get_unchecked(ijm1 + ix);
                eyt += w12 * ey.get_unchecked(ijm1 + ix + 1);
                eyt += w20 * ey.get_unchecked(ijp1 + ix - 1);
                eyt += w21 * ey.get_unchecked(ijp1 + ix);
                eyt += w22 * ey.get_unchecked(ijp1 + ix + 1);

                ezt = w00 * ez.get_unchecked(ij + ix - 1);
                ezt += w01 * ez.get_unchecked(ij + ix);
                ezt += w02 * ez.get_unchecked(ij + ix + 1);
                ezt += w10 * ez.get_unchecked(ijm1 + ix - 1);
                ezt += w11 * ez.get_unchecked(ijm1 + ix);
                ezt += w12 * ez.get_unchecked(ijm1 + ix + 1);
                ezt += w20 * ez.get_unchecked(ijp1 + ix - 1);
                ezt += w21 * ez.get_unchecked(ijp1 + ix);
                ezt += w22 * ez.get_unchecked(ijp1 + ix + 1);

                ext *= self.beta; eyt *= self.beta; ezt *= self.beta;

                bxt = w00 * bx.get_unchecked(ij + ix - 1);
                bxt += w01 * bx.get_unchecked(ij + ix);
                bxt += w02 * bx.get_unchecked(ij + ix + 1);
                bxt += w10 * bx.get_unchecked(ijm1 + ix - 1);
                bxt += w11 * bx.get_unchecked(ijm1 + ix);
                bxt += w12 * bx.get_unchecked(ijm1 + ix + 1);
                bxt += w20 * bx.get_unchecked(ijp1 + ix - 1);
                bxt += w21 * bx.get_unchecked(ijp1 + ix);
                bxt += w22 * bx.get_unchecked(ijp1 + ix + 1);

                byt = w00 * by.get_unchecked(ij + ix - 1);
                byt += w01 * by.get_unchecked(ij + ix);
                byt += w02 * by.get_unchecked(ij + ix + 1);
                byt += w10 * by.get_unchecked(ijm1 + ix - 1);
                byt += w11 * by.get_unchecked(ijm1 + ix);
                byt += w12 * by.get_unchecked(ijm1 + ix + 1);
                byt += w20 * by.get_unchecked(ijp1 + ix - 1);
                byt += w21 * by.get_unchecked(ijp1 + ix);
                byt += w22 * by.get_unchecked(ijp1 + ix + 1);

                bzt = w00 * bz.get_unchecked(ij + ix - 1);
                bzt += w01 * bz.get_unchecked(ij + ix);
                bzt += w02 * bz.get_unchecked(ij + ix + 1);
                bzt += w10 * bz.get_unchecked(ijm1 + ix - 1);
                bzt += w11 * bz.get_unchecked(ijm1 + ix);
                bzt += w12 * bz.get_unchecked(ijm1 + ix + 1);
                bzt += w20 * bz.get_unchecked(ijp1 + ix - 1);
                bzt += w21 * bz.get_unchecked(ijp1 + ix);
                bzt += w22 * bz.get_unchecked(ijp1 + ix + 1);

                bxt *= self.alpha; byt *= self.alpha; bzt *= self.alpha;
            }
            //  Now, the Boris push:
            ux = *px + ext;
            uy = *py + eyt;
            uz = *pz + ezt;
            pt = ux * ux + uy * uy + uz * uz;
            gt = (1. + pt * *CSQINV ).sqrt().powi(-1);

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

            *psa = (1.0 + (*px * *px + *py * *py + *pz * *pz) * *CSQINV).sqrt()
        }
    }
    fn deposit_current (&self, jx: &mut Vec::<f32>, jy: &mut Vec::<f32>, jz: &mut Vec::<f32>) {
        // local vars we will use
        let mut ij: usize; let mut ijm1: usize; let mut ijp1: usize;

        // for the weights
        let mut w00: f32; let mut w01: f32; let mut w02: f32;
        let mut w10: f32; let mut w11: f32; let mut w12: f32;
        let mut w20: f32; let mut w21: f32; let mut w22: f32;

        let mut vx: f32; let mut vy: f32; let mut vz: f32;
        let mut psa_inv: f32;

        for (ix, iy, dx, dy, px, py, pz, psa) in izip!(&self.ix, &self.iy, &self.dx, &self.dy, &self.px, &self.py, &self.pz, &self.psa) {
            ijm1 = iy - 1;
            ijp1 = iy + 1;
            //if ix1 >= *SIZE_X {
            //    ix1 -= *SIZE_X;
            //    ix2 -= *SIZE_X;
            //} else if ix2 >= *SIZE_X {
            //    ix2 -= *SIZE_X;
            //}
            ij = iy *  (2 + *SIZE_X); ijm1 *= 2 + *SIZE_X; ijp1 *= 2 + *SIZE_X;
            psa_inv = psa.powf(-1.0);
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
            unsafe {
                *jx.get_unchecked_mut(ijm1 + ix -1) += w00 * vx;
                *jx.get_unchecked_mut(ijm1 + ix) += w01 * vx;
                *jx.get_unchecked_mut(ijm1 + ix + 1) += w02 * vx;
                *jx.get_unchecked_mut(ij + ix - 1) += w10 * vx;
                *jx.get_unchecked_mut(ij + ix) += w11 * vx;
                *jx.get_unchecked_mut(ij + ix + 1) += w12 * vx;
                *jx.get_unchecked_mut(ijp1 + ix - 1) += w20 * vx;
                *jx.get_unchecked_mut(ijp1 + ix) += w21 * vx;
                *jx.get_unchecked_mut(ijp1 + ix + 1) += w22 * vx;

                *jy.get_unchecked_mut(ijm1 + ix - 1) += w00 * vy;
                *jy.get_unchecked_mut(ijm1 + ix) += w01 * vy;
                *jy.get_unchecked_mut(ijm1 + ix + 1) += w02 * vy;
                *jy.get_unchecked_mut(ij + ix - 1) += w10 * vy;
                *jy.get_unchecked_mut(ij + ix) += w11 * vy;
                *jy.get_unchecked_mut(ij + ix + 1) += w12 * vy;
                *jy.get_unchecked_mut(ijp1 + ix - 1) += w20 * vy;
                *jy.get_unchecked_mut(ijp1 + ix) += w21 * vy;
                *jy.get_unchecked_mut(ijp1 + ix + 1) += w22 * vy;

                *jz.get_unchecked_mut(ijm1 + ix - 1) += w00 * vz;
                *jz.get_unchecked_mut(ijm1 + ix) += w01 * vz;
                *jz.get_unchecked_mut(ijm1 + ix + 1) += w02 * vz;
                *jz.get_unchecked_mut(ij + ix - 1) += w10 * vz;
                *jz.get_unchecked_mut(ij + ix) += w11 * vz;
                *jz.get_unchecked_mut(ij + ix + 1) += w12 * vz;
                *jz.get_unchecked_mut(ijp1 + ix - 1) += w20 * vz;
                *jz.get_unchecked_mut(ijp1 + ix) += w21 * vz;
                *jz.get_unchecked_mut(ijp1 + ix + 1) += w22 * vz;
            }
        }
    }
    fn move_and_deposit(&mut self,  jx: &mut Vec::<f32>, jy: &mut Vec::<f32>, jz: &mut Vec::<f32>) {
        // FIRST we update positions of particles
        let mut c1: f32;
        for (ix, iy, dx, dy, px, py, psa) in izip!(&mut self.ix, &mut self.iy, &mut self.dx, &mut self.dy, & self.px, & self.py, & self.psa) {
            c1 =  0.5 * *DT * psa.powi(-1);
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
        self.apply_bc();


        // Deposit currents
        self.deposit_current(jx, jy, jz);

        // UPDATE POS AGAIN!
        for (ix, iy, dx, dy, px, py, psa) in izip!(&mut self.ix, &mut self.iy, &mut self.dx, &mut self.dy, & self.px, & self.py, & self.psa) {
            c1 =  0.5 * *DT * psa.powi(-1);
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
        self.apply_bc();

        // # CALCULATE DENSITY
        //calculateDens(self.x, self.y, self.dsty)#, self.charge)
        //self.sim.dsty += self.charge*self.dsty
    }
}


fn fibonacci() {
    let mut sim = new_sim();
    sim.add_species(1.0, 1.0, 1E-3);
    sim.add_species(-1.0, 1.0, 1E-3);
    sim.run();
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
