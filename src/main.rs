use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_distr::Standard;
use ndarray::prelude::*;
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
    e_x: Array2::<f32>,
    e_y: Array2::<f32>,
    e_z: Array2::<f32>,
    b_x: Array2::<f32>,
    b_y: Array2::<f32>,
    b_z: Array2::<f32>,
    prtls: Vec<Prtl>
}

impl Sim {
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
            mass: mass,
            vth: vth,
            alpha: alpha,
            beta: beta
        };
        prtl.initialize_positions();
        prtl.initialize_velocities();
        prtl.apply_bc();
        self.prtls.push(prtl);
    }
}
fn new_sim() -> Sim {
    let sim = Sim {
            e_x: Array2::<f32>::zeros((*SIZE_Y, *SIZE_X)),
            e_y: Array2::<f32>::zeros((*SIZE_Y, *SIZE_X)),
            e_z: Array2::<f32>::zeros((*SIZE_Y, *SIZE_X)),
            b_x: Array2::<f32>::zeros((*SIZE_Y, *SIZE_X)),
            b_y: Array2::<f32>::zeros((*SIZE_Y, *SIZE_X)),
            b_z: Array2::<f32>::ones((*SIZE_Y, *SIZE_X)),
            prtls: Vec::<Prtl>::new()
        };
    sim
}
struct Prtl {
    x: Vec<f32>,
    y: Vec<f32>,
    px: Vec<f32>,
    py: Vec<f32>,
    pz: Vec<f32>,
    psa: Vec<f32>, // Lorentz Factors
    charge: f32,
    mass: f32,
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
    fn boris_push(&mut self, ex: &Array2::<f32>, ey: &Array2::<f32>, ez: &Array2::<f32>,
        bx: &Array2::<f32>, by: &Array2::<f32>, bz: &Array2::<f32>) {
        // local vars we will use
        let mut ix: usize; let mut dx: f32; let mut iy: usize; let mut dy: f32;
        let mut iy1: usize; let mut iy2: usize; let mut ix1: usize; let mut ix2: usize;

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
            dy = *y - y.round();
            iy = y.round() as usize;
            iy1 = iy + 1;
            iy2 = iy + 2;
            ix1 = ix + 1;
            ix2 = ix + 2;
            if iy1 >= *SIZE_Y {
                iy1 -= *SIZE_Y;
                iy2 -= *SIZE_Y;
            } else if iy2 >= *SIZE_Y {
                iy2 -= *SIZE_Y;
            }
            if ix1 >= *SIZE_X {
                ix1 -= *SIZE_X;
                ix2 -= *SIZE_X;
            } else if ix2 >= *SIZE_X {
                ix2 -= *SIZE_X;
            }
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
            ext = w00 * ex[[iy, ix]];
            ext += w01 * ex[[iy, ix + 1]];
            ext += w02 * ex[[iy, ix + 2]];
            ext += w10 * ex[[iy + 1, ix]];
            ext += w11 * ex[[iy + 1, ix + 1]];
            ext += w12 * ex[[iy + 1, ix + 2]];
            ext += w20 * ex[[iy + 2, ix]];
            ext += w21 * ex[[iy + 2, ix + 1]];
            ext += w22 * ex[[iy + 2, ix + 2]];
            ext *= self.beta;

            eyt = w00 * ey[[iy, ix]];
            eyt += w01 * ey[[iy, ix + 1]];
            eyt += w02 * ey[[iy, ix + 2]];
            eyt += w10 * ey[[iy + 1, ix]];
            eyt += w11 * ey[[iy + 1, ix + 1]];
            eyt += w12 * ey[[iy + 1, ix + 2]];
            eyt += w20 * ey[[iy + 2, ix]];
            eyt += w21 * ey[[iy + 2, ix + 1]];
            eyt += w22 * ey[[iy + 2, ix + 2]];
            eyt *= self.beta;

            ezt = w00 * ez[[iy, ix]];
            ezt += w01 * ez[[iy, ix1]];
            ezt += w02 * ez[[iy, ix2]];
            ezt += w10 * ez[[iy1, ix]];
            ezt += w11 * ez[[iy1, ix1]];
            ezt += w12 * ez[[iy1, ix2]];
            ezt += w20 * ez[[iy2, ix]];
            ezt += w21 * ez[[iy2, ix1]];
            ezt += w22 * ez[[iy2, ix2]];
            ezt *= self.beta;

            bxt = w00 * ex[[iy, ix]];
            bxt += w01 * ex[[iy, ix1]];
            bxt += w02 * ex[[iy, ix2]];
            bxt += w10 * ex[[iy1, ix]];
            bxt += w11 * ex[[iy1, ix1]];
            bxt += w12 * ex[[iy1, ix2]];
            bxt += w20 * ex[[iy2, ix]];
            bxt += w21 * ex[[iy2, ix1]];
            bxt += w22 * ex[[iy2, ix2]];
            bxt *= self.alpha;

            byt = w00 * by[[iy, ix]];
            byt += w01 * by[[iy, ix + 1]];
            byt += w02 * by[[iy, ix + 2]];
            byt += w10 * by[[iy + 1, ix]];
            byt += w11 * by[[iy + 1, ix + 1]];
            byt += w12 * by[[iy + 1, ix + 2]];
            byt += w20 * by[[iy + 2, ix]];
            byt += w21 * by[[iy + 2, ix + 1]];
            byt += w22 * by[[iy + 2, ix + 2]];
            byt *= self.alpha;

            bzt = w00 * bz[[iy, ix]];
            bzt += w01 * bz[[iy, ix + 1]];
            bzt += w02 * bz[[iy, ix + 2]];
            bzt += w10 * bz[[iy + 1, ix]];
            bzt += w11 * bz[[iy + 1, ix + 1]];
            bzt += w12 * bz[[iy + 1, ix + 2]];
            bzt += w20 * bz[[iy + 2, ix]];
            bzt += w21 * bz[[iy + 2, ix + 1]];
            bzt += w22 * bz[[iy + 2, ix + 2]];
            bzt *= self.alpha;

            //  Now, the Boris push:
            ux = *px + ext;
            uy = *py + eyt;
            uz = *pz + ezt;
            pt = ux * ux + uy * uy + uz * uz;
            gt = (1. + pt * *CSQINV ).sqrt().powf(-1.0);

            bxt *= gt;
            byt *= gt;
            bzt *= gt;

            boris = 2.0 * (1.0 + bxt * bxt + byt * byt + bzt * bzt).powf(-1.0);

            uxt = ux + uy*bzt - uz*byt;
            uyt = uy + uz*bxt - ux*bzt;
            uzt = uz + ux*byt - uy*bxt;

            *px = ux + boris * (uyt * bzt - uzt * byt) + ext;
            *py = uy + boris * (uzt * bxt - uxt * bzt) + eyt;
            *pz = uz + boris * (uxt * byt - uyt * bxt) + ezt;

            *psa = (1.0 + (*px * *px + *py * *py + *pz * *pz) * *CSQINV).sqrt()
        }
    }
}

fn main() {
    let mut sim = new_sim();
    sim.add_species(1.0, 1.0, 1E-3);
    sim.add_species(-1.0, 1.0, 1E-3);
    //println!("{} {} {}", ions.px[0], ions.py[0], ions.pz[0]);
    //println!("{} {} {}", ions.px[0], ions.py[0], ions.pz[0]);
    //println!("{}", *BETA_INJ);
}
