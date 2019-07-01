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
            *psa = 1.0 + (*px * *px + *py * *py + *pz * *pz)/(*C * *C);
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
}

fn main() {
    let mut sim = new_sim();
    sim.add_species(1.0, 1.0, 1E-3);
    sim.add_species(-1.0, 1.0, 1E-3);

    //println!("{} {} {}", ions.px[0], ions.py[0], ions.pz[0]);
    //println!("{}", *BETA_INJ);
}
