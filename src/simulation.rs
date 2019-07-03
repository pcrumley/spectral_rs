use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_distr::Standard;

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
    //prtls: Vec<Prtl>,
    t: u32,
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
        };
    sim
}
