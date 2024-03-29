pub mod flds;
pub mod prtls;
pub mod save;

use flds::Flds;
use prtls::Prtl;
use save::save_output;
use serde::Deserialize;
use std::{fs, time::SystemTime};

use anyhow::{Context, Result};
use itertools::izip;
// We use a type alias for f64, Float, to easily support
// double and single precision.
#[cfg(feature = "dprec")]
pub type Float = f64;

#[cfg(not(feature = "dprec"))]
pub type Float = f32;

// these control how many prtls you grab in the
// rayon parallelization
pub const PRTL_CHUNK_SIZE: usize = 1000;
pub const FLD_CHUNK_SIZE: usize = 1000;

// some helpers to make the unit tests pass in both
// double precision and single precision modes
#[cfg(feature = "dprec")]
pub const E_TOL: Float = 1E-13;

#[cfg(not(feature = "dprec"))]
pub const E_TOL: Float = 1E-4;

#[derive(Deserialize, Clone)]
pub struct Config {
    pub params: Params,
    pub setup: Setup,
    pub output: Output,
}

#[derive(Deserialize, Clone)]
pub struct Setup {
    pub t_final: u32,
}

#[derive(Deserialize, Clone)]
pub struct Output {
    pub track_prtls: bool,
    pub write_output: bool,
    pub track_interval: u32,
    pub output_interval: u32,
    pub stride: usize,
    pub istep: usize,
}

#[derive(Deserialize, Clone)]
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
        let cfg: Config =
            toml::from_str(&contents).with_context(|| "Could not parse Config file")?;

        // the number of cells must be even for the fft algorithm to work.
        if cfg.params.size_x % 2 != 0 || cfg.params.size_y % 2 != 0 {
            let msg = match (cfg.params.size_x % 2 == 0, cfg.params.size_y % 2 == 0) {
                (true, false) => "y",
                (false, true) => "x",
                (false, false) => "x & y",
                (true, true) => unreachable!(),
            };
            return Err(anyhow::Error::msg(format!(
                "Number of cells in {} direction must be even",
                msg
            )));
        }
        if cfg.params.delta * 2 >= cfg.params.size_x {
            return Err(anyhow::Error::msg(
                "Delta must be less than 1/2 the size of size_x",
            ));
        }
        if cfg.params.gamma_inj <= 1.0 {
            return Err(anyhow::Error::msg("Lorentz factor must be greater than 1"));
        }

        // Number of cells in each direction must be divisible by the downsampling
        // for saving outputs to make the downsampling algorithm a simple skip.
        if cfg.params.size_x % cfg.output.istep != 0 || cfg.params.size_y % cfg.output.istep != 0 {
            let msg = match (
                cfg.params.size_x % cfg.output.istep == 0,
                cfg.params.size_y % cfg.output.istep == 0,
            ) {
                (true, false) => "y",
                (false, true) => "x",
                (false, false) => "x & y",
                (true, true) => unreachable!(),
            };
            return Err(anyhow::Error::msg(format!(
                "Number of cells in {} direction must be divisible by 2",
                msg
            )));
        }

        Ok(cfg)
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

    for prtl in prtls.iter_mut() {
        sim.move_and_deposit(prtl, &mut flds);
    }

    for fld in &mut [
        &mut flds.j_x,
        &mut flds.j_y,
        &mut flds.j_z,
        &mut flds.dsty,
        &mut flds.dens,
    ] {
        fld.deposit_ghosts();
    }
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
            save_output(t, &sim, &flds, &prtls)?;
        }
        // Zero out currents and density
        println!("{}", t);
        for fld in &mut [
            &mut flds.j_x.spatial,
            &mut flds.j_y.spatial,
            &mut flds.j_z.spatial,
            &mut flds.dsty.spatial,
            &mut flds.dens.spatial,
        ] {
            for v in fld.iter_mut() {
                *v = 0.0;
            }
        }
        println!("moving & dep prtl");
        let dep_time = SystemTime::now();
        // deposit current. This part is finished.
        for prtl in prtls.iter_mut() {
            sim.move_and_deposit(prtl, &mut flds);
        }
        for fld in &mut [
            &mut flds.j_x,
            &mut flds.j_y,
            &mut flds.j_z,
            &mut flds.dsty,
            &mut flds.dens,
        ] {
            fld.deposit_ghosts();
        }

        println!("{:?}", dep_time.elapsed().unwrap());
        // solve field. This part is NOT finished
        println!("solving fields");
        let solve_time = SystemTime::now();
        flds.update(&sim);
        println!("{:?}", solve_time.elapsed().unwrap());

        // push prtls finished
        println!("pushing prtl");
        let push_time = SystemTime::now();
        for prtl in prtls.iter_mut() {
            prtl.boris_push(&sim, &flds)
        }
        println!("{:?}", push_time.elapsed().unwrap());

        sim.t.set(t);
    }
    Ok(())
}

pub struct Sim {
    pub t: std::cell::Cell<u32>,
    pub t_final: u32,
    pub size_x: usize,
    pub size_y: usize,
    pub delta: usize,
    pub dt: Float,
    pub c: Float,
    pub dens: u32,
    pub gamma_inj: Float, // Speed of upstream flow
    pub prtl_num: usize,  // = *DENS * ( *SIZE_X - 2* *DELTA) * *SIZE_Y;
    pub n_pass: u8,       // = 4; //Number of filter passes
    pub config: Config,
}

pub fn build_test_sim() -> Sim {
    // convenience method that returns a sim with some simple values
    // we can pass to our test methods
    let cfg = Config {
        output: Output {
            track_prtls: false,
            write_output: false,
            track_interval: 100,
            output_interval: 100,
            stride: 4,
            istep: 1,
        },
        setup: Setup { t_final: 1000 },
        params: Params {
            size_x: 24,
            size_y: 12,
            dt: 0.1,
            delta: 5,
            c: 3.0,
            dens: 2,
            gamma_inj: 5.0,
            n_pass: 4,
        },
    };
    Sim::new(&cfg)
}

impl Sim {
    pub fn new(cfg: &Config) -> Sim {
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
            config: cfg.clone(),
        }
    }

    fn deposit_current(&self, prtl: &Prtl, flds: &mut Flds) {
        // local vars we will use

        // The [i,j] position in the array. Slightly complicated because
        // Using a 1d vec to represent 2D array for speed.
        // Here is the layout if it were a 2d array
        // +----------+--------+----------+
        // | ijm1 - 1 |  ijm1  | ijm1 + 1 |
        // +----------+--------+----------+
        // |  ij - 1  |   ij   |  ij + 1  |
        // +----------+--------+----------+
        // | ijp1 - 1 |  ijp1  | ijp1 + 1 |
        // +----------+--------+----------+

        let mut ij: usize;
        let mut ijm1: usize;
        let mut ijp1: usize;

        // Similarly for the weights
        // +-----+-----+-----+
        // | w00 | w01 | w02 |
        // +-----+-----+-----+
        // | w10 | w11 | w12 |
        // +-----+-----+-----+
        // | w20 | w21 | w22 |
        // +-----+-----+-----+

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

        let j_x = &mut flds.j_x.spatial;
        let j_y = &mut flds.j_y.spatial;
        let j_z = &mut flds.j_z.spatial;

        // Assert that all fields have the same length. Should be guaranteed
        // by their construction but not a bad idea to do it anyway.
        if !cfg!(feature = "uchecked") {
            assert_eq!(j_x.len(), j_y.len());
            assert_eq!(j_x.len(), j_z.len());
        }

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
            if !cfg!(feature = "unchecked") {
                // this assertion guarantees we do not try to grab a value
                // that is larger than array length.
                assert!(ijp1 + ix + 1 < j_x.len());
                // We do not need to do assert ijm1 + ix - 1 >= 0 because
                // we asserted ix > 0 and ijm1 >=0

                // We also previously asserted that all current arrays have
                // the same length.
            }
            // safe due to previous assertions in the code
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

        let dsty = &mut flds.dsty.spatial;
        let dens = &mut flds.dens.spatial;
        for (ix, iy, dx, dy) in izip!(&prtl.ix, &prtl.iy, &prtl.dx, &prtl.dy) {
            if !cfg!(feature = "unchecked") {
                assert!(*iy > 0);
                assert!(*ix > 0);
            }
            ijm1 = iy - 1;
            ijp1 = iy + 1;
            ij = iy * (2 + self.size_x);
            ijm1 *= 2 + self.size_x;
            ijp1 *= 2 + self.size_x;

            // CALC WEIGHTS
            // 2nd order
            // The weighting scheme prtl is in middle
            // # +------+------+------+
            // # | w0,0 | w0,1 | w0,2 |
            // # +------+------+------+
            // # | w1,0 | w1,1 | w1,2 |
            // # +------+------+------+
            // # | w2,0 | w2,1 | w2,2 |
            // # +------+------+------+
            w00 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w01 = 0.5 * (0.5 - dy) * (0.5 - dy) * (0.75 - dx * dx); // y0
            w02 = 0.5 * (0.5 - dy) * (0.5 - dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0
            w10 = (0.75 - dy * dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w11 = (0.75 - dy * dy) * (0.75 - dx * dx); // y0
            w12 = (0.75 - dy * dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0
            w20 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 - dx) * (0.5 - dx); // y0
            w21 = 0.5 * (0.5 + dy) * (0.5 + dy) * (0.75 - dx * dx); // y0
            w22 = 0.5 * (0.5 + dy) * (0.5 + dy) * 0.5 * (0.5 + dx) * (0.5 + dx); // y0

            // Deposit the density
            if !cfg!(feature = "unchecked") {
                // this assertion guarantees we do not try to grab a value
                // that is larger than array length.
                assert!(ijp1 + ix + 1 < dsty.len());
                // We do not need to do assert ijm1 + ix - 1 >= 0 because
                // we asserted ix > 0 and ijm1 >=0
            }
            // safe because of previous assertions
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

                *dens.get_unchecked_mut(ijm1 + ix - 1) += w00;
                *dens.get_unchecked_mut(ijm1 + ix) += w01;
                *dens.get_unchecked_mut(ijm1 + ix + 1) += w02;
                *dens.get_unchecked_mut(ij + ix - 1) += w10;
                *dens.get_unchecked_mut(ij + ix) += w11;
                *dens.get_unchecked_mut(ij + ix + 1) += w12;
                *dens.get_unchecked_mut(ijp1 + ix - 1) += w20;
                *dens.get_unchecked_mut(ijp1 + ix) += w21;
                *dens.get_unchecked_mut(ijp1 + ix + 1) += w22;
            }
        }
    }

    fn move_and_deposit(&self, prtl: &mut Prtl, flds: &mut Flds) {
        // FIRST we update positions of particles
        prtl.update_position(self);
        prtl.apply_bc(self);

        // Deposit currents
        self.deposit_current(prtl, flds);
        // UPDATE POS AGAIN!
        prtl.update_position(self);
        prtl.apply_bc(self);
        self.calc_density(&*prtl, flds);
    }
}

#[cfg(test)]
pub mod tests {
    use super::build_test_sim;
    #[test]
    fn sim_init() {
        let sim = build_test_sim();
        assert_eq!(sim.size_x, 24);
        assert_eq!(sim.size_y, 12);
    }
}
