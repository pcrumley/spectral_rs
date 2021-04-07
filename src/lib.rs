mod flds;
mod prtls;

use flds::Flds;
use prtls::Prtl;
use serde::Deserialize;
use std::fs;

use anyhow::{Context, Result};
use itertools::izip;
// We use a type alias for f64, Float, to easily support
// double and single precision.
#[cfg(feature = "dprec")]
type Float = f64;

#[cfg(not(feature = "dprec"))]
type Float = f32;

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

pub struct Sim {
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

    #[inline(always)]
    fn spatial_get_index(&self, pos: crate::flds::Pos) -> usize {
        // Convenience method to get a position in the array.
        // Slightly complicated because
        // Using a 1d vec to represent 2D array for speed.
        // Here is the layout if it were a 2d array,
        // with the 1D vec position in []
        // ----------------------------------
        // |   [0]    |   [1]    |   [2]    |
        // |  row: 0  |  row: 0  |  row: 0  |
        // |  col: 0  |  col: 1  |  col: 2  |
        // |          |          |          |
        // ----------------------------------
        // |   [3]    |   [4]    |   [5]    |
        // |  row: 1  |  row: 1  |  row: 1  |
        // |  col: 0  |  col: 1  |  col: 2  |
        // |          |          |          |
        // ----------------------------------
        // |   [6]    |   [7]    |   [8]    |
        // |  row: 2  |  row: 2  |  row: 2  |
        // |  col: 0  |  col: 1  |  col: 2  |
        // |          |          |          |
        // ----------------------------------

        let row_len = self.size_x + 2;
        if !cfg!(feature = "unchecked") {
            assert!(pos.col < row_len);
            assert!(pos.row < self.size_y + 2);
        }
        pos.row * row_len + pos.col
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

        let j_x = &mut flds.j_x.spatial;
        let j_y = &mut flds.j_y.spatial;
        let j_z = &mut flds.j_z.spatial;

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
        Flds::deposit_ghosts(self, j_x);
        Flds::deposit_ghosts(self, j_y);
        Flds::deposit_ghosts(self, j_z);
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
        Flds::deposit_ghosts(self, dsty);
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
