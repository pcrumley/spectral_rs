use crate::{Flds, Float, Sim};
use itertools::izip;
use rand::prelude::*;
use rand_distr::Standard;
use rand_distr::StandardNormal;

pub(crate) struct Prtl {
    pub ix: Vec<usize>,
    pub iy: Vec<usize>,
    pub dx: Vec<Float>,
    pub dy: Vec<Float>,
    pub px: Vec<Float>,
    pub py: Vec<Float>,
    pub pz: Vec<Float>,
    pub psa: Vec<Float>, // Lorentz Factors
    pub charge: Float,
    pub alpha: Float,
    pub beta: Float,
    pub vth: Float,
    pub tag: Vec<u64>,
    pub track: Vec<bool>,
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
    pub(crate) fn new(sim: &Sim, charge: Float, mass: Float, vth: Float) -> Prtl {
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
    #[inline(always)]
    pub(crate) fn update_position(&mut self, sim: &Sim) {
        let mut c1: Float;
        let dt = sim.dt;
        for (ix, iy, dx, dy, px, py, psa) in izip!(
            &mut self.ix,
            &mut self.iy,
            &mut self.dx,
            &mut self.dy,
            &self.px,
            &self.py,
            &self.psa
        ) {
            if !cfg!(feature = "unchecked") {
                // check that we don't underflow
                assert!(*ix > 0);
                assert!(*iy > 0);
            }
            c1 = 0.5 * dt * psa.powi(-1);
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
    }

    #[inline(always)]
    pub(crate) fn apply_bc(&mut self, sim: &Sim) {
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
        if cfg!(feature = "periodic") {
            for ix in self.ix.iter_mut() {
                if *ix < 1 {
                    *ix += sim.size_x;
                } else if *ix > sim.size_x {
                    *ix -= sim.size_x;
                }
            }
        } else {
            // hit the wall at right hand side
            /*
             * This is asserted at sim creation.
             *
             * if !cfg!(feature = "unchecked") {
             *   assert!(2 * sim.delta < sim.size_x);
             * }
             */
            let wall_loc = sim.size_x + 1 - sim.delta;
            if !cfg!(feature = "unchecked") {
                assert!(
                    *self
                        .ix
                        .iter()
                        .max()
                        .expect("Could not find max of prtl arr. should never happen")
                        <= wall_loc + 1
                );
                assert!(
                    *self
                        .ix
                        .iter()
                        .min()
                        .expect("Could not find min of prtl arr. should never happen")
                        > sim.delta - 1
                );
            }
            for (ix, dx, px) in izip!(self.ix.iter_mut(), self.dx.iter_mut(), self.px.iter_mut()) {
                if *ix < wall_loc {
                    // Do nothing.
                } else if *ix == wall_loc {
                    // if dx is positive that means it has crossed the wall,
                    // so we flip the sign of px and dx. Could put in an if
                    // statement but doing this way for branch prediction
                    // reasons.
                    let sgn = dx.signum();
                    *px *= -sgn;
                    *dx *= -sgn;
                } else {
                    *dx += 1.0;
                    *ix -= 1;
                    *px *= -1.0;
                }
            }
        }
    }
    fn initialize_positions(&mut self, sim: &Sim) {
        // A method to calculate the initial, non-random
        // position of the particles
        let mut c1 = 0;
        // let mut rng = thread_rng();
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
    }
    fn initialize_velocities(&mut self, sim: &Sim) {
        let csqinv = 1. / (sim.c * sim.c);
        let beta_inj = Float::sqrt(1. - sim.gamma_inj.powi(-2));
        // println!("{}", beta_inj);
        if false {
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
            let mut flipper: Float = -1.0;
            for (n, px, psa) in izip!(0..self.px.len(), &mut self.px, &mut self.psa) {
                *px = flipper.powi(n as i32) * sim.c * sim.gamma_inj * beta_inj;
                *psa = 1.0 + (*px * *px) * csqinv;
                *psa = psa.sqrt();
            }
            // println!("{:?}", self.px);
        }
    }
    pub(crate) fn boris_push(&mut self, sim: &Sim, flds: &Flds) {
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
            // +-----+-----+-----+
            // | w00 | w01 | w02 |
            // +-----+-----+-----+
            // | w10 | w11 | w12 |
            // +-----+-----+-----+
            // | w20 | w21 | w22 |
            // +-----+-----+-----+
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
