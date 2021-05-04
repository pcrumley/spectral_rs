use crate::{Flds, Float, Sim, PRTL_CHUNK_SIZE};
use itertools::izip;
use rand::prelude::*;
use rand_distr::Standard;
use rand_distr::StandardNormal;
use rayon::prelude::*;

pub(crate) struct Prtl {
    pub ix: Vec<usize>,
    pub iy: Vec<usize>,
    pub dx: Vec<Float>,
    pub dy: Vec<Float>,
    pub px: Vec<Float>,
    pub py: Vec<Float>,
    pub pz: Vec<Float>,
    pub psa: Vec<Float>, // Lorentz Factors
    pub weights: Vec<[Float; 9]>,
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
            weights: vec![[0.0; 9]; sim.prtl_num],
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
    pub(crate) fn calculate_weights(&mut self) {
        // CALC WEIGHTS
        // 2nd order
        // The weighting scheme prtl is in middle
        // +----+----+----+
        // | w0 | w1 | w2 |
        // +----+----+----+
        // | w3 | w4 | w5 |
        // +----+----+----+
        // | w6 | w7 | w8 |
        // +----+----+----+
        (&mut self.weights, &self.dx, &self.dy)
            .into_par_iter()
            .chunks(PRTL_CHUNK_SIZE)
            .for_each(|o| {
                o.into_iter().for_each(|(w, dx, dy)| {
                    *w = [
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
                })
            });
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
                        > sim.delta
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
                    r1 += (k as Float) / (sim.dens as Float);
                    self.dx[c1 + k] = r1 - 0.5;
                    self.dy[c1 + k] = r1 - 0.5;
                    self.tag[c1 + k] = (c1 + k) as u64;
                }
                c1 += sim.dens as usize;
                // helper_arr = -.5+0.25+np.arange(dens)*.5
            }
        }
        assert!(*self.ix.iter().max().unwrap() <= sim.size_x - sim.delta);
        assert!(*self.ix.iter().min().unwrap() >= 1 + sim.delta);
    }
    fn initialize_velocities(&mut self, sim: &Sim) {
        let csqinv = 1. / (sim.c * sim.c);
        let beta_inj = Float::sqrt(1. - sim.gamma_inj.powi(-2));
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
            for (px, psa) in izip!(&mut self.px, &mut self.psa) {
                *px = flipper * sim.c * sim.gamma_inj * beta_inj;
                *psa = 1.0 + (*px * *px) * csqinv;
                *psa = psa.sqrt();
                flipper *= -1.0;
            }
            // println!("{:?}", self.px);
        }
    }
    pub(crate) fn boris_push(&mut self, sim: &Sim, flds: &Flds) {
        // local vars we will use

        let csqinv = 1. / (sim.c * sim.c);
        let beta = self.beta;
        let alpha = self.alpha;
        // get direct pointers to avoid unnecessary lookups
        let e_x = &flds.e_x.spatial;
        let e_y = &flds.e_y.spatial;
        let e_z = &flds.e_z.spatial;

        let b_x = &flds.b_x.spatial;
        let b_y = &flds.b_y.spatial;
        let b_z = &flds.b_z.spatial;

        self.calculate_weights();
        let size_x = sim.size_x;
        (
            &self.ix,
            &self.iy,
            &self.weights,
            &mut self.px,
            &mut self.py,
            &mut self.pz,
            &mut self.psa,
        )
            .into_par_iter()
            .chunks(PRTL_CHUNK_SIZE)
            .for_each(|o| {
                o.into_iter()
                    .for_each(|(ix, iy, weights, px, py, pz, psa)| {
                        if !cfg!(feature = "unchecked") {
                            assert!(*iy > 0);
                            assert!(*ix > 0);
                        }
                        let ij = iy * (2 + size_x) + ix;
                        let ijm1 = ij - (2 + size_x);
                        let ijp1 = ij + (2 + size_x);

                        let places = &[
                            ijm1 - 1,
                            ijm1,
                            ijm1 + 1,
                            ij - 1,
                            ij,
                            ij + 1,
                            ijp1 - 1,
                            ijp1,
                            ijp1 + 1,
                        ];

                        // INTERPOLATE ALL THE FIELDS
                        if !cfg!(feature = "unchecked") {
                            assert!(places[8] < e_x.len());
                        }
                        // safe because of previous assertion

                        let mut ext: Float;
                        let mut eyt: Float;
                        let mut ezt: Float;
                        let mut bxt: Float;
                        let mut byt: Float;
                        let mut bzt: Float;
                        unsafe {
                            ext = weights
                                .iter()
                                .zip(places.into_iter())
                                .map(|(&w, &i)| w * e_x.get_unchecked(i))
                                .sum::<Float>();

                            eyt = weights
                                .iter()
                                .zip(places.iter())
                                .map(|(&w, &i)| w * e_y.get_unchecked(i))
                                .sum::<Float>();

                            ezt = weights
                                .iter()
                                .zip(places.iter())
                                .map(|(&w, &i)| w * e_z.get_unchecked(i))
                                .sum::<Float>();

                            bxt = weights
                                .iter()
                                .zip(places.iter())
                                .map(|(&w, &i)| w * b_x.get_unchecked(i))
                                .sum::<Float>();

                            byt = weights
                                .iter()
                                .zip(places.iter())
                                .map(|(&w, &i)| w * b_y.get_unchecked(i))
                                .sum::<Float>();

                            bzt = weights
                                .iter()
                                .zip(places.iter())
                                .map(|(&w, &i)| w * b_z.get_unchecked(i))
                                .sum::<Float>();
                        }
                        ext *= beta;
                        eyt *= beta;
                        ezt *= beta;
                        bxt *= alpha;
                        byt *= alpha;
                        bzt *= alpha;
                        //  Now, the Boris push:
                        let ux = *px + ext;
                        let uy = *py + eyt;
                        let uz = *pz + ezt;
                        let gt = (1. + (ux * ux + uy * uy + uz * uz) * csqinv)
                            .sqrt()
                            .powi(-1);

                        bxt *= gt;
                        byt *= gt;
                        bzt *= gt;

                        let boris = 2.0 * (1.0 + bxt * bxt + byt * byt + bzt * bzt).powi(-1);

                        let uxt = ux + uy * bzt - uz * byt;
                        let uyt = uy + uz * bxt - ux * bzt;
                        let uzt = uz + ux * byt - uy * bxt;

                        *px = ux + boris * (uyt * bzt - uzt * byt) + ext;
                        *py = uy + boris * (uzt * bxt - uxt * bzt) + eyt;
                        *pz = uz + boris * (uxt * byt - uyt * bxt) + ezt;

                        *psa = (1.0 + (*px * *px + *py * *py + *pz * *pz) * csqinv).sqrt()
                    })
            });
    }
}
