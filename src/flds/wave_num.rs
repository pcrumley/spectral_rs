use crate::{Float, Sim};
use itertools::izip;
const PI: Float = std::f64::consts::PI as Float;

pub struct WaveNumbers {
    pub k_x: Vec<Float>,
    pub k_y: Vec<Float>,
    pub k_norm: Vec<Float>,
}

impl WaveNumbers {
    pub fn new(sim: &Sim) -> WaveNumbers {
        let mut k_x = vec![0.0; sim.size_x * sim.size_y];
        let mut k_y = vec![0.0; sim.size_y * sim.size_x];
        let mut k_norm = vec![0.0; sim.size_y * sim.size_x];

        // Build the k basis of FFT
        for i in 0..sim.size_y {
            for j in 0..sim.size_x {
                let ind = i * sim.size_x + j;
                // FIRST DO K_X
                k_x[ind] = j as Float;
                if j >= sim.size_x / 2 + 1 {
                    k_x[ind] -= sim.size_x as Float;
                }
                k_x[ind] *= 2.0 * PI / (sim.size_x as Float);
                // NOW DO K_Y
                k_y[ind] = i as Float;
                if i >= sim.size_y / 2 + 1 {
                    k_y[ind] -= sim.size_y as Float;
                }
                k_y[ind] *= 2.0 * PI / (sim.size_y as Float);
            }
        }
        // Make the norm:
        for (norm, kx, ky) in izip!(&mut k_norm, &k_x, &k_y) {
            *norm = 1. / (kx * kx + ky * ky);
        }
 

        WaveNumbers {
            k_x,
            k_y,
            k_norm,
        }
    } 
}
