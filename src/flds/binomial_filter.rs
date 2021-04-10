use crate::{Float, Sim};

#[inline(always)]
fn binomial_filter_2_d(sim: &Sim, in_vec: &mut Vec<Float>, wrkspace: &mut Vec<Float>) {
    // wrkspace should be same size as fld
    if !cfg!(feature = "unchecked") {
        assert!(in_vec.len() == wrkspace.len());
        assert!(in_vec.len() == (sim.size_x + 2) * (sim.size_y + 2));
    }

    let weights: [Float; 3] = [0.25, 0.5, 0.25];
    // account for ghost zones
    // FIRST FILTER IN X-DIRECTION

    for _ in 0..sim.n_pass {
        for i in ((sim.size_x + 2)..(sim.size_y + 1) * (sim.size_x + 2)).step_by(sim.size_x + 2) {
            for j in 1..sim.size_x + 1 {
                wrkspace[i] = weights
                    .iter()
                    .zip(&in_vec[i + j - 1..i + j + 1])
                    .map(|(&w, &f)| w * f)
                    .sum::<Float>();
            }
        }

        Flds::update_ghosts(&sim, wrkspace);

        // NOW FILTER IN Y-DIRECTION AND PUT VALS IN in_vec
        for i in ((sim.size_x + 2)..(sim.size_y + 1) * (sim.size_x + 2)).step_by(sim.size_x + 2) {
            for j in 1..sim.size_x + 1 {
                in_vec[i] = weights
                    .iter()
                    .zip(
                        wrkspace[i + j - (sim.size_x + 2)..i + j + (sim.size_x + 2)]
                            .iter()
                            .step_by(sim.size_x + 2),
                    )
                    .map(|(&w, &f)| w * f)
                    .sum::<Float>();
            }
        }

        Flds::update_ghosts(&sim, in_vec);
    }
}


