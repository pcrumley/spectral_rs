use crate::{
    flds::{field::Field, Flds},
    prtls::Prtl,
    Float, Sim,
};
use anyhow::{Context, Result};

pub(crate) fn save_fld_spatial(fld: &Field, outdir: &str) -> Result<()> {
    let size_x = fld.no_ghost_dim.size_x;
    let size_y = fld.no_ghost_dim.size_y;
    let spatial = &fld.spatial;
    let mut out_vec: Vec<Float> = Vec::with_capacity(size_y * size_x);

    for iy in 0..size_y {
        let ij_ghosts = (iy + 1) * (size_x + 2);
        out_vec.extend(spatial.iter().skip(ij_ghosts).take(size_x));
    }

    npy::to_file(format!("{}/flds/{}.npy", outdir, fld.name), out_vec)
        .context(format!("Could not save {} data to file", fld.name))?;

    Ok(())
}

pub(crate) fn save_output(t: u32, sim: &Sim, flds: &Flds, prtls: &Vec<Prtl>) -> Result<()> {
    let cfg = &sim.config;
    if t % cfg.output.output_interval == 0 {
        let output_prefix = format!("output/dat_{:05}", t / cfg.output.output_interval);
        std::fs::create_dir_all(&output_prefix).context("Unable to create output directory")?;

        std::fs::create_dir_all(&format!("{}/flds", &output_prefix))
            .context("Unable to create output directory")?;
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

        let u: Vec<Float> = prtls[0]
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
        for fld in &[
            &flds.j_x, &flds.j_y, &flds.j_z, &flds.e_x, &flds.e_y, &flds.e_z, &flds.b_x, &flds.b_y,
            &flds.b_z,
        ] {
            save_fld_spatial(fld, &output_prefix)?;
        }
    }

    Ok(())

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
}
