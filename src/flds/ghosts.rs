use crate::{flds::field::Pos, Float, Sim};

#[inline(always)]
pub fn update_ghosts(sim: &Sim, fld: &mut Vec<Float>) -> () {
    let size_x = sim.size_x;
    let size_y = sim.size_y;
    if !cfg!(feature = "unchecked") {
        assert!(fld.len() == (size_x + 2) * (size_y + 2));
    }
    // Copy bottom row into top ghost row
    let ghost_start = sim.spatial_get_index(Pos { row: 0, col: 1 });
    let ghost_range = ghost_start..ghost_start + size_x;
    let real_start = sim.spatial_get_index(Pos {
        row: sim.size_y,
        col: 1,
    });
    let real_range = real_start..real_start + size_x;
    for (ighost, ireal) in ghost_range.zip(real_range) {
        unsafe {
            *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
        }
    }
    // Copy top row into bottom ghost row
    let ghost_start = sim.spatial_get_index(Pos {
        row: size_y + 1,
        col: 1,
    });
    let ghost_range = ghost_start..ghost_start + size_x;
    let real_start = sim.spatial_get_index(Pos { row: 1, col: 1 });
    let real_range = real_start..real_start + size_x;
    for (ighost, ireal) in ghost_range.zip(real_range) {
        unsafe {
            *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
        }
    }
    // copy into left ghost columns from right real column
    let ghost_start = sim.spatial_get_index(Pos { row: 1, col: 0 });
    let ghost_end = sim.spatial_get_index(Pos {
        row: 1 + size_y,
        col: 0,
    });
    let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
    let real_start = sim.spatial_get_index(Pos {
        row: 1,
        col: size_x,
    });
    let real_end = sim.spatial_get_index(Pos {
        row: 1 + size_y,
        col: size_x,
    });
    let real_range = (real_start..real_end).step_by(2 + size_x);
    for (ighost, ireal) in ghost_range.zip(real_range) {
        unsafe {
            *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
        }
    }

    // copy into right ghost columns from left real column
    let ghost_start = sim.spatial_get_index(Pos {
        row: 1,
        col: size_x + 1,
    });
    let ghost_end = sim.spatial_get_index(Pos {
        row: 1 + size_y,
        col: size_x + 1,
    });
    let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
    let real_start = sim.spatial_get_index(Pos { row: 1, col: 1 });
    let real_end = sim.spatial_get_index(Pos {
        row: 1 + size_y,
        col: 1,
    });
    let real_range = (real_start..real_end).step_by(2 + size_x);
    for (ighost, ireal) in ghost_range.zip(real_range) {
        unsafe {
            *fld.get_unchecked_mut(ighost) = *fld.get_unchecked(ireal);
        }
    }
    // now do the corners
    // copy into top left from bottom right
    let btm_right = sim.spatial_get_index(Pos {
        row: size_y,
        col: size_x,
    });
    unsafe { *fld.get_unchecked_mut(0) = *fld.get_unchecked(btm_right) }

    // copy into top right from bottom left
    let btm_left = sim.spatial_get_index(Pos {
        row: size_y,
        col: 1,
    });
    unsafe { *fld.get_unchecked_mut(size_x + 1) = *fld.get_unchecked(btm_left) }

    // copy into bottom left from top right
    let ghost_btm_left = sim.spatial_get_index(Pos {
        row: size_y + 1,
        col: 0,
    });
    let top_right = sim.spatial_get_index(Pos {
        row: 1,
        col: size_x,
    });
    unsafe { *fld.get_unchecked_mut(ghost_btm_left) = *fld.get_unchecked(top_right) }

    // Copy into bottom right from top left
    let ghost_btm_right = sim.spatial_get_index(Pos {
        row: size_y + 1,
        col: size_x + 1,
    });
    let top_left = sim.spatial_get_index(Pos { row: 1, col: 1 });
    unsafe { *fld.get_unchecked_mut(ghost_btm_right) = *fld.get_unchecked(top_left) }
}

#[inline(always)]
pub fn deposit_ghosts(sim: &Sim, fld: &mut Vec<Float>) -> () {
    let size_x = sim.size_x;
    let size_y = sim.size_y;
    if !cfg!(feature = "unchecked") {
        assert!(fld.len() == (size_x + 2) * (size_y + 2));
    }
    // deposit top ghost row into last row
    let ghost_start = sim.spatial_get_index(Pos { row: 0, col: 1 });
    let ghost_range = ghost_start..ghost_start + size_x;
    let real_start = sim.spatial_get_index(Pos {
        row: sim.size_y,
        col: 1,
    });
    let real_range = real_start..real_start + size_x;
    for (ighost, ireal) in ghost_range.zip(real_range) {
        unsafe {
            *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
        }
    }
    // deposit bottom ghost row into top real row
    let ghost_start = sim.spatial_get_index(Pos {
        row: size_y + 1,
        col: 1,
    });
    let ghost_range = ghost_start..ghost_start + size_x;
    let real_start = sim.spatial_get_index(Pos { row: 1, col: 1 });
    let real_range = real_start..real_start + size_x;
    for (ighost, ireal) in ghost_range.zip(real_range) {
        unsafe {
            *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
        }
    }
    // deposit left ghost columns into right real column
    let ghost_start = sim.spatial_get_index(Pos { row: 1, col: 0 });
    let ghost_end = sim.spatial_get_index(Pos {
        row: 1 + size_y,
        col: 0,
    });
    let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
    let real_start = sim.spatial_get_index(Pos {
        row: 1,
        col: size_x,
    });
    let real_end = sim.spatial_get_index(Pos {
        row: 1 + size_y,
        col: size_x,
    });
    let real_range = (real_start..real_end).step_by(2 + size_x);
    for (ighost, ireal) in ghost_range.zip(real_range) {
        unsafe {
            *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
        }
    }

    // deposit right ghost columns into left real column
    let ghost_start = sim.spatial_get_index(Pos {
        row: 1,
        col: size_x + 1,
    });
    let ghost_end = sim.spatial_get_index(Pos {
        row: 1 + size_y,
        col: size_x + 1,
    });
    let ghost_range = (ghost_start..ghost_end).step_by(size_x + 2);
    let real_start = sim.spatial_get_index(Pos { row: 1, col: 1 });
    let real_end = sim.spatial_get_index(Pos {
        row: 1 + size_y,
        col: 1,
    });
    let real_range = (real_start..real_end).step_by(2 + size_x);
    for (ighost, ireal) in ghost_range.zip(real_range) {
        unsafe {
            *fld.get_unchecked_mut(ireal) += *fld.get_unchecked(ighost);
        }
    }
    // now do the corners
    // deposit top left into bottom right
    let btm_right = sim.spatial_get_index(Pos {
        row: size_y,
        col: size_x,
    });
    unsafe { *fld.get_unchecked_mut(btm_right) += *fld.get_unchecked(0) }

    // depost top right into bottom left
    let btm_left = sim.spatial_get_index(Pos {
        row: size_y,
        col: 1,
    });
    unsafe { *fld.get_unchecked_mut(btm_left) += *fld.get_unchecked(size_x + 1) }

    // depost bottom left into top right
    let ghost_btm_left = sim.spatial_get_index(Pos {
        row: size_y + 1,
        col: 0,
    });
    let top_right = sim.spatial_get_index(Pos {
        row: 1,
        col: size_x,
    });
    unsafe { *fld.get_unchecked_mut(top_right) += *fld.get_unchecked(ghost_btm_left) }

    // depost bottom right into top left
    let ghost_btm_right = sim.spatial_get_index(Pos {
        row: size_y + 1,
        col: size_x + 1,
    });
    let top_left = sim.spatial_get_index(Pos { row: 1, col: 1 });
    unsafe { *fld.get_unchecked_mut(top_left) += *fld.get_unchecked(ghost_btm_right) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_test_sim;
    #[test]
    fn ghost_deposit() {
        // made an example using google sheets
        let mut input: Vec<Float> = vec![
            1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.8, 1.8, 1.8, 1.8,
            1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1,
            1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.3, 1.3, 1.3, 1.3, 1.3,
            1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3,
        ];
        let expected_output: Vec<Float> = vec![
            1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.8, 1.8, 1.8, 1.8,
            1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.4, 4.21, 1.11, 1.11, 1.11, 1.11, 1.11,
            1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31,
            1.31, 1.31, 1.31, 3.81, 1.8, 1.4, 1.81, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.41, 1.8, 1.4, 1.81, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.41, 1.8, 1.4, 1.81,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.41, 1.8, 1.4, 1.81, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 1.41, 1.8, 1.1, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 1.11, 1.3, 1.1, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11,
            1.3, 1.1, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.3, 1.1, 1.31, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.3, 1.1, 1.31, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 1.11, 1.3, 1.1, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 1.11, 1.3, 1.1, 4.51, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41,
            1.41, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 4.31, 1.3, 1.1,
            1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.3, 1.3, 1.3, 1.3, 1.3,
            1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3,
        ];
        let sim = build_test_sim();
        assert_eq!(input.len(), expected_output.len());
        assert_eq!(input.len(), (sim.size_x + 2) * (sim.size_y + 2));
        deposit_ghosts(&sim, &mut input);
        for (v, expected_v) in input.into_iter().zip(expected_output) {
            assert_eq!(v, expected_v);
        }
    }

    #[test]
    fn ghosts_update() {
        // made an example using google sheets
        let mut input: Vec<Float> = vec![
            1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.8, 1.8, 1.8, 1.8,
            1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.4, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.8, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 1.3, 1.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.3, 1.1,
            1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.3, 1.3, 1.3, 1.3, 1.3,
            1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3,
        ];
        let expected_output: Vec<Float> = vec![
            4.31, 4.51, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.81,
            1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 4.31, 4.51, 3.81, 4.21,
            1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.31, 1.31, 1.31,
            1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 3.81, 4.21, 1.41, 1.81, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.41, 1.81, 1.41, 1.81, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 1.41, 1.81, 1.41, 1.81, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 1.41, 1.81, 1.41, 1.81, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.41, 1.81, 1.11, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.31,
            1.11, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.31, 1.11, 1.31,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.31, 1.11, 1.31, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.11, 1.31, 1.11, 1.31, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 1.11, 1.31, 1.11, 1.31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 1.11, 1.31, 4.31, 4.51, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41,
            1.41, 1.41, 1.41, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81,
            4.31, 4.51, 3.81, 4.21, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.11,
            1.11, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 3.81, 4.21,
        ];
        let sim = build_test_sim();
        assert_eq!(input.len(), expected_output.len());
        assert_eq!(input.len(), (sim.size_x + 2) * (sim.size_y + 2));
        deposit_ghosts(&sim, &mut input);
        update_ghosts(&sim, &mut input);
        for (v, expected_v) in input.into_iter().zip(expected_output) {
            assert_eq!(v, expected_v);
        }
    }
}
