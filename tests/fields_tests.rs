mod common;

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use spectral_rs::{flds::Flds, Float};

fn init_flds() -> Flds {
    let sim = common::setup_sim();
    assert_eq!(sim.size_x, 24);
    assert_eq!(sim.size_y, 12);
    Flds::new(&sim)
}

#[test]
fn test_field_init() {
    let expected_spatial_val: Vec<Float> = vec![0.; (24 + 2) * (12 + 2)];
    let expected_complex_val: Vec<Complex<Float>> = vec![Complex::zero(); 24 * 12];
    let flds = init_flds();
    for fld in &[
        flds.j_x, flds.j_y, flds.j_z, flds.dsty, flds.b_x, flds.b_y, flds.b_z,
    ] {
        assert_eq!(fld.spatial.len(), expected_spatial_val.len());
        assert_eq!(fld.spectral.len(), expected_complex_val.len());
        for (v, expected_v) in fld.spatial.iter().zip(expected_spatial_val.iter()) {
            assert_eq!(v, expected_v);
        }
        for (v, expected_v) in fld.spectral.iter().zip(expected_complex_val.iter()) {
            assert_eq!(v, expected_v);
        }
    }
}
