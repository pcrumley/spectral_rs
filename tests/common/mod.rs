use spectral_rs::{Config, Output, Params, Setup, Sim};
pub fn setup_sim() -> Sim {
    // This is a function that sets up a dummy small
    // simulation so that it can be used in testing;
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
