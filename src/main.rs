use std::process;

fn main() {
    let config = spectral_rs::Config::new().unwrap_or_else(|err| {
            println!("Problem parsing arguments: {}", err);
            process::exit(1);
        });
    if let Err(e) = spectral_rs::run(config) {
        println!("Application error: {}", e);
        process::exit(1);
    }
}
