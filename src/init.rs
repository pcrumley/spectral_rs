use serde::{Deserialize};
use std::fs;
#[derive(Deserialize)]

pub struct Config {
    pub params: Params
}


#[derive(Deserialize)]
pub struct Params {
    pub size_x: u32,
    pub size_y: u32,
    pub delta: u32,
    pub dt: f32,
    pub c: f32,
    pub dens: f32,
    pub gamma_inj: f32,
    pub n_pass: u32,
}


pub fn read() -> Config {
    let contents = fs::read_to_string("config.toml")
       .expect("Something went wrong reading the config.toml file");

    let config: Config = toml::from_str(&contents).unwrap();
    config
}
