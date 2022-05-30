# spectral_rs [![Build Status](https://api.travis-ci.com/pcrumley/spectral_rs.svg?branch=master)](https://app.travis-ci.com/github/pcrumley/spectral_rs)
A rust based relativistic pseudo-spectral PIC method

![Unmagnetized shock](https://user-images.githubusercontent.com/15001732/115772199-d46b3500-a37c-11eb-825d-96ab24d58760.png)

This is still very much a work in progress but can correctly model unmagnetized shocks

To run install rust https://www.rust-lang.org/tools/install

clone this repo. cd to the main directory (one with `config.toml`). Edit `config.toml` as you see fit, and then

`cargo run --release` 

By default the code is in single precision, but if you want it to run in double precision

`cargo run --release --features dprec`

I also memory "unsafe" code but provide runtime assertions that prove memory safety is not violated. If you don't want
have that very small runtime overhead you can run it as
`cargo run --release --features "dprec unchecked"`
