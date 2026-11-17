//! Standalone `es-sac` binary launching directly into the interactive `SAC>` shell or `.m`
//! macro script engine (`es-sac [-m <script.m> [args...]] [-c "<cmds>"] [files...]`).

#![forbid(unsafe_code)]

use std::io::Write;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match earthsnail_sac::run_es_sac_args(&args) {
        Ok(out) => {
            if !out.is_empty() {
                let mut stdout = std::io::stdout().lock();
                let _ = stdout.write_all(out.as_bytes());
                let _ = stdout.flush();
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("es-sac error: {err}");
            ExitCode::FAILURE
        }
    }
}
