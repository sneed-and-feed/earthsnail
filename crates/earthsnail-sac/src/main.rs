//! Primary `earthsnail` dual-mode CLI binary (`inspect`, `convert`, `filter`, `transfer`, `pick`,
//! `taup`, `ppsd`, `rf`, `fk`, `plot`, `fdsn`, `seedlink`, `verify`, and `sac`).

#![forbid(unsafe_code)]

use std::io::Write;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match earthsnail_sac::run_cli_args(&args) {
        Ok(out) => {
            if !out.is_empty() {
                let mut stdout = std::io::stdout().lock();
                let _ = stdout.write_all(out.as_bytes());
                let _ = stdout.flush();
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("earthsnail error: {err}");
            ExitCode::FAILURE
        }
    }
}
