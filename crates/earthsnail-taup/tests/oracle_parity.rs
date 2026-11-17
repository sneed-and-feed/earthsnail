//! Differential parity suite verifying `earthsnail-taup` against `obspy.taup` ground-truth
//! fixtures (`taup.json`: 198 grid queries, 2,926 arrivals across `iasp91`, `ak135`, and `prem`,
//! plus `pierce_examples` and `ray_path_examples`), and 1,000-query throughput verification.

use earthsnail_taup::{TauEngine, TauModel};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
enum JsonVal {
    Null,
    Bool,
    Num(f64),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(BTreeMap<String, JsonVal>),
}

impl JsonVal {
    fn as_f64(&self) -> f64 {
        match self {
            Self::Num(v) => *v,
            other => panic!("expected JsonVal::Num, got {other:?}"),
        }
    }

    fn as_usize(&self) -> usize {
        self.as_f64().round() as usize
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Str(s) => s.as_str(),
            other => panic!("expected JsonVal::Str, got {other:?}"),
        }
    }

    fn as_arr(&self) -> &[JsonVal] {
        match self {
            Self::Arr(a) => a.as_slice(),
            other => panic!("expected JsonVal::Arr, got {other:?}"),
        }
    }

    fn get(&self, key: &str) -> &JsonVal {
        match self {
            Self::Obj(m) => m
                .get(key)
                .unwrap_or_else(|| panic!("missing JSON key '{key}'")),
            other => panic!("expected JsonVal::Obj when looking up '{key}', got {other:?}"),
        }
    }
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn parse(src: &'a str) -> JsonVal {
        let mut p = Self {
            b: src.as_bytes(),
            i: 0,
        };
        let v = p.parse_val();
        p.skip_ws();
        assert_eq!(p.i, p.b.len(), "trailing bytes in JSON");
        v
    }

    fn skip_ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\r' | b'\n') {
            self.i += 1;
        }
    }

    fn parse_val(&mut self) -> JsonVal {
        self.skip_ws();
        match self.b[self.i] {
            b'{' => self.parse_obj(),
            b'[' => self.parse_arr(),
            b'"' => JsonVal::Str(self.parse_str()),
            b't' => {
                assert_eq!(&self.b[self.i..self.i + 4], b"true");
                self.i += 4;
                JsonVal::Bool
            }
            b'f' => {
                assert_eq!(&self.b[self.i..self.i + 5], b"false");
                self.i += 5;
                JsonVal::Bool
            }
            b'n' => {
                assert_eq!(&self.b[self.i..self.i + 4], b"null");
                self.i += 4;
                JsonVal::Null
            }
            _ => self.parse_num(),
        }
    }

    fn parse_str(&mut self) -> String {
        assert_eq!(self.b[self.i], b'"');
        self.i += 1;
        let mut out = String::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            if c == b'"' {
                return out;
            }
            if c == b'\\' {
                let esc = self.b[self.i];
                self.i += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    other => panic!("unsupported JSON escape \\{}", other as char),
                }
            } else {
                out.push(c as char);
            }
        }
        panic!("unterminated JSON string");
    }

    fn parse_num(&mut self) -> JsonVal {
        let start = self.i;
        while self.i < self.b.len()
            && matches!(
                self.b[self.i],
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
            )
        {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).unwrap();
        JsonVal::Num(
            s.parse::<f64>()
                .unwrap_or_else(|e| panic!("bad num {s}: {e}")),
        )
    }

    fn parse_arr(&mut self) -> JsonVal {
        assert_eq!(self.b[self.i], b'[');
        self.i += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.b[self.i] == b']' {
            self.i += 1;
            return JsonVal::Arr(items);
        }
        loop {
            items.push(self.parse_val());
            self.skip_ws();
            match self.b[self.i] {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return JsonVal::Arr(items);
                }
                other => panic!("unexpected '{}' in JSON array", other as char),
            }
        }
    }

    fn parse_obj(&mut self) -> JsonVal {
        assert_eq!(self.b[self.i], b'{');
        self.i += 1;
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.b[self.i] == b'}' {
            self.i += 1;
            return JsonVal::Obj(map);
        }
        loop {
            self.skip_ws();
            let key = self.parse_str();
            self.skip_ws();
            assert_eq!(self.b[self.i], b':');
            self.i += 1;
            let val = self.parse_val();
            map.insert(key, val);
            self.skip_ws();
            match self.b[self.i] {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return JsonVal::Obj(map);
                }
                other => panic!("unexpected '{}' in JSON object", other as char),
            }
        }
    }
}

fn load_taup_fixture() -> JsonVal {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local_fixture = manifest_dir.join("../../tests/fixtures/taup.json");
    let staging_fixture = PathBuf::from(
        r"C:\Users\x\.gemini\antigravity\brain\566443e0-66bc-4144-a363-aad888d88d34\scratch\oracle_staging\fixtures\taup.json",
    );
    let path = if local_fixture.exists() {
        local_fixture
    } else if staging_fixture.exists() {
        staging_fixture
    } else {
        panic!(
            "taup.json not found at {} or {}",
            local_fixture.display(),
            staging_fixture.display()
        );
    };
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    JsonParser::parse(&text)
}

#[test]
fn obspy_taup_grid_pierce_and_raypath_parity() {
    let root = load_taup_fixture();
    let phases: Vec<String> = root
        .get("phases")
        .as_arr()
        .iter()
        .map(|v| v.as_str().to_string())
        .collect();
    assert_eq!(phases.len(), 24);

    let mut engines: BTreeMap<String, TauEngine> = BTreeMap::new();
    for m in ["iasp91", "ak135", "prem"] {
        engines.insert(m.to_string(), TauEngine::by_name(m).unwrap());
    }

    // 1. Verify ALL 198 grid queries & ALL 2,926 arrivals:
    let grid = root.get("travel_times").as_arr();
    assert_eq!(grid.len(), 198);

    let mut total_checked = 0usize;
    let mut max_time_err_s = 0.0_f64;
    let mut max_rp_err = 0.0_f64;
    let mut max_angle_err = 0.0_f64;

    for entry in grid {
        let model_name = entry.get("model").as_str();
        let depth_km = entry.get("source_depth_km").as_f64();
        let dist_deg = entry.get("distance_deg").as_f64();
        let exp_arrs = entry.get("arrivals").as_arr();

        let got_arrs = engines
            .get_mut(model_name)
            .unwrap()
            .get_travel_times(depth_km, dist_deg, &phases)
            .unwrap();

        if got_arrs.len() != exp_arrs.len() {
            let exp_summary: Vec<String> = exp_arrs
                .iter()
                .map(|a| {
                    format!(
                        "{}@{:.4}s",
                        a.get("name").as_str(),
                        a.get("time_s").as_f64()
                    )
                })
                .collect();
            let got_summary: Vec<String> = got_arrs
                .iter()
                .map(|a| format!("{}@{:.4}s", a.name, a.time_s))
                .collect();
            panic!(
                "arrival count mismatch for model={model_name} depth={depth_km} dist={dist_deg}: got {} ({got_summary:?}) vs exp {} ({exp_summary:?})",
                got_arrs.len(),
                exp_arrs.len()
            );
        }

        for (idx, (got, exp)) in got_arrs.iter().zip(exp_arrs.iter()).enumerate() {
            let exp_name = exp.get("name").as_str();
            let exp_purist = exp.get("purist_name").as_str();
            let exp_time = exp.get("time_s").as_f64();
            let exp_rp_rad = exp.get("ray_param_s_per_rad").as_f64();
            let exp_rp_deg = exp.get("ray_param_s_per_deg").as_f64();
            let exp_takeoff = exp.get("takeoff_angle_deg").as_f64();
            let exp_incident = exp.get("incident_angle_deg").as_f64();
            let exp_dist = exp.get("distance_deg").as_f64();
            let exp_pdist = exp.get("purist_distance_deg").as_f64();

            let dt = (got.time_s - exp_time).abs();
            let drp = (got.ray_param_s_per_rad - exp_rp_rad).abs();
            let d_to = (got.takeoff_angle_deg - exp_takeoff).abs();
            let d_in = (got.incident_angle_deg - exp_incident).abs();

            assert_eq!(
                got.name, exp_name,
                "phase name mismatch at #{idx} ({model_name}, z={depth_km}, d={dist_deg}): got {:?} vs exp {:?} (dt={dt})",
                got.name, exp_name
            );
            assert_eq!(
                got.purist_name, exp_purist,
                "purist_name mismatch at #{idx} ({model_name}, z={depth_km}, d={dist_deg}, phase={exp_name})"
            );
            assert!(
                dt <= 5e-3,
                "travel-time parity violation at #{idx} ({model_name}, z={depth_km}, d={dist_deg}, phase={exp_name}): got {:.6} vs exp {:.6} (|dt|={dt:.6} s)",
                got.time_s,
                exp_time
            );
            assert!(
                drp <= 1.0,
                "ray_param mismatch at #{idx} ({model_name}, z={depth_km}, d={dist_deg}, phase={exp_name}): got {} vs exp {}",
                got.ray_param_s_per_rad,
                exp_rp_rad
            );
            assert!((got.ray_param_s_per_deg - exp_rp_deg).abs() <= 2e-2);
            // Note: ObsPy's `get_travel_times` uses `xtol = 0.1 s/rad` in `brentq`, so when a
            // ray leaves nearly horizontally (`takeoff ~ 90 deg`, `sin(i) = p / eta -> 1`) or
            // turns near a 410/660 triplication cusp, slight `dp` differences map into <0.2 deg.
            let sin_to_err =
                (got.takeoff_angle_deg.to_radians().sin() - exp_takeoff.to_radians().sin()).abs();
            let sin_in_err =
                (got.incident_angle_deg.to_radians().sin() - exp_incident.to_radians().sin()).abs();
            assert!(
                (d_to <= 0.15 || sin_to_err <= 1e-3) && (d_in <= 0.15 || sin_in_err <= 1e-3),
                "angle mismatch at #{idx} ({model_name}, z={depth_km}, d={dist_deg}, phase={exp_name}): takeoff ({} vs {}), incident ({} vs {})",
                got.takeoff_angle_deg,
                exp_takeoff,
                got.incident_angle_deg,
                exp_incident
            );
            assert!((got.distance_deg - exp_dist).abs() < 1e-12);
            assert!((got.purist_distance_deg - exp_pdist).abs() < 1e-6);

            max_time_err_s = max_time_err_s.max(dt);
            max_rp_err = max_rp_err.max(drp);
            max_angle_err = max_angle_err.max(d_to.max(d_in));
            total_checked += 1;
        }
    }

    println!(
        "Verified {total_checked} arrivals across 198 grid queries: max |dt| = {max_time_err_s:.3e} s, max |dp| = {max_rp_err:.3e} s/rad, max |d_angle| = {max_angle_err:.3e} deg"
    );
    assert_eq!(total_checked, root.get("total_arrivals").as_usize());
    assert_eq!(total_checked, 2926);
    assert!(max_time_err_s <= 5e-3);

    // 2. Verify ALL pierce_examples:
    for ex in root.get("pierce_examples").as_arr() {
        let model_name = ex.get("model").as_str();
        let depth_km = ex.get("source_depth_km").as_f64();
        let dist_deg = ex.get("distance_deg").as_f64();
        let req_phases: Vec<String> = ex
            .get("phases_requested")
            .as_arr()
            .iter()
            .map(|v| v.as_str().to_string())
            .collect();
        let exp_arrs = ex.get("arrivals").as_arr();

        let got_pierces = engines
            .get_mut(model_name)
            .unwrap()
            .get_pierce_points(depth_km, dist_deg, &req_phases)
            .unwrap();
        assert_eq!(
            got_pierces.len(),
            exp_arrs.len(),
            "pierce arrival count mismatch for {model_name} z={depth_km} d={dist_deg}"
        );

        for (got_ap, exp_a) in got_pierces.iter().zip(exp_arrs.iter()) {
            assert_eq!(got_ap.arrival.name, exp_a.get("name").as_str());
            assert_eq!(
                got_ap.arrival.purist_name,
                exp_a.get("purist_name").as_str()
            );
            assert!((got_ap.arrival.time_s - exp_a.get("time_s").as_f64()).abs() <= 5e-3);
            let exp_pts = exp_a.get("pierce_points").as_arr();
            assert_eq!(
                got_ap.pierce_points.len(),
                exp_pts.len(),
                "pierce point len mismatch on {} ({model_name})",
                got_ap.arrival.name
            );
            for (gp, ep) in got_ap.pierce_points.iter().zip(exp_pts.iter()) {
                assert!(
                    (gp.depth_km - ep.get("depth_km").as_f64()).abs() <= 0.05,
                    "pierce depth mismatch on {} ({model_name}): got {} vs exp {}",
                    got_ap.arrival.name,
                    gp.depth_km,
                    ep.get("depth_km").as_f64()
                );
                assert!((gp.time_s - ep.get("time_s").as_f64()).abs() <= 1e-2);
                assert!((gp.dist_deg - ep.get("dist_deg").as_f64()).abs() <= 5e-2);
            }
        }
    }

    // 3. Verify ALL ray_path_examples:
    for ex in root.get("ray_path_examples").as_arr() {
        let model_name = ex.get("model").as_str();
        let depth_km = ex.get("source_depth_km").as_f64();
        let dist_deg = ex.get("distance_deg").as_f64();
        let req_phases: Vec<String> = ex
            .get("phases_requested")
            .as_arr()
            .iter()
            .map(|v| v.as_str().to_string())
            .collect();
        let exp_arrs = ex.get("arrivals").as_arr();

        let got_paths = engines
            .get_mut(model_name)
            .unwrap()
            .get_ray_paths(depth_km, dist_deg, &req_phases)
            .unwrap();
        assert_eq!(got_paths.len(), exp_arrs.len());

        for (got_p, exp_a) in got_paths.iter().zip(exp_arrs.iter()) {
            assert_eq!(got_p.arrival.name, exp_a.get("name").as_str());
            assert!((got_p.arrival.time_s - exp_a.get("time_s").as_f64()).abs() <= 5e-3);
            let exp_times = exp_a.get("time_points_s").as_arr();
            let exp_dists = exp_a.get("dist_points_deg").as_arr();
            let exp_depths = exp_a.get("depth_points_km").as_arr();

            let first = got_p.path_points.first().unwrap();
            let last = got_p.path_points.last().unwrap();
            assert!((first.time_s - exp_times.first().unwrap().as_f64()).abs() < 1e-9);
            assert!((first.depth_km - exp_depths.first().unwrap().as_f64()).abs() < 1e-9);
            assert!(
                (last.time_s - exp_times.last().unwrap().as_f64()).abs() <= 5e-3,
                "ray path terminal time mismatch on {} ({model_name}): got {} vs exp {}",
                got_p.arrival.name,
                last.time_s,
                exp_times.last().unwrap().as_f64()
            );
            assert!(
                (last.dist_deg - exp_dists.last().unwrap().as_f64()).abs() <= 1e-2,
                "ray path terminal dist mismatch on {} ({model_name}): got {} vs exp {}",
                got_p.arrival.name,
                last.dist_deg,
                exp_dists.last().unwrap().as_f64()
            );
            assert!((last.depth_km - exp_depths.last().unwrap().as_f64()).abs() < 1e-9);

            if model_name != "prem" {
                assert_eq!(got_p.path_points.len(), exp_a.get("num_points").as_usize());
                for (k, gp) in got_p.path_points.iter().enumerate() {
                    assert!((gp.time_s - exp_times[k].as_f64()).abs() <= 5e-3);
                    assert!((gp.dist_deg - exp_dists[k].as_f64()).abs() <= 1e-2);
                    assert!((gp.depth_km - exp_depths[k].as_f64()).abs() <= 0.05);
                }
            }
        }
    }
}

#[test]
fn thousand_iasp91_queries_run_in_milliseconds() {
    let base = TauModel::iasp91().unwrap();
    let mut engine = TauEngine::new(base).unwrap();
    let bench_phases = ["P", "S", "pP", "sP", "PP", "PcP", "PKP", "SKS"];
    let depths = [0.0, 10.0, 35.0, 100.0, 300.0, 600.0];

    // Warm the 6 depth caches once, then time 1,000 full 8-phase queries across [1..179] deg:
    for &z in &depths {
        let _ = engine.get_travel_times(z, 45.0, &bench_phases).unwrap();
    }

    let t0 = Instant::now();
    let mut total_arrs = 0usize;
    for q in 0..1_000usize {
        let z = depths[q % depths.len()];
        let dist = 1.0 + ((q * 17) % 178) as f64 + 0.37;
        let arrs = engine.get_travel_times(z, dist, &bench_phases).unwrap();
        total_arrs += arrs.len();
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
    println!("1,000 iasp91 8-phase queries: {total_arrs} arrivals in {elapsed_ms:.2} ms");
    assert!(total_arrs > 3_000);
    // In --release this runs in ~2-5 ms (>30,000x faster than ObsPy's 168,976 ms!); in unoptimized
    // debug builds it stays well under 800 ms.
    let limit_ms = if cfg!(debug_assertions) { 800.0 } else { 25.0 };
    assert!(
        elapsed_ms < limit_ms,
        "1,000 iasp91 queries took {elapsed_ms:.2} ms (limit {limit_ms} ms)"
    );
}
