//! Pure, bounded reducer kernel. Coverage and duplicate delivery belong to the host.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const INPUT_BYTES: usize = 262_144;
const OUTPUT_BYTES: usize = 65_536;
const MAX_VALUES: usize = 1_024;
const MAX_COUNT: u64 = 9_007_199_254_740_991;
static mut INPUT: [u8; INPUT_BYTES] = [0; INPUT_BYTES];
static mut OUTPUT: [u8; OUTPUT_BYTES] = [0; OUTPUT_BYTES];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct Position(u64, String);
impl Position {
    fn validate(&self) -> Result<(), String> {
        if self.0 > MAX_COUNT || self.1.is_empty() || self.1.len() > 128 || !self.1.is_ascii() {
            return Err("position requires a safe nonnegative step and 1..128 ASCII source ID".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Point { value: f64, position: Position }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Contribution { value: Option<f64>, position: Position }
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "reducer", deny_unknown_fields)]
enum State {
    #[serde(rename = "mean/v1")]
    Mean { version: u32, count: u64, sum: f64 },
    #[serde(rename = "envelope/v1")]
    Envelope { version: u32, count: u64, first: Option<Point>, min: Option<Point>, max: Option<Point>, last: Option<Point> },
}
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Identity { reducer: String },
    Add { state: State, values: Vec<Contribution> },
    Merge { left: State, right: State },
    Finalize { state: State },
}
fn identity(reducer: &str) -> Result<State, String> {
    match reducer {
        "mean/v1" => Ok(State::Mean { version: 1, count: 0, sum: 0.0 }),
        "envelope/v1" => Ok(State::Envelope { version: 1, count: 0, first: None, min: None, max: None, last: None }),
        _ => Err("unknown reducer; select mean/v1 or envelope/v1".into()),
    }
}
fn validate(state: &State) -> Result<(), String> {
    let (version, count) = match state {
        State::Mean { version, count, sum } => {
            if !sum.is_finite() || (*count == 0 && *sum != 0.0) { return Err("invalid mean sum".into()); }
            (*version, *count)
        }
        State::Envelope { version, count, first, min, max, last } => {
            let points = [first, min, max, last];
            if points.iter().any(|p| p.is_some() != (*count > 0)) { return Err("envelope points disagree with count".into()); }
            for p in points.into_iter().flatten() {
                p.position.validate()?;
                if !p.value.is_finite() { return Err("nonfinite envelope value".into()); }
            }
            let mut distinct = 0;
            for (index, candidate) in points.iter().enumerate() {
                if let Some(point) = candidate {
                    if !points[..index].iter().any(|earlier| earlier.as_ref()
                        .is_some_and(|earlier| earlier.position == point.position)) {
                        distinct += 1;
                    }
                }
            }
            if distinct > *count {
                return Err("envelope retains more positions than its count".into());
            }
            if let (Some(f), Some(lo), Some(hi), Some(l)) = (first, min, max, last) {
                if points.into_iter().flatten().any(|p| p.position < f.position || p.position > l.position || p.value < lo.value || p.value > hi.value) {
                    return Err("inconsistent envelope bounds".into());
                }
                for a in points.into_iter().flatten() {
                    for b in points.into_iter().flatten() {
                        if a.position == b.position && a.value != b.value { return Err("conflicting values at one position".into()); }
                    }
                }
                if *count == 1 && (f.position != l.position || f.value != lo.value || f.value != hi.value) { return Err("inconsistent singleton envelope".into()); }
            }
            (*version, *count)
        }
    };
    if version != 1 || count > MAX_COUNT { return Err("unsupported state version or count".into()); }
    Ok(())
}
fn count_sum(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b).filter(|n| *n <= MAX_COUNT).ok_or_else(|| "count exceeds safe integer bound".into())
}
fn choose(a: Option<Point>, b: Option<Point>, mode: &str) -> Option<Point> {
    match (a, b) {
        (None, b) => b,
        (a, None) => a,
        (Some(a), Some(b)) => {
            let pick_b = match mode {
                "first" => b.position < a.position,
                "last" => b.position > a.position,
                "min" => b.value < a.value || (b.value == a.value && b.position < a.position),
                _ => b.value > a.value || (b.value == a.value && b.position < a.position),
            };
            Some(if pick_b { b } else { a })
        }
    }
}
fn merge(left: State, right: State) -> Result<State, String> {
    validate(&left)?; validate(&right)?;
    let state = match (left, right) {
        (State::Mean { count: a, sum: x, .. }, State::Mean { count: b, sum: y, .. }) => {
            State::Mean { version: 1, count: count_sum(a, b)?, sum: x + y }
        }
        (State::Envelope { count: a, first: af, min: amin, max: amax, last: al, .. }, State::Envelope { count: b, first: bf, min: bmin, max: bmax, last: bl, .. }) => {
            State::Envelope { version: 1, count: count_sum(a,b)?, first: choose(af,bf,"first"), min: choose(amin,bmin,"min"), max: choose(amax,bmax,"max"), last: choose(al,bl,"last") }
        }
        _ => return Err("cannot merge different reducers".into()),
    };
    validate(&state)?;
    Ok(state)
}
fn handle(request: Request) -> Result<Value, String> {
    match request {
        Request::Identity { reducer } => Ok(json!(identity(&reducer)?)),
        Request::Merge { left, right } => Ok(json!(merge(left, right)?)),
        Request::Add { mut state, values } => {
            validate(&state)?;
            if values.len() > MAX_VALUES { return Err("batch exceeds 1024 contributions".into()); }
            for contribution in values {
                contribution.position.validate()?;
                if let Some(value) = contribution.value {
                    if !value.is_finite() { return Err("nonfinite contribution".into()); }
                    let single = match &state {
                        State::Mean { .. } => State::Mean { version: 1, count: 1, sum: value },
                        State::Envelope { .. } => {
                            let point = Some(Point { value, position: contribution.position });
                            State::Envelope { version: 1, count: 1, first: point.clone(), min: point.clone(), max: point.clone(), last: point }
                        }
                    };
                    state = merge(state, single)?;
                }
            }
            Ok(json!(state))
        }
        Request::Finalize { state } => {
            validate(&state)?;
            Ok(match state {
                State::Mean { count, sum, .. } => json!({"count": count, "value": if count == 0 { None } else { Some(sum / count as f64) }}),
                State::Envelope { count, first, min, max, last, .. } => json!({"count":count,"first":first,"min":min,"max":max,"last":last}),
            })
        }
    }
}
#[no_mangle]
pub extern "C" fn abi_version() -> u32 { 1 }
#[no_mangle]
pub extern "C" fn input_ptr() -> *mut u8 { std::ptr::addr_of_mut!(INPUT).cast() }
#[no_mangle]
pub extern "C" fn output_ptr() -> *const u8 { std::ptr::addr_of!(OUTPUT).cast() }
#[no_mangle]
pub extern "C" fn input_capacity() -> u32 { INPUT_BYTES as u32 }
#[no_mangle]
pub extern "C" fn execute(length: u32) -> u32 {
    let result = if length as usize > INPUT_BYTES {
        Err("request exceeds 262144 bytes".to_string())
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(input_ptr(), length as usize) };
        serde_json::from_slice::<Request>(bytes).map_err(|e| format!("invalid request: {e}")).and_then(handle)
    };
    let response = match result { Ok(value) => json!({"ok":value}), Err(error) => json!({"error":error}) };
    let bytes = serde_json::to_vec(&response).expect("bounded serializable response");
    if bytes.len() > OUTPUT_BYTES { return 0; }
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), std::ptr::addr_of_mut!(OUTPUT).cast(), bytes.len()); }
    bytes.len() as u32
}
