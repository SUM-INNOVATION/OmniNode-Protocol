//! Issue #97 — real-transport precision boundary tests.
//!
//! These exercise the ACTUAL production HTTP path end-to-end:
//!
//! ```text
//! raw HTTP JSON body  →  UreqTransport::call  →  serde_json::Value
//!                     →  SumChainClient::get_chain_params
//!                     →  ChainParamsInfo.min_fee (u128)
//! ```
//!
//! A local hermetic `TcpListener` on `127.0.0.1:0` serves a LITERAL
//! JSON-RPC response body, so a `min_fee` above `u64::MAX` travels as
//! real bytes over a real socket through the real `ureq`-backed
//! transport — not a `FakeJsonRpcTransport` and not a direct
//! `from_str::<ChainParamsInfo>` (both of which bypass the intermediate
//! `serde_json::Value` where the precision loss would otherwise occur).
//!
//! The whole point of Issue #97's deeper fix: widening the DTO field to
//! `u128` is necessary but NOT sufficient — without `serde_json`'s
//! `arbitrary_precision` feature (enabled crate-locally) the integer is
//! coerced to `f64` at the `Value` stage, before the typed `u128` field
//! is ever reached.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread::JoinHandle;

use omni_sumchain::{ChainParamsInfo, JsonRpcTransport, SumChainClient, UreqTransport};
use omni_zkml::ChainClientError;

const SEED: [u8; 32] = [42u8; 32];

/// Find the first occurrence of `needle` in `haystack`.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Spin up a one-shot localhost HTTP server that replies to a single
/// request with a JSON-RPC envelope wrapping `result_body` verbatim.
/// Returns the base URL and the server thread handle.
///
/// The server drains the incoming request (headers + `Content-Length`
/// body) before replying so `ureq` never blocks on a half-read socket.
fn serve_one_jsonrpc_result(result_body: &str) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind localhost ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    let url = format!("http://{addr}/");
    let envelope = format!(r#"{{"jsonrpc":"2.0","id":1,"result":{result_body}}}"#);

    let handle = std::thread::spawn(move || {
        let Ok((mut stream, _)) = listener.accept() else {
            return;
        };

        // Read until the end of the request headers.
        let mut buf: Vec<u8> = Vec::new();
        let mut tmp = [0u8; 1024];
        let mut headers_end: Option<usize> = None;
        while headers_end.is_none() {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&tmp[..n]);
                    headers_end = find_subslice(&buf, b"\r\n\r\n").map(|p| p + 4);
                }
                Err(_) => break,
            }
        }

        // Drain the request body per Content-Length so the client's write
        // completes and the socket is clean before we respond.
        if let Some(he) = headers_end {
            let head = String::from_utf8_lossy(&buf[..he]);
            let content_len = head
                .lines()
                .find_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    lower
                        .strip_prefix("content-length:")
                        .and_then(|v| v.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            let mut remaining = content_len.saturating_sub(buf.len() - he);
            while remaining > 0 {
                match stream.read(&mut tmp) {
                    Ok(0) => break,
                    Ok(n) => remaining = remaining.saturating_sub(n),
                    Err(_) => break,
                }
            }
        }

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            envelope.len(),
            envelope
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
    });

    (url, handle)
}

/// Drive `chain_getChainParams` end-to-end through the real ureq
/// transport, with `min_fee` spliced as the raw JSON token `min_fee_token`.
fn fetch_chain_params_real_http(
    min_fee_token: &str,
) -> Result<ChainParamsInfo, ChainClientError> {
    let body =
        format!(r#"{{"finality_depth":10,"min_fee":{min_fee_token},"chain_id":31337}}"#);
    let (url, handle) = serve_one_jsonrpc_result(&body);
    let client = SumChainClient::new(url, SEED);
    let out = client.get_chain_params();
    let _ = handle.join();
    out
}

// ── Success boundaries via the real transport ────────────────────────────────

#[test]
fn real_transport_min_fee_one() {
    let params = fetch_chain_params_real_http("1").expect("min_fee=1 must parse");
    assert_eq!(params.min_fee, 1u128);
    assert_eq!(params.chain_id, 31337);
}

#[test]
fn real_transport_min_fee_u64_max() {
    let params =
        fetch_chain_params_real_http("18446744073709551615").expect("u64::MAX must parse");
    assert_eq!(params.min_fee, u64::MAX as u128);
}

#[test]
fn real_transport_min_fee_above_u64_max() {
    // u64::MAX + 1 — the exact case the u64 DTO could never represent and
    // that f64 coercion at the `Value` stage would have silently mangled.
    let params =
        fetch_chain_params_real_http("18446744073709551616").expect("u64::MAX+1 must parse");
    assert_eq!(params.min_fee, u64::MAX as u128 + 1);
}

#[test]
fn real_transport_min_fee_u128_max() {
    let params = fetch_chain_params_real_http("340282366920938463463374607431768211455")
        .expect("u128::MAX must parse");
    assert_eq!(params.min_fee, u128::MAX);
}

// ── Rejection boundaries via the real transport ──────────────────────────────

#[test]
fn real_transport_min_fee_negative_is_rejected() {
    let res = fetch_chain_params_real_http("-1");
    assert!(res.is_err(), "negative min_fee must not parse into u128");
}

#[test]
fn real_transport_min_fee_above_u128_max_is_rejected() {
    // u128::MAX + 1.
    let res = fetch_chain_params_real_http("340282366920938463463374607431768211456");
    assert!(res.is_err(), "a value above u128::MAX must not parse");
}

#[test]
fn real_transport_min_fee_fractional_is_rejected() {
    let res = fetch_chain_params_real_http("1.5");
    assert!(res.is_err(), "a fractional min_fee must not parse into u128");
}

#[test]
fn real_transport_min_fee_scientific_is_rejected() {
    // The chain RPC does not emit scientific notation for min_fee; assert
    // we reject it rather than silently accepting a coerced value.
    let res = fetch_chain_params_real_http("1e3");
    assert!(res.is_err(), "scientific-notation min_fee must not parse into u128");
}

// ── Prove the intermediate `serde_json::Value` is exact / non-float ──────────

#[test]
fn real_transport_intermediate_value_preserves_exact_integer() {
    // Call the transport DIRECTLY so we can inspect the raw `Value` that
    // sits between the HTTP body and the typed `ChainParamsInfo`.
    let body = r#"{"finality_depth":10,"min_fee":18446744073709551616,"chain_id":31337}"#;
    let (url, handle) = serve_one_jsonrpc_result(body);
    let transport = UreqTransport::new(url);
    let result = transport
        .call("chain_getChainParams", serde_json::json!([]))
        .expect("transport call must succeed");
    let _ = handle.join();

    let min_fee_val = &result["min_fee"];
    let token = min_fee_val.to_string();

    // (1) The exact integer token survives verbatim. If it had been
    // coerced to f64, this would render as `18446744073709552000` or
    // `1.8446744073709552e19`, never the exact digits below.
    assert_eq!(
        token, "18446744073709551616",
        "intermediate Value dropped the exact integer token (float coercion?)"
    );

    // (2) It is a pure integer token, never floating point: no decimal
    // point and no exponent. This is the direct proof that the number
    // never became an `f64` at the `Value` stage.
    assert!(
        !token.contains('.') && !token.contains('e') && !token.contains('E'),
        "min_fee must remain an exact integer token, never a float: {token}"
    );

    // (3) It round-trips to the exact u128.
    let recovered: u128 =
        serde_json::from_value(min_fee_val.clone()).expect("exact u128 must be recoverable");
    assert_eq!(recovered, u64::MAX as u128 + 1);
}
