//! The candidate-neutral integer workload, as a HOST-SIDE reference only.
//! Nothing here is proven — no zkVM guest runs. This is the plain-Rust model of
//! the computation both candidates would eventually prove, and it emits the
//! candidate-neutral 996-byte statement as a zero-spec-hash
//! [`SyntheticJournal`](crate::statement::SyntheticJournal) **template**.
//!
//! A **bounded integer decoder-only transformer layer-group unit** (integer
//! RMSNorm → causal attention with historical KV → FFN, residual-connected) and a
//! **SelectToken unit** (final-norm → LM-head → integer argmax, ties broken to the
//! lowest token id). Everything is `i16` at fixed-point scale `S = 2^8`.
//!
//! Raw residual / KV / token bytes come from the **frozen B0 workload byte
//! helpers** ([`residual_state_bytes`](crate::b0::workload::residual_state_bytes),
//! [`kv_state_bytes`](crate::b0::workload::kv_state_bytes),
//! [`token_seq_bytes`](crate::b0::workload::token_seq_bytes)), so the committed
//! object bytes are byte-identical to the frozen encoders. The attention uses a
//! deterministic ReLU-normalized stand-in for softmax; it is a harness workload,
//! not a fidelity claim, and it is NOT the frozen production algorithm.

use crate::b0::codec::DecodeError;
use crate::b0::consts;
use crate::b0::derived_input::DerivedInputV1;
use crate::b0::enums::{InputSlotKind, ObjectKind, SlotKind, UnitKind};
use crate::b0::manifest::{
    InputManifestV1, InputSlotDescriptorV1, OutputManifestV1, SlotDescriptorV1,
};
use crate::b0::workload::{kv_state_bytes, residual_state_bytes, token_seq_bytes, KvPair};
use crate::manifest::output_slot;
use crate::object::ObjectCommitmentV1;
use crate::statement::{R0ComputationStatementV2, SyntheticJournal};
use crate::FIXED_POINT_SCALE_LOG2;

// ── Frozen bounded dimensions (must equal the B0 frozen scalars) ─────────────
pub const D_MODEL: usize = 8;
pub const N_HEADS: usize = 2;
pub const HEAD_DIM: usize = D_MODEL / N_HEADS; // 4
pub const FFN_DIM: usize = 16;
pub const VOCAB: usize = 16;
pub const MAX_SEQ: usize = 8;
/// End-of-sequence token id (the last vocab id).
pub const EOS_TOKEN: u32 = (VOCAB - 1) as u32;

const SCALE_LOG2: u32 = FIXED_POINT_SCALE_LOG2 as u32;

// ── Integer primitives (pattern reused from the halo2 reference) ─────────────

/// Saturate an `i64` to the `i16` range (never wraps).
#[inline]
pub fn saturate_to_i16(x: i64) -> i16 {
    x.clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

/// Element-wise ReLU on an `i16`.
#[inline]
pub fn relu_i16(x: i16) -> i16 {
    if x > 0 {
        x
    } else {
        0
    }
}

/// Round-half-away-from-zero division by `2^scale_log2` (requantization).
#[inline]
pub fn requantize(n: i64, scale_log2: u32) -> i64 {
    if scale_log2 == 0 {
        return n;
    }
    let half: i64 = 1i64 << (scale_log2 - 1);
    if n >= 0 {
        (n + half) >> scale_log2
    } else {
        -(((-n) + half) >> scale_log2)
    }
}

/// Round-half-away-from-zero division by an arbitrary positive divisor `d`.
#[inline]
pub fn round_half_away_div(n: i64, d: i64) -> i64 {
    debug_assert!(d > 0);
    let half = d / 2;
    if n >= 0 {
        (n + half) / d
    } else {
        -(((-n) + half) / d)
    }
}

/// Integer square root of a non-negative `i64` (floor).
#[inline]
pub fn isqrt_i64(v: i64) -> i64 {
    debug_assert!(v >= 0);
    if v < 2 {
        return v;
    }
    let mut x = v;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + v / x) / 2;
    }
    x
}

/// Saturating element-wise `i16` residual add.
fn residual_add(a: &[i16; D_MODEL], b: &[i16; D_MODEL]) -> [i16; D_MODEL] {
    let mut out = [0i16; D_MODEL];
    for i in 0..D_MODEL {
        out[i] = saturate_to_i16(a[i] as i64 + b[i] as i64);
    }
    out
}

/// Integer RMSNorm: `out_i = saturate((x_i * gamma_i) / rms)`, `rms =
/// isqrt(mean(x^2)) + 1` (the `+1` guarantees a positive divisor).
fn rmsnorm(x: &[i16; D_MODEL], gamma: &[i16; D_MODEL]) -> [i16; D_MODEL] {
    let mut sum_sq: i64 = 0;
    for &v in x.iter() {
        sum_sq += (v as i64) * (v as i64);
    }
    let mean_sq = sum_sq / (D_MODEL as i64);
    let rms = isqrt_i64(mean_sq) + 1;
    let mut out = [0i16; D_MODEL];
    for i in 0..D_MODEL {
        let n = (x[i] as i64) * (gamma[i] as i64);
        out[i] = saturate_to_i16(round_half_away_div(n, rms));
    }
    out
}

/// Integer dense (no bias): `out_j = saturate(requantize(Σ_i in_i * W[i][j], S))`.
fn dense(input: &[i16], w: &[Vec<i16>], out_dim: usize) -> Vec<i16> {
    let mut out = vec![0i16; out_dim];
    for (j, o) in out.iter_mut().enumerate() {
        let mut acc: i64 = 0;
        for (i, &xi) in input.iter().enumerate() {
            acc += (xi as i64) * (w[i][j] as i64);
        }
        *o = saturate_to_i16(requantize(acc, SCALE_LOG2));
    }
    out
}

/// Integer dense projecting any-length `input` to a fixed `[i16; D_MODEL]`
/// output (no fallible conversion — the output width is a compile-time array).
fn dense_d_model(input: &[i16], w: &[Vec<i16>]) -> [i16; D_MODEL] {
    let mut out = [0i16; D_MODEL];
    for (j, o) in out.iter_mut().enumerate() {
        let mut acc: i64 = 0;
        for (i, &xi) in input.iter().enumerate() {
            acc += (xi as i64) * (w[i][j] as i64);
        }
        *o = saturate_to_i16(requantize(acc, SCALE_LOG2));
    }
    out
}

// ── Model (the authenticated weights) ────────────────────────────────────────

/// The bounded transformer's frozen integer weights. Deterministically derived
/// from a seed so tests can pin behaviour; the *bytes* are what the public
/// `model_commitment` binds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    /// Caller-assigned 32-byte model id (distinct from the content commitment).
    pub model_id: [u8; 32],
    pub attn_norm_gamma: [i16; D_MODEL],
    pub wq: Vec<Vec<i16>>,
    pub wk: Vec<Vec<i16>>,
    pub wv: Vec<Vec<i16>>,
    pub wo: Vec<Vec<i16>>,
    pub ffn_norm_gamma: [i16; D_MODEL],
    pub w1: Vec<Vec<i16>>,
    pub w2: Vec<Vec<i16>>,
    pub final_norm_gamma: [i16; D_MODEL],
    pub w_lmhead: Vec<Vec<i16>>,
}

/// Magic + version prefixing the canonical model bytes.
const MODEL_MAGIC: u32 = 0x5230_4d44; // "R0MD"
const MODEL_VERSION: u32 = 1;

fn lcg_next(state: &mut u64) -> i16 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let v = ((*state >> 33) & 0x7fff_ffff) as i64;
    ((v % 129) - 64) as i16 // range [-64, 64]
}

fn matrix(state: &mut u64, rows: usize, cols: usize) -> Vec<Vec<i16>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| lcg_next(state)).collect())
        .collect()
}

fn gamma(state: &mut u64) -> [i16; D_MODEL] {
    let mut g = [0i16; D_MODEL];
    for x in g.iter_mut() {
        *x = 256 + (lcg_next(state) as i32).clamp(-64, 64) as i16;
    }
    g
}

impl Model {
    /// Deterministically construct a model from `seed` + `model_id`.
    pub fn deterministic(seed: u64, model_id: [u8; 32]) -> Self {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        Self {
            model_id,
            attn_norm_gamma: gamma(&mut s),
            wq: matrix(&mut s, D_MODEL, D_MODEL),
            wk: matrix(&mut s, D_MODEL, D_MODEL),
            wv: matrix(&mut s, D_MODEL, D_MODEL),
            wo: matrix(&mut s, D_MODEL, D_MODEL),
            ffn_norm_gamma: gamma(&mut s),
            w1: matrix(&mut s, D_MODEL, FFN_DIM),
            w2: matrix(&mut s, FFN_DIM, D_MODEL),
            final_norm_gamma: gamma(&mut s),
            w_lmhead: matrix(&mut s, D_MODEL, VOCAB),
        }
    }

    /// Canonical model bytes: `MODEL_MAGIC ‖ MODEL_VERSION ‖ all weights (LE i16,
    /// fixed order)`. This is the opaque object the `model_commitment` binds.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MODEL_MAGIC.to_le_bytes());
        b.extend_from_slice(&MODEL_VERSION.to_le_bytes());
        push_vec(&mut b, &self.attn_norm_gamma);
        push_mat(&mut b, &self.wq);
        push_mat(&mut b, &self.wk);
        push_mat(&mut b, &self.wv);
        push_mat(&mut b, &self.wo);
        push_vec(&mut b, &self.ffn_norm_gamma);
        push_mat(&mut b, &self.w1);
        push_mat(&mut b, &self.w2);
        push_vec(&mut b, &self.final_norm_gamma);
        push_mat(&mut b, &self.w_lmhead);
        b
    }

    /// Public B0 `ObjectCommitmentV1` (`object_kind = Model`) over the canonical
    /// model bytes. Fallible, per the B0 checked constructor.
    pub fn commitment(&self) -> Result<ObjectCommitmentV1, DecodeError> {
        ObjectCommitmentV1::commit(ObjectKind::Model, &self.canonical_bytes())
    }
}

fn push_vec(buf: &mut Vec<u8>, v: &[i16]) {
    for &x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}
fn push_mat(buf: &mut Vec<u8>, m: &[Vec<i16>]) {
    for row in m {
        push_vec(buf, row);
    }
}

// ── The transformer layer-group unit ─────────────────────────────────────────

/// Per-position KV history for the single modeled layer (scale-S vectors).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvCache {
    pub k: Vec<[i16; D_MODEL]>,
    pub v: Vec<[i16; D_MODEL]>,
}

impl KvCache {
    /// `(key, value)` pairs in the B0 [`KvPair`] shape.
    pub fn pairs(&self) -> Vec<KvPair> {
        self.k.iter().copied().zip(self.v.iter().copied()).collect()
    }

    /// Raw KV bytes via the frozen B0 encoder (32 bytes per `(key, value)` pair).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        kv_state_bytes(&self.pairs())
    }
}

/// Output of the transformer layer-group unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerGroupOutput {
    pub new_residual: [i16; D_MODEL],
    pub updated_kv: KvCache,
    pub residual_slot: SlotDescriptorV1,
    pub kv_slot: SlotDescriptorV1,
    pub output_manifest: OutputManifestV1,
}

/// Run the bounded integer transformer layer group for one position.
// Range loops index parallel arrays by the same head-dim / position counter, so
// explicit indexing is clearer than zipped iterators.
#[allow(clippy::needless_range_loop)]
pub fn run_layer_group(
    model: &Model,
    prior_residual: &[i16; D_MODEL],
    prior_kv: &KvCache,
    position: u32,
) -> Result<LayerGroupOutput, DecodeError> {
    let x = *prior_residual;

    // 1. Pre-attention RMSNorm.
    let xn = rmsnorm(&x, &model.attn_norm_gamma);

    // 2. Q/K/V projections.
    let q = dense_d_model(&xn, &model.wq);
    let k = dense_d_model(&xn, &model.wk);
    let v = dense_d_model(&xn, &model.wv);

    // 3. Append current K/V to the historical cache.
    let mut kv = prior_kv.clone();
    kv.k.push(k);
    kv.v.push(v);
    let seq = kv.k.len(); // = position + 1

    // 4. Causal relu-normalized attention per head.
    let mut attn_out = [0i16; D_MODEL];
    for h in 0..N_HEADS {
        let lo = h * HEAD_DIM;
        let hi = lo + HEAD_DIM;
        let mut weights = vec![0i64; seq];
        let mut sum_w: i64 = 0;
        for (j, w_slot) in weights.iter_mut().enumerate() {
            let mut score: i64 = 0;
            for d in lo..hi {
                score += (q[d] as i64) * (kv.k[j][d] as i64);
            }
            let sq = requantize(score, SCALE_LOG2);
            let w = relu_i16(saturate_to_i16(sq)) as i64;
            *w_slot = w;
            sum_w += w;
        }
        for d in lo..hi {
            let ctx = if sum_w == 0 {
                let mut acc: i64 = 0;
                for j in 0..seq {
                    acc += kv.v[j][d] as i64;
                }
                round_half_away_div(acc, seq as i64)
            } else {
                let mut acc: i64 = 0;
                for j in 0..seq {
                    acc += weights[j] * (kv.v[j][d] as i64);
                }
                round_half_away_div(acc, sum_w)
            };
            attn_out[d] = saturate_to_i16(ctx);
        }
    }

    // 5. Output projection + residual.
    let o = dense_d_model(&attn_out, &model.wo);
    let x = residual_add(&x, &o);

    // 6. Pre-FFN RMSNorm + FFN + residual.
    let xn2 = rmsnorm(&x, &model.ffn_norm_gamma);
    let h1_raw = dense(&xn2, &model.w1, FFN_DIM);
    let h1: Vec<i16> = h1_raw.iter().map(|&z| relu_i16(z)).collect();
    let y = dense_d_model(&h1, &model.w2);
    let new_residual = residual_add(&x, &y);

    // 7. Commit the output slots (each carries its OWN length/root, under the
    //    object kind its slot kind maps to: ResidualStream→ResidualState,
    //    KvCache→KvState).
    let residual_slot = output_slot(
        SlotKind::ResidualStream,
        position,
        &residual_state_bytes(&new_residual),
    )?;
    let kv_slot = output_slot(SlotKind::KvCache, position, &kv.canonical_bytes())?;
    let output_manifest = OutputManifestV1 {
        slots: vec![residual_slot.clone(), kv_slot.clone()],
    };

    Ok(LayerGroupOutput {
        new_residual,
        updated_kv: kv,
        residual_slot,
        kv_slot,
        output_manifest,
    })
}

// ── The SelectToken unit ─────────────────────────────────────────────────────

/// Output of the SelectToken unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectTokenOutput {
    pub selected_token: u32,
    pub eos_flag: bool,
    pub logits: [i16; VOCAB],
    pub updated_token_seq: ObjectCommitmentV1,
    pub token_seq_bytes: Vec<u8>,
}

/// Final-norm → LM head → integer argmax (ties → lowest token id).
pub fn run_select_token(
    model: &Model,
    final_residual: &[i16; D_MODEL],
    prev_tokens: &[u32],
) -> Result<SelectTokenOutput, DecodeError> {
    let xn = rmsnorm(final_residual, &model.final_norm_gamma);
    let logits_v = dense(&xn, &model.w_lmhead, VOCAB);
    let mut logits = [0i16; VOCAB];
    logits.copy_from_slice(&logits_v);

    // Integer argmax with lowest-id tie-break.
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (i, &l) in logits.iter().enumerate().skip(1) {
        if l > best_val {
            best_val = l;
            best_idx = i;
        }
    }
    let selected_token = best_idx as u32;
    let eos_flag = selected_token == EOS_TOKEN;

    let mut tokens = prev_tokens.to_vec();
    tokens.push(selected_token);
    let bytes = token_seq_bytes(&tokens);
    let updated_token_seq = ObjectCommitmentV1::commit(ObjectKind::TokenSeq, &bytes)?;

    Ok(SelectTokenOutput {
        selected_token,
        eos_flag,
        logits,
        updated_token_seq,
        token_seq_bytes: bytes,
    })
}

// ── Reference executor: produces the candidate-neutral statement template ────

/// Shared inputs identifying a generation step. The tokenizer is represented by
/// the frozen `tokenizer_id` context field — there is NO tokenizer object
/// commitment (`ObjectKind::Tokenizer` is reserved and rejected).
#[derive(Clone, Debug)]
pub struct StepContext {
    pub job_id: [u8; 32],
    pub session_id: [u8; 32],
    pub unit_id: [u8; 32],
    pub unit_index: u32,
    pub generation_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub sequence_length: u32,
    pub position: u32,
    pub tokenizer_id: [u8; 32],
}

impl StepContext {
    /// A minimal deterministic context for tests / benchmarks.
    pub fn sample(position: u32) -> Self {
        Self {
            job_id: crate::blake3_32(b"job"),
            session_id: crate::blake3_32(b"session"),
            unit_id: crate::blake3_32(b"unit"),
            unit_index: 0,
            generation_index: position,
            layer_start: 0,
            layer_end: 1,
            sequence_length: position + 1,
            position,
            tokenizer_id: crate::blake3_32(b"tokenizer"),
        }
    }
}

/// Typed unit inputs: the prior residual stream, prior KV cache, and token
/// prefix. The executor commits these (and builds the B0 input manifest +
/// derived input) internally.
pub struct UnitInputs<'a> {
    pub prior_residual: &'a [i16; D_MODEL],
    pub prior_kv: &'a KvCache,
    pub token_prefix: &'a [u32],
}

/// The plain-Rust reference executor. Deterministic and pure.
pub struct ReferenceExecutor;

impl ReferenceExecutor {
    /// Run the transformer layer-group unit and build its journal template.
    pub fn run_layer_group_unit(
        model: &Model,
        ctx: &StepContext,
        inputs: &UnitInputs,
    ) -> Result<(LayerGroupOutput, SyntheticJournal), DecodeError> {
        let out = run_layer_group(model, inputs.prior_residual, inputs.prior_kv, ctx.position)?;
        let mut s = Self::base_statement(model, ctx, inputs)?;
        s.unit_kind = UnitKind::TransformerLayerGroup;
        s.output_manifest = out.output_manifest.try_commitment()?;
        s.selected_token = u32::MAX;
        s.updated_token_seq_commitment = ObjectCommitmentV1::empty(ObjectKind::TokenSeq);
        s.eos_flag = 0;
        Ok((out, SyntheticJournal::from_statement_template(s)?))
    }

    /// Run the SelectToken unit and build its journal template.
    pub fn run_select_token_unit(
        model: &Model,
        ctx: &StepContext,
        inputs: &UnitInputs,
        final_residual: &[i16; D_MODEL],
        prev_tokens: &[u32],
    ) -> Result<(SelectTokenOutput, SyntheticJournal), DecodeError> {
        let out = run_select_token(model, final_residual, prev_tokens)?;
        let mut s = Self::base_statement(model, ctx, inputs)?;
        s.unit_kind = UnitKind::SelectToken;
        // The select-token unit emits an empty output manifest (its product is
        // the selected token + updated token sequence).
        let empty_manifest = OutputManifestV1 { slots: vec![] };
        s.output_manifest = empty_manifest.try_commitment()?;
        s.selected_token = out.selected_token;
        s.updated_token_seq_commitment = out.updated_token_seq.clone();
        s.eos_flag = out.eos_flag as u8;
        Ok((out, SyntheticJournal::from_statement_template(s)?))
    }

    /// Build the common statement fields (inputs + derived input + input
    /// manifest), leaving the output half for the caller to fill.
    fn base_statement(
        model: &Model,
        ctx: &StepContext,
        inputs: &UnitInputs,
    ) -> Result<R0ComputationStatementV2, DecodeError> {
        let model_commitment = model.commitment()?;

        // Input commitments over the frozen B0 workload byte encodings.
        let prior_residual_stream = ObjectCommitmentV1::commit(
            ObjectKind::PriorResidual,
            &residual_state_bytes(inputs.prior_residual),
        )?;
        let prior_kv_cache =
            ObjectCommitmentV1::commit(ObjectKind::PriorKv, &inputs.prior_kv.canonical_bytes())?;
        let token_prefix = ObjectCommitmentV1::commit(
            ObjectKind::TokenPrefix,
            &token_seq_bytes(inputs.token_prefix),
        )?;

        // The B0 input manifest reuses the three input commitments as its slots.
        let input_manifest_v1 = InputManifestV1 {
            slots: vec![
                InputSlotDescriptorV1 {
                    slot_kind: InputSlotKind::PriorResidual,
                    slot_index: 0,
                    commitment: prior_residual_stream.clone(),
                },
                InputSlotDescriptorV1 {
                    slot_kind: InputSlotKind::PriorKv,
                    slot_index: 0,
                    commitment: prior_kv_cache.clone(),
                },
                InputSlotDescriptorV1 {
                    slot_kind: InputSlotKind::TokenPrefix,
                    slot_index: 0,
                    commitment: token_prefix.clone(),
                },
            ],
        };
        let input_manifest = input_manifest_v1.try_commitment()?;

        // The B0 derived input (350 bytes) binds the identities of the model and
        // the three input commitments.
        let derived = DerivedInputV1 {
            job_id: ctx.job_id,
            session_id: ctx.session_id,
            unit_id: ctx.unit_id,
            generation_index: ctx.generation_index,
            model_id: model.model_id,
            model_commitment_identity: model_commitment.identity(),
            layer_start: ctx.layer_start,
            layer_end: ctx.layer_end,
            prior_residual_commitment_identity: prior_residual_stream.identity(),
            prior_kv_commitment_identity: prior_kv_cache.identity(),
            token_prefix_commitment_identity: token_prefix.identity(),
            position: ctx.position,
            sequence_length: ctx.sequence_length,
        };
        let derived_input_commitment =
            ObjectCommitmentV1::commit(ObjectKind::DerivedInput, &derived.encode())?;

        Ok(R0ComputationStatementV2 {
            b0_pre_spec_hash: [0u8; 32],
            job_id: ctx.job_id,
            session_id: ctx.session_id,
            unit_id: ctx.unit_id,
            unit_kind: UnitKind::TransformerLayerGroup, // overwritten by caller
            unit_index: ctx.unit_index,
            generation_index: ctx.generation_index,
            model_id: model.model_id,
            model_commitment,
            tokenizer_id: ctx.tokenizer_id,
            head_dim: HEAD_DIM as u16,
            ffn_dim: FFN_DIM as u16,
            layer_start: ctx.layer_start,
            layer_end: ctx.layer_end,
            vocab_size: VOCAB as u32,
            d_model: D_MODEL as u32,
            n_heads: N_HEADS as u32,
            derived_input_commitment,
            prior_residual_stream,
            prior_kv_cache,
            token_prefix,
            input_manifest,
            sequence_length: ctx.sequence_length,
            position: ctx.position,
            output_manifest: ObjectCommitmentV1::empty(ObjectKind::OutputManifest), // overwritten
            selected_token: u32::MAX,                                               // overwritten
            updated_token_seq_commitment: ObjectCommitmentV1::empty(ObjectKind::TokenSeq), // overwritten
            eos_flag: 0, // overwritten
            max_cycles: consts::MAX_CYCLES,
            max_d_model: consts::MAX_D_MODEL,
            max_seq_len: consts::MAX_SEQ_LEN,
            max_output_tokens: consts::MAX_OUTPUT_TOKENS,
            max_manifest_slots: consts::MAX_MANIFEST_SLOTS,
            max_state_bytes: consts::MAX_STATE_BYTES,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Model {
        Model::deterministic(42, crate::blake3_32(b"model-a"))
    }

    fn inputs<'a>(
        residual: &'a [i16; D_MODEL],
        kv: &'a KvCache,
        prefix: &'a [u32],
    ) -> UnitInputs<'a> {
        UnitInputs {
            prior_residual: residual,
            prior_kv: kv,
            token_prefix: prefix,
        }
    }

    #[test]
    fn workload_dims_match_b0_frozen_scalars() {
        assert_eq!(D_MODEL as u32, consts::D_MODEL);
        assert_eq!(N_HEADS as u32, consts::N_HEADS);
        assert_eq!(HEAD_DIM as u16, consts::HEAD_DIM);
        assert_eq!(FFN_DIM as u16, consts::FFN_DIM);
        assert_eq!(VOCAB as u32, consts::VOCAB_SIZE);
        assert_eq!(MAX_SEQ as u32, consts::MAX_SEQ);
    }

    #[test]
    fn primitives_match_reference_pattern() {
        assert_eq!(saturate_to_i16(100_000), i16::MAX);
        assert_eq!(saturate_to_i16(-100_000), i16::MIN);
        assert_eq!(relu_i16(-3), 0);
        assert_eq!(requantize(128, 8), 1);
        assert_eq!(round_half_away_div(5, 2), 3);
        assert_eq!(isqrt_i64(65536), 256);
    }

    #[test]
    fn layer_group_is_deterministic_and_bounded() {
        let m = model();
        let x = [10i16, -20, 30, -40, 50, -60, 70, -80];
        let kv = KvCache::default();
        let a = run_layer_group(&m, &x, &kv, 0).unwrap();
        let b = run_layer_group(&m, &x, &kv, 0).unwrap();
        assert_eq!(a, b);
        for v in a.new_residual {
            assert!((i16::MIN..=i16::MAX).contains(&v));
        }
        assert_eq!(a.updated_kv.k.len(), 1);
    }

    #[test]
    fn causal_attention_consumes_history() {
        let m = model();
        let x_a = [1i16, 2, 3, 4, 5, 6, 7, 8];
        let x_b = [80i16, -70, 60, -50, 40, -30, 20, -10];
        let step0 = run_layer_group(&m, &x_a, &KvCache::default(), 0).unwrap();
        let with_history = run_layer_group(&m, &x_b, &step0.updated_kv, 1).unwrap();
        assert_eq!(with_history.updated_kv.k.len(), 2);
        let no_history = run_layer_group(&m, &x_b, &KvCache::default(), 0).unwrap();
        assert_ne!(with_history.new_residual, no_history.new_residual);
    }

    #[test]
    fn select_token_argmax_ties_to_lowest_id() {
        let m = model();
        let x = [3i16, -3, 3, -3, 3, -3, 3, -3];
        let out = run_select_token(&m, &x, &[1, 2, 3]).unwrap();
        let mut want = 0usize;
        for i in 1..VOCAB {
            if out.logits[i] > out.logits[want] {
                want = i;
            }
        }
        assert_eq!(out.selected_token as usize, want);
        assert_eq!(out.token_seq_bytes.len(), 4 * 4); // 3 prev + 1 new
    }

    #[test]
    fn reference_executor_journal_is_deterministic_996_template() {
        let m = model();
        let ctx = StepContext::sample(0);
        let x = [7i16, -7, 7, -7, 7, -7, 7, -7];
        let kv = KvCache::default();
        let (_o1, j1) =
            ReferenceExecutor::run_layer_group_unit(&m, &ctx, &inputs(&x, &kv, &[])).unwrap();
        let (_o2, j2) =
            ReferenceExecutor::run_layer_group_unit(&m, &ctx, &inputs(&x, &kv, &[])).unwrap();
        assert_eq!(j1.bytes(), j2.bytes());
        assert_eq!(j1.len(), 996);
        // Template discipline: spec-hash field zeroed.
        assert_eq!(j1.spec_hash(), [0u8; 32]);
        // The journal round-trips through the strict B0 statement decoder.
        let s = j1.decode().unwrap();
        assert_eq!(s.d_model, D_MODEL as u32);
        assert_eq!(s.vocab_size, VOCAB as u32);
        assert_eq!(s.unit_kind, UnitKind::TransformerLayerGroup);
    }

    #[test]
    fn flipping_a_weight_byte_changes_model_commitment_and_journal() {
        let m0 = model();
        let mut m1 = m0.clone();
        m1.wq[0][0] = m1.wq[0][0].wrapping_add(1);
        assert_ne!(
            m0.commitment().unwrap().identity(),
            m1.commitment().unwrap().identity()
        );

        let ctx = StepContext::sample(0);
        let x = [1i16; D_MODEL];
        let kv = KvCache::default();
        let (_a, j0) =
            ReferenceExecutor::run_layer_group_unit(&m0, &ctx, &inputs(&x, &kv, &[])).unwrap();
        let (_b, j1) =
            ReferenceExecutor::run_layer_group_unit(&m1, &ctx, &inputs(&x, &kv, &[])).unwrap();
        assert_ne!(
            j0.computation_statement_hash(),
            j1.computation_statement_hash()
        );
    }
}
