//! O3 (#105): the SelectToken result chain.
//!
//! The client side of the autoregressive loop, over token ids only (no
//! tokenizer is registered yet). Each proven SelectToken result is checked
//! against the frozen B0 `SelectToken` statement rules before it extends the
//! sequence:
//!
//! * `selected < vocab_size`;
//! * `eos_flag == (selected == vocab_size − 1)`;
//! * the updated token sequence is the prior sequence followed by `selected`
//!   (u32 little-endian), committed as a `TokenSeq` object. The EOS token is
//!   part of that committed sequence.
//! * the same bytes, as the next step's input, are committed as a
//!   `TokenPrefix` object. The two commitments differ only in object kind, and
//!   a statement binds each to its own field.
//!
//! The stop rule and count are #105's:
//!
//! * generation stops at EOS or after `MAX_OUTPUT_TOKENS` outputs;
//! * the output count excludes EOS;
//! * no step is scheduled after the sequence finishes ("units after EOS never
//!   assigned").

use sumchain_wire::b0::codec::DecodeError;
use sumchain_wire::b0::consts::MAX_OUTPUT_TOKENS;
use sumchain_wire::b0::enums::ObjectKind;
use sumchain_wire::b0::object_commitment::ObjectCommitmentV1;
use sumchain_wire::b0::workload::token_seq_bytes;

/// Why the sequence ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// The last selection was EOS.
    Eos,
    /// `MAX_OUTPUT_TOKENS` non-EOS tokens were produced.
    MaxOutputTokens,
}

/// Why a token or a selection result was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TokenError {
    #[error("vocabulary size must be at least 1")]
    EmptyVocabulary,
    #[error("token {token} is outside the vocabulary of {vocab_size}")]
    OutOfVocabulary { token: u32, vocab_size: u32 },
    #[error(
        "selection {selected} carries eos_flag {eos_flag}, but EOS is token {eos} \
         (vocab_size − 1)"
    )]
    EosFlagMismatch {
        selected: u32,
        eos_flag: u8,
        eos: u32,
    },
    #[error("the sequence already finished ({0:?}); no further step is scheduled")]
    Finished(Finish),
    #[error("token sequence could not be committed: {0}")]
    Commit(DecodeError),
}

/// A prompt plus every selection applied to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSequence {
    vocab_size: u32,
    prompt_len: usize,
    tokens: Vec<u32>,
    output_count: u32,
    finished: Option<Finish>,
}

impl TokenSequence {
    /// Start from a tokenized prompt. Every prompt token must be in the
    /// vocabulary.
    pub fn new(prompt: Vec<u32>, vocab_size: u32) -> Result<Self, TokenError> {
        if vocab_size == 0 {
            return Err(TokenError::EmptyVocabulary);
        }
        if let Some(&token) = prompt.iter().find(|&&t| t >= vocab_size) {
            return Err(TokenError::OutOfVocabulary { token, vocab_size });
        }
        Ok(Self {
            vocab_size,
            prompt_len: prompt.len(),
            tokens: prompt,
            output_count: 0,
            finished: None,
        })
    }

    /// The EOS token id: `vocab_size − 1`.
    pub fn eos(&self) -> u32 {
        self.vocab_size - 1
    }

    /// Apply one proven SelectToken result, checked against the statement
    /// rules. Returns the updated `TokenSeq` commitment, which must equal the
    /// statement's `updated_token_seq_commitment`. A refused result leaves the
    /// sequence unchanged.
    pub fn apply_select(
        &mut self,
        selected: u32,
        eos_flag: u8,
    ) -> Result<ObjectCommitmentV1, TokenError> {
        if let Some(f) = self.finished {
            return Err(TokenError::Finished(f));
        }
        if selected >= self.vocab_size {
            return Err(TokenError::OutOfVocabulary {
                token: selected,
                vocab_size: self.vocab_size,
            });
        }
        let is_eos = selected == self.eos();
        if eos_flag != u8::from(is_eos) {
            return Err(TokenError::EosFlagMismatch {
                selected,
                eos_flag,
                eos: self.eos(),
            });
        }
        let mut tokens = self.tokens.clone();
        tokens.push(selected);
        let commitment = commit(ObjectKind::TokenSeq, &tokens)?;
        self.tokens = tokens;
        if is_eos {
            self.finished = Some(Finish::Eos);
        } else {
            self.output_count += 1;
            if self.output_count >= MAX_OUTPUT_TOKENS {
                self.finished = Some(Finish::MaxOutputTokens);
            }
        }
        Ok(commitment)
    }

    /// Whether another SelectToken step may be scheduled.
    pub fn next_step_allowed(&self) -> bool {
        self.finished.is_none()
    }

    pub fn finished(&self) -> Option<Finish> {
        self.finished
    }

    /// Outputs produced so far, EOS excluded.
    pub fn output_count(&self) -> u32 {
        self.output_count
    }

    /// The produced output tokens, EOS excluded.
    pub fn output_tokens(&self) -> &[u32] {
        let outputs = &self.tokens[self.prompt_len..];
        match self.finished {
            Some(Finish::Eos) => &outputs[..outputs.len() - 1],
            _ => outputs,
        }
    }

    /// Every token so far, prompt first, EOS included if selected.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The sequence so far as a `TokenSeq` object: what the last step
    /// committed as its `updated_token_seq_commitment`.
    pub fn sequence_commitment(&self) -> Result<ObjectCommitmentV1, TokenError> {
        commit(ObjectKind::TokenSeq, &self.tokens)
    }

    /// The sequence so far as a `TokenPrefix` object: the next statement's
    /// `token_prefix`.
    pub fn prefix_commitment(&self) -> Result<ObjectCommitmentV1, TokenError> {
        commit(ObjectKind::TokenPrefix, &self.tokens)
    }
}

fn commit(kind: ObjectKind, tokens: &[u32]) -> Result<ObjectCommitmentV1, TokenError> {
    ObjectCommitmentV1::commit(kind, &token_seq_bytes(tokens)).map_err(TokenError::Commit)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: u32 = 16;
    const EOS: u32 = VOCAB - 1;

    #[test]
    fn eos_stops_the_sequence_and_is_not_counted() {
        let mut s = TokenSequence::new(vec![3, 4], VOCAB).unwrap();
        s.apply_select(7, 0).unwrap();
        s.apply_select(EOS, 1).unwrap();
        assert_eq!(s.finished(), Some(Finish::Eos));
        assert_eq!(s.output_count(), 1);
        assert_eq!(s.output_tokens(), &[7]);
        // EOS stays in the committed sequence.
        assert_eq!(s.tokens(), &[3, 4, 7, EOS]);
        assert!(!s.next_step_allowed());
        assert_eq!(s.apply_select(1, 0), Err(TokenError::Finished(Finish::Eos)));
    }

    #[test]
    fn the_sequence_stops_after_max_output_tokens() {
        let mut s = TokenSequence::new(vec![], VOCAB).unwrap();
        for i in 0..MAX_OUTPUT_TOKENS {
            assert!(s.next_step_allowed());
            s.apply_select(i % EOS, 0).unwrap();
        }
        assert_eq!(s.finished(), Some(Finish::MaxOutputTokens));
        assert_eq!(s.output_count(), MAX_OUTPUT_TOKENS);
        assert_eq!(s.output_tokens().len() as u32, MAX_OUTPUT_TOKENS);
        assert_eq!(
            s.apply_select(0, 0),
            Err(TokenError::Finished(Finish::MaxOutputTokens))
        );
    }

    #[test]
    fn an_immediate_eos_is_an_empty_output() {
        let mut s = TokenSequence::new(vec![1], VOCAB).unwrap();
        s.apply_select(EOS, 1).unwrap();
        assert_eq!(s.output_count(), 0);
        assert!(s.output_tokens().is_empty());
    }

    #[test]
    fn results_that_break_the_statement_rules_are_refused_and_change_nothing() {
        let mut s = TokenSequence::new(vec![2], VOCAB).unwrap();
        let before = s.clone();
        assert_eq!(
            s.apply_select(VOCAB, 0),
            Err(TokenError::OutOfVocabulary {
                token: VOCAB,
                vocab_size: VOCAB
            })
        );
        assert_eq!(
            s.apply_select(EOS, 0),
            Err(TokenError::EosFlagMismatch {
                selected: EOS,
                eos_flag: 0,
                eos: EOS
            })
        );
        assert_eq!(
            s.apply_select(3, 1),
            Err(TokenError::EosFlagMismatch {
                selected: 3,
                eos_flag: 1,
                eos: EOS
            })
        );
        assert_eq!(
            s.apply_select(3, 2),
            Err(TokenError::EosFlagMismatch {
                selected: 3,
                eos_flag: 2,
                eos: EOS
            })
        );
        assert_eq!(s, before);
    }

    #[test]
    fn a_prompt_outside_the_vocabulary_is_refused() {
        assert_eq!(
            TokenSequence::new(vec![1, VOCAB], VOCAB),
            Err(TokenError::OutOfVocabulary {
                token: VOCAB,
                vocab_size: VOCAB
            })
        );
        assert_eq!(
            TokenSequence::new(vec![], 0),
            Err(TokenError::EmptyVocabulary)
        );
    }

    #[test]
    fn each_step_commits_to_the_prior_sequence_followed_by_the_selection() {
        let mut s = TokenSequence::new(vec![9, 8], VOCAB).unwrap();
        let prefix = s.prefix_commitment().unwrap();
        let updated = s.apply_select(5, 0).unwrap();
        assert_ne!(prefix.merkle_root(), updated.merkle_root());
        assert_eq!(updated, s.sequence_commitment().unwrap());
        // The same bytes as the next step's input: another object kind.
        let next_prefix = s.prefix_commitment().unwrap();
        assert_eq!(next_prefix.merkle_root(), updated.merkle_root());
        assert_eq!(next_prefix.object_kind(), ObjectKind::TokenPrefix);
        assert_eq!(updated.object_kind(), ObjectKind::TokenSeq);
        let mut bytes = token_seq_bytes(&[9, 8]);
        bytes.extend_from_slice(&5u32.to_le_bytes());
        assert_eq!(
            updated,
            ObjectCommitmentV1::commit(ObjectKind::TokenSeq, &bytes).unwrap()
        );
    }
}
