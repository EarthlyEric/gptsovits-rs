use rand::distributions::WeightedIndex;
use rand::prelude::*;

/// Autoregressive token sampler matching GPT-SoVITS sampling order.
pub fn sample_next_token(
    logits: &[f32],
    history_tokens: &[i64],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
) -> i64 {
    if logits.is_empty() {
        return 1024;
    }

    let mut modified_logits = logits.to_vec();

    // 1. Repetition penalty. GPT-SoVITS applies this to the complete y history,
    // including the semantic prompt tokens.
    if (repetition_penalty - 1.0).abs() > 1e-4 {
        let mut penalized = vec![false; modified_logits.len()];
        for &token in history_tokens {
            if token >= 0 {
                let idx = token as usize;
                if idx < modified_logits.len() && !penalized[idx] {
                    penalized[idx] = true;
                    let score = modified_logits[idx];
                    if score < 0.0 {
                        modified_logits[idx] = score * repetition_penalty;
                    } else {
                        modified_logits[idx] = score / repetition_penalty;
                    }
                }
            }
        }
    }

    // 2. Nucleus filtering is performed on the unscaled logits in upstream.
    if top_p.is_finite() && top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = modified_logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((_, max_logit)) = indexed.first().copied() {
            let exp_scores: Vec<f32> = indexed
                .iter()
                .map(|(_, score)| (*score - max_logit).exp())
                .collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            if sum_exp > 0.0 && sum_exp.is_finite() {
                let mut cumulative = 0.0;
                for (rank, ((index, _), exp_score)) in indexed.iter().zip(exp_scores).enumerate() {
                    cumulative += exp_score / sum_exp;
                    if cumulative > top_p && rank > 0 {
                        modified_logits[*index] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    // 3. Temperature scaling.
    let temp = if temperature.is_finite() {
        temperature.max(1e-5)
    } else {
        1.0
    };
    for score in &mut modified_logits {
        if score.is_finite() {
            *score /= temp;
        }
    }

    // 4. Top-K filtering.
    let mut indexed: Vec<(usize, f32)> = modified_logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if indexed.is_empty() {
        return 1024;
    }

    let k = top_k.max(1).min(indexed.len());
    let top_k_cutoff = indexed[k - 1].1;
    for value in &mut modified_logits {
        if !value.is_finite() || *value < top_k_cutoff {
            *value = f32::NEG_INFINITY;
        }
    }

    // 5. Softmax.
    let max_logit = modified_logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let mut exp_scores = Vec::with_capacity(modified_logits.len());
    let mut sum_exp = 0.0f32;
    for &score in &modified_logits {
        if !score.is_finite() {
            exp_scores.push(0.0);
        } else {
            let exp_val = (score - max_logit).exp();
            exp_scores.push(exp_val);
            sum_exp += exp_val;
        }
    }

    if sum_exp <= 0.0 {
        return 1024; // EOS fallback
    }

    // 6. Sample from the remaining distribution.
    let probs: Vec<(usize, f32)> = exp_scores
        .into_iter()
        .enumerate()
        .map(|(i, val)| (i, val / sum_exp))
        .collect();
    let weights: Vec<f32> = probs.iter().map(|(_, p)| *p).collect();
    if let Ok(dist) = WeightedIndex::new(&weights) {
        let mut rng = rand::thread_rng();
        let selected_idx = dist.sample(&mut rng);
        probs[selected_idx].0 as i64
    } else {
        // Fallback: argmax
        indexed[0].0 as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_logits_return_eos() {
        assert_eq!(sample_next_token(&[], &[], 1.0, 1, 1.0, 1.0), 1024);
    }

    #[test]
    fn top_k_one_is_argmax() {
        for _ in 0..8 {
            assert_eq!(
                sample_next_token(&[0.1, 4.0, 2.0], &[], 1.0, 1, 1.0, 1.0),
                1
            );
        }
    }

    #[test]
    fn non_finite_logits_do_not_escape_distribution() {
        for _ in 0..8 {
            let token =
                sample_next_token(&[f32::NAN, f32::NEG_INFINITY, 3.0], &[], 1.0, 3, 1.0, 1.0);
            assert_eq!(token, 2);
        }
    }

    #[test]
    fn repeated_history_tokens_are_penalized_once() {
        let logits = [10.0, 3.5];
        let once = sample_next_token(&logits, &[0], 1.0, 1, 1.0, 2.0);
        let repeated = sample_next_token(&logits, &[0, 0], 1.0, 1, 1.0, 2.0);

        assert_eq!(once, 0);
        assert_eq!(repeated, once);
    }
}
