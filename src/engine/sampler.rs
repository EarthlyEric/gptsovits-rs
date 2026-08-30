use rand::distributions::WeightedIndex;
use rand::prelude::*;

/// Autoregressive token sampler with temperature, top_k, top_p, and sliding-window repetition penalty
pub fn sample_next_token(
    logits: &[f32],
    history_tokens: &[i64],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
) -> i64 {
    let mut modified_logits = logits.to_vec();

    // 1. Sliding window repetition penalty (only penalize recent tokens to allow natural phonetic reuse)
    if (repetition_penalty - 1.0).abs() > 1e-4 {
        let window_start = history_tokens.len().saturating_sub(16);
        for &token in &history_tokens[window_start..] {
            let idx = token as usize;
            if idx < modified_logits.len() {
                let score = modified_logits[idx];
                if score < 0.0 {
                    modified_logits[idx] = score * repetition_penalty;
                } else {
                    modified_logits[idx] = score / repetition_penalty;
                }
            }
        }
    }

    // 2. Temperature scaling
    let temp = temperature.max(1e-5);
    for score in &mut modified_logits {
        *score /= temp;
    }

    // 3. Top-K filtering
    let mut indexed: Vec<(usize, f32)> = modified_logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k = top_k.clamp(1, indexed.len());
    let top_k_cutoff = indexed[k - 1].1;
    for val in modified_logits.iter_mut() {
        if *val < top_k_cutoff {
            *val = f32::NEG_INFINITY;
        }
    }

    // 4. Softmax
    let max_logit = modified_logits
        .iter()
        .cloned()
        .filter(|v| !v.is_nan() && *v != f32::NEG_INFINITY)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut exp_scores = Vec::with_capacity(modified_logits.len());
    let mut sum_exp = 0.0f32;
    for &score in &modified_logits {
        if score == f32::NEG_INFINITY {
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

    let mut probs: Vec<(usize, f32)> = exp_scores
        .into_iter()
        .enumerate()
        .map(|(i, val)| (i, val / sum_exp))
        .collect();

    // 5. Top-P (Nucleus) filtering
    if top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum_prob = 0.0f32;
        let mut cut_idx = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            cum_prob += p;
            if cum_prob > top_p && i > 0 {
                cut_idx = i + 1;
                break;
            }
        }
        probs.truncate(cut_idx);
    }

    // 6. Sample from distribution
    let weights: Vec<f32> = probs.iter().map(|(_, p)| *p).collect();
    if let Ok(dist) = WeightedIndex::new(&weights) {
        let mut rng = rand::thread_rng();
        let selected_idx = dist.sample(&mut rng);
        probs[selected_idx].0 as i64
    } else {
        // Fallback: argmax
        probs[0].0 as i64
    }
}
