use ndarray::Array2;

/// Align character-level BERT hidden states (1024-dim) with phonemes using word2ph counts
/// Each character's embedding vector is repeated word2ph[i] times.
pub fn align_bert_to_phones(
    char_bert: Option<&Array2<f32>>,
    word2ph: &[usize],
    num_phones: usize,
) -> Array2<f32> {
    const BERT_DIM: usize = 1024;

    if let Some(bert) = char_bert {
        let num_chars = bert.shape()[0];
        let mut phone_bert_data = Vec::with_capacity(num_phones * BERT_DIM);

        for (i, &repeat_count) in word2ph.iter().enumerate() {
            if i < num_chars {
                let char_row = bert.row(i);
                for _ in 0..repeat_count {
                    phone_bert_data.extend_from_slice(char_row.as_slice().unwrap());
                }
            } else {
                // Pad with zeros if word2ph length exceeds bert char count
                phone_bert_data.resize(phone_bert_data.len() + repeat_count * BERT_DIM, 0.0);
            }
        }

        // Pad or truncate to exact num_phones
        phone_bert_data.resize(num_phones * BERT_DIM, 0.0);

        Array2::from_shape_vec((num_phones, BERT_DIM), phone_bert_data)
            .unwrap_or_else(|_| Array2::zeros((num_phones, BERT_DIM)))
    } else {
        // Return zeros matrix if no BERT embedding
        Array2::zeros((num_phones, BERT_DIM))
    }
}
