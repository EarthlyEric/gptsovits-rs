use std::path::Path;
use tokenizers::Tokenizer;
use anyhow::Result;

pub struct BertTokenizer {
    tokenizer: Tokenizer,
}

impl BertTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let tokenizer = Tokenizer::from_file(path_ref)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", path_ref, e))?;
        Ok(Self { tokenizer })
    }

    /// Tokenize text for RoBERTa BERT model
    /// Returns (input_ids, attention_mask, token_type_ids)
    pub fn encode(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer encoding error: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

        Ok((input_ids, attention_mask, token_type_ids))
    }
}
