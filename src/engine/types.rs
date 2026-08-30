use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelVersion {
    V1,
    #[default]
    V2,
    #[serde(rename = "v2Pro", alias = "v2pro")]
    V2Pro,
    #[serde(rename = "v2ProPlus", alias = "v2proplus")]
    V2ProPlus,
    V3,
    V4,
}

impl ModelVersion {
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "v1" => ModelVersion::V1,
            "v2pro" | "v2_pro" => ModelVersion::V2Pro,
            "v2proplus" | "v2_pro_plus" | "v2pro+" => ModelVersion::V2ProPlus,
            "v3" => ModelVersion::V3,
            "v4" => ModelVersion::V4,
            _ => ModelVersion::V2,
        }
    }

    pub fn sampling_rate(&self) -> u32 {
        match self {
            ModelVersion::V1 | ModelVersion::V2 | ModelVersion::V2Pro | ModelVersion::V2ProPlus => 32000,
            ModelVersion::V3 => 24000,
            ModelVersion::V4 => 48000,
        }
    }

    pub fn symbols_version(&self) -> &'static str {
        match self {
            ModelVersion::V1 => "v1",
            _ => "v2",
        }
    }

    pub fn default_sample_steps(&self) -> usize {
        match self {
            ModelVersion::V3 => 32,
            ModelVersion::V4 => 16,
            _ => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub text: String,
    pub text_lang: String,
    pub prompt_text: String,
    pub prompt_lang: String,
    pub ref_audio: Vec<f32>,
    pub ref_sr: u32,
    pub text_split_method: String,
    pub fragment_interval: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub speed: f32,
    pub sample_steps: usize,
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            text_lang: "zh".to_string(),
            prompt_text: String::new(),
            prompt_lang: "zh".to_string(),
            ref_audio: Vec::new(),
            ref_sr: 32000,
            text_split_method: "cut5".to_string(),
            fragment_interval: 0.2,
            top_k: 15,
            top_p: 1.0,
            temperature: 1.0,
            repetition_penalty: 1.35,
            speed: 1.0,
            sample_steps: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}
