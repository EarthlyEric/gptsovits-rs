use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_bind_address")]
    pub bind_address: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub api_key: String,
    #[serde(default = "default_concurrency")]
    pub max_concurrency: usize,
    #[serde(default = "default_voices_config")]
    pub voices_config: String,
}

fn default_bind_address() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    9880
}

fn default_concurrency() -> usize {
    8
}

fn default_voices_config() -> String {
    "voices.toml".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default = "default_cuda_device_id")]
    pub cuda_device_id: i32,
    #[serde(default)]
    pub cuda_lib_dir: String,
    #[serde(default)]
    pub cudnn_lib_dir: String,
    #[serde(default = "default_intra_threads")]
    pub intra_threads: usize,
    #[serde(default = "default_inter_threads")]
    pub inter_threads: usize,
}

fn default_device() -> String {
    "cuda".to_string()
}

fn default_cuda_device_id() -> i32 {
    0
}

fn default_intra_threads() -> usize {
    4
}

fn default_inter_threads() -> usize {
    2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V1V2ModelConfig {
    #[serde(default)]
    pub t2s_encoder_path: String,
    #[serde(default)]
    pub t2s_fsdec_path: String,
    #[serde(default)]
    pub t2s_sdec_path: String,
    #[serde(default)]
    pub vits_path: String,
    #[serde(default = "default_v1_v2_sr")]
    pub sampling_rate: u32,
}

fn default_v1_v2_sr() -> u32 {
    32000
}

impl Default for V1V2ModelConfig {
    fn default() -> Self {
        Self {
            t2s_encoder_path: String::new(),
            t2s_fsdec_path: String::new(),
            t2s_sdec_path: String::new(),
            vits_path: String::new(),
            sampling_rate: 32000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V3V4ModelConfig {
    #[serde(default)]
    pub t2s_encoder_path: String,
    #[serde(default)]
    pub t2s_fsdec_path: String,
    #[serde(default)]
    pub t2s_sdec_path: String,
    #[serde(default)]
    pub dit_path: String,
    #[serde(default)]
    pub vocoder_path: String,
    #[serde(default = "default_v3_sr")]
    pub sampling_rate: u32,
    #[serde(default = "default_sample_steps")]
    pub sample_steps: usize,
}

fn default_v3_sr() -> u32 {
    24000
}

fn default_sample_steps() -> usize {
    32
}

impl Default for V3V4ModelConfig {
    fn default() -> Self {
        Self {
            t2s_encoder_path: String::new(),
            t2s_fsdec_path: String::new(),
            t2s_sdec_path: String::new(),
            dit_path: String::new(),
            vocoder_path: String::new(),
            sampling_rate: 24000,
            sample_steps: 32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomModelConfig {
    #[serde(default = "default_model_version")]
    pub model_version: String,
    #[serde(default)]
    pub model_dir: String,
    #[serde(default)]
    pub t2s_encoder_path: String,
    #[serde(default)]
    pub t2s_fsdec_path: String,
    #[serde(default)]
    pub t2s_sdec_path: String,
    #[serde(default)]
    pub vits_path: String,
    #[serde(default)]
    pub dit_path: String,
    #[serde(default)]
    pub vocoder_path: String,
    #[serde(default)]
    pub sampling_rate: Option<u32>,
    #[serde(default)]
    pub sample_steps: Option<usize>,
}

impl Default for CustomModelConfig {
    fn default() -> Self {
        Self {
            model_version: "v2".to_string(),
            model_dir: String::new(),
            t2s_encoder_path: String::new(),
            t2s_fsdec_path: String::new(),
            t2s_sdec_path: String::new(),
            vits_path: String::new(),
            dit_path: String::new(),
            vocoder_path: String::new(),
            sampling_rate: None,
            sample_steps: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    #[serde(default = "default_model_version")]
    pub default_version: String,
    #[serde(default)]
    pub enabled_base_versions: Vec<String>,
    #[serde(default = "default_cnhubert_path")]
    pub cnhubert_path: String,
    #[serde(default = "default_bert_path")]
    pub bert_path: String,
    #[serde(default = "default_tokenizer_path")]
    pub bert_tokenizer_path: String,
    #[serde(default = "default_speaker_path")]
    pub speaker_path: String,

    #[serde(default)]
    pub v1: V1V2ModelConfig,
    #[serde(default)]
    pub v2: V1V2ModelConfig,
    #[serde(rename = "v2Pro", default)]
    pub v2_pro: V1V2ModelConfig,
    #[serde(rename = "v2ProPlus", default)]
    pub v2_pro_plus: V1V2ModelConfig,
    #[serde(default)]
    pub v3: V3V4ModelConfig,
    #[serde(default)]
    pub v4: V3V4ModelConfig,

    #[serde(default)]
    pub custom: HashMap<String, CustomModelConfig>,
}

fn default_model_version() -> String {
    "v2".to_string()
}

fn default_cnhubert_path() -> String {
    "models/chinese-hubert-base/cnhubert.onnx".to_string()
}

fn default_bert_path() -> String {
    "models/chinese-roberta-wwm-ext-large/bert.onnx".to_string()
}

fn default_tokenizer_path() -> String {
    "models/chinese-roberta-wwm-ext-large/tokenizer.json".to_string()
}

fn default_speaker_path() -> String {
    "models/sv.onnx".to_string()
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            default_version: default_model_version(),
            enabled_base_versions: Vec::new(),
            cnhubert_path: default_cnhubert_path(),
            bert_path: default_bert_path(),
            bert_tokenizer_path: default_tokenizer_path(),
            speaker_path: default_speaker_path(),
            v1: V1V2ModelConfig::default(),
            v2: V1V2ModelConfig::default(),
            v2_pro: V1V2ModelConfig::default(),
            v2_pro_plus: V1V2ModelConfig::default(),
            v3: V3V4ModelConfig::default(),
            v4: V3V4ModelConfig {
                sampling_rate: 48000,
                sample_steps: 16,
                ..Default::default()
            },
            custom: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    #[serde(default)]
    pub models: ModelsConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: default_bind_address(),
            port: default_port(),
            api_key: String::new(),
            max_concurrency: default_concurrency(),
            voices_config: default_voices_config(),
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: default_device(),
            cuda_device_id: default_cuda_device_id(),
            cuda_lib_dir: String::new(),
            cudnn_lib_dir: String::new(),
            intra_threads: default_intra_threads(),
            inter_threads: default_inter_threads(),
        }
    }
}

impl AppConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(path_ref)
            .map_err(|e| anyhow!("Failed to read config file {:?}: {}", path_ref, e))?;

        let config: AppConfig = toml::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse config file {:?}: {}", path_ref, e))?;

        Ok(config)
    }
}
