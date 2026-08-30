use crate::audio::AudioFormat;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Streaming delivery format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum StreamFormat {
    #[default]
    Audio,
    Sse,
}

/// Dynamic custom voice configuration for zero-shot cloning
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DynamicVoiceObject {
    /// Path to the 3~10s reference audio WAV file
    #[schema(example = "voices/sandrone/ref.wav")]
    pub ref_audio_path: String,

    /// Transcript / text of the reference audio for prompt conditioning
    #[schema(example = "我是「木偶」桑多涅。不必多言，做好你分内的事情即可。")]
    pub prompt_text: String,

    /// Language code of prompt_text (zh, en, ja, ko, yue, auto)
    #[schema(example = "zh")]
    pub prompt_lang: String,

    /// Language code of target input text (zh, en, ja, ko, yue, auto)
    #[schema(example = "zh")]
    pub text_lang: String,

    /// Model version architecture (v1, v2, v2Pro, v2ProPlus, v3, v4)
    #[serde(default = "default_model_version")]
    #[schema(default = "v2ProPlus", example = "v2ProPlus")]
    pub model_version: String,

    /// Text segmentation strategy
    #[serde(default = "default_text_split_method")]
    #[schema(default = "cut5", example = "cut5")]
    pub text_split_method: String,

    /// Silence pause between segmented sentences in seconds
    #[serde(default = "default_fragment_interval")]
    #[schema(default = 0.2, example = 0.2)]
    pub fragment_interval: f32,

    /// Top-K sampling parameter
    #[serde(default = "default_top_k")]
    #[schema(default = 15, example = 15)]
    pub top_k: usize,

    /// Top-P nucleus sampling parameter
    #[serde(default = "default_top_p")]
    #[schema(default = 1.0, example = 1.0)]
    pub top_p: f32,

    /// Sampling temperature (higher = more expressive/random, lower = more stable)
    #[serde(default = "default_temperature")]
    #[schema(default = 1.0, example = 1.0)]
    pub temperature: f32,

    /// Repetition penalty for autoregressive semantic token generation
    #[serde(default = "default_repetition_penalty")]
    #[schema(default = 1.35, example = 1.35)]
    pub repetition_penalty: f32,
}

fn default_model_version() -> String {
    "v2ProPlus".to_string()
}
fn default_text_split_method() -> String {
    "cut5".to_string()
}
fn default_fragment_interval() -> f32 {
    0.2
}
fn default_top_k() -> usize {
    15
}
fn default_top_p() -> f32 {
    1.0
}
fn default_temperature() -> f32 {
    1.0
}
fn default_repetition_penalty() -> f32 {
    1.35
}

/// OpenAI TTS Speech Request Schema (OpenAPI 3.1.0 compatible)
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SpeechRequest {
    /// TTS model identifier (e.g. gpt-sovits-v2, gpt-sovits-v4, or custom model name like sandrone)
    #[schema(example = "gpt-sovits-v2")]
    pub model: String,

    /// Text to synthesize (1..4096 characters)
    #[schema(example = "先帝創業未半而中道崩殂，今天下三分，益州疲弊。")]
    pub input: String,

    /// Voice identifier (e.g. "default", "sandrone") or dynamic custom voice configuration object
    #[schema(value_type = Object, example = json!("default"))]
    pub voice: serde_json::Value,

    /// Optional instructions controlling speaking style, tone, or emotion
    #[serde(default)]
    #[schema(example = "Speak in a calm and dignified tone.")]
    pub instructions: Option<String>,

    /// Audio output format (mp3, opus, aac, flac, wav, pcm)
    #[serde(default)]
    #[schema(default = "mp3")]
    pub response_format: Option<AudioFormat>,

    /// Speed of the synthesized speech (0.25 to 4.0, default 1.0)
    #[serde(default)]
    #[schema(default = 1.0, minimum = 0.25, maximum = 4.0, example = 1.0)]
    pub speed: Option<f32>,

    /// Streaming format (audio or sse, default audio)
    #[serde(default)]
    #[schema(default = "audio")]
    pub stream_format: Option<StreamFormat>,
}

/// OpenAI Standard Error Response Schema
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorDetail {
    #[schema(example = "Input text cannot be empty.")]
    pub message: String,
    #[serde(rename = "type")]
    #[schema(example = "invalid_request_error")]
    pub type_: String,
    #[schema(example = "input")]
    pub param: Option<String>,
    #[schema(example = "empty_input")]
    pub code: Option<String>,
}

impl ErrorResponse {
    pub fn new(
        message: impl Into<String>,
        type_: impl Into<String>,
        param: Option<&str>,
        code: Option<&str>,
    ) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                type_: type_.into(),
                param: param.map(|s| s.to_string()),
                code: code.map(|s| s.to_string()),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelObject {
    #[schema(example = "gpt-sovits-v2")]
    pub id: String,
    #[schema(example = "model")]
    pub object: String,
    #[schema(example = 1700000000)]
    pub created: u64,
    #[schema(example = "official-base")]
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelListResponse {
    #[schema(example = "list")]
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct VoiceObject {
    #[schema(example = "default")]
    pub id: String,
    #[schema(example = "Default Voice Preset")]
    pub name: String,
    #[schema(example = "v2")]
    pub model_version: String,
    #[schema(example = "zh")]
    pub prompt_lang: String,
    #[schema(example = "zh")]
    pub text_lang: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct VoiceListResponse {
    #[schema(example = "list")]
    pub object: String,
    pub data: Vec<VoiceObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    #[schema(example = "ok")]
    pub status: String,
    #[schema(example = "gptsovits-rs")]
    pub engine: String,
    #[schema(example = "0.1.0")]
    pub version: String,
    #[schema(example = 8)]
    pub available_permits: usize,
}
