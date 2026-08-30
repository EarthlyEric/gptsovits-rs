use crate::audio::AudioFormat;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StreamFormat {
    #[default]
    Audio,
    Sse,
}

/// OpenAI TTS Speech Request Schema (OpenAPI 3.1.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRequest {
    /// TTS model identifier (e.g. gpt-4o-mini-tts, gpt-sovits-v2, etc.)
    pub model: String,

    /// Text to synthesize (1..4096 characters)
    pub input: String,

    /// Voice identifier (e.g. "alloy") or custom voice configuration object
    pub voice: serde_json::Value,

    /// Optional instructions controlling speaking style, tone, or emotion
    #[serde(default)]
    pub instructions: Option<String>,

    /// Audio output format (mp3, opus, aac, flac, wav, pcm)
    #[serde(default)]
    pub response_format: Option<AudioFormat>,

    /// Speed of the synthesized speech (0.25 to 4.0, default 1.0)
    #[serde(default)]
    pub speed: Option<f32>,

    /// Streaming format (audio or sse, default audio)
    #[serde(default)]
    pub stream_format: Option<StreamFormat>,
}

/// OpenAI Standard Error Response Schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub param: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceObject {
    pub id: String,
    pub name: String,
    pub model_version: String,
    pub prompt_lang: String,
    pub text_lang: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceListResponse {
    pub object: String,
    pub data: Vec<VoiceObject>,
}
