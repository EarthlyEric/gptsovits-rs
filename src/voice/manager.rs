use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicePreset {
    pub ref_audio_path: String,
    #[serde(default)]
    pub prompt_text: String,
    #[serde(default = "default_lang")]
    pub prompt_lang: String,
    #[serde(default = "default_lang")]
    pub text_lang: String,
    #[serde(default = "default_model_version")]
    pub model_version: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
}

fn default_lang() -> String {
    "zh".to_string()
}

fn default_model_version() -> String {
    "v2".to_string()
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicesFile {
    pub voices: HashMap<String, VoicePreset>,
}

#[derive(Debug, Clone)]
pub struct VoiceManager {
    voices: HashMap<String, VoicePreset>,
}

impl VoiceManager {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            // Return default voice presets
            return Ok(Self::default());
        }

        let content = fs::read_to_string(path_ref)
            .map_err(|e| anyhow!("Failed to read voices config {:?}: {}", path_ref, e))?;

        let file_data: VoicesFile = toml::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse voices config {:?}: {}", path_ref, e))?;

        Ok(Self {
            voices: file_data.voices,
        })
    }

    pub fn get_voice(&self, name: &str) -> Option<&VoicePreset> {
        self.voices.get(name)
    }

    pub fn list_voices(&self) -> Vec<(String, VoicePreset)> {
        self.voices
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Resolve voice either from preset name or custom dynamic JSON object
    pub fn resolve_voice(&self, voice_val: &serde_json::Value) -> Result<VoicePreset> {
        match voice_val {
            serde_json::Value::String(name) => {
                if let Some(preset) = self.get_voice(name) {
                    Ok(preset.clone())
                } else {
                    // Try case-insensitive lookup
                    let name_lower = name.to_lowercase();
                    for (k, v) in &self.voices {
                        if k.to_lowercase() == name_lower {
                            return Ok(v.clone());
                        }
                    }
                    Err(anyhow!("Voice '{}' not found in voice presets.", name))
                }
            }
            serde_json::Value::Object(_) => {
                let preset: VoicePreset = serde_json::from_value(voice_val.clone())
                    .map_err(|e| anyhow!("Invalid custom voice object schema: {}", e))?;
                Ok(preset)
            }
            _ => Err(anyhow!("Invalid voice type: must be string or object.")),
        }
    }
}

impl Default for VoiceManager {
    fn default() -> Self {
        let mut voices = HashMap::new();
        let default_preset = VoicePreset {
            ref_audio_path: "voices/default/ref.wav".to_string(),
            prompt_text: "你好，我是語音合成助理。".to_string(),
            prompt_lang: "zh".to_string(),
            text_lang: "zh".to_string(),
            model_version: "v2".to_string(),
            top_k: 15,
            top_p: 1.0,
            temperature: 1.0,
            repetition_penalty: 1.35,
        };
        voices.insert("default".to_string(), default_preset.clone());
        voices.insert("alloy".to_string(), default_preset.clone());
        voices.insert("echo".to_string(), default_preset.clone());
        voices.insert("fable".to_string(), default_preset.clone());
        voices.insert("onyx".to_string(), default_preset.clone());
        voices.insert("nova".to_string(), default_preset.clone());
        voices.insert("shimmer".to_string(), default_preset.clone());
        voices.insert("sandrone".to_string(), default_preset);

        Self { voices }
    }
}
