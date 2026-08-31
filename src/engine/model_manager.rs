use anyhow::{anyhow, Result};
use ndarray::Array2;
use std::collections::HashMap;

use crate::audio::{adjust_speed, resample};
use crate::config::AppConfig;
use crate::engine::cfm_v3_v4::CfmV3V4Model;
use crate::engine::cnhubert::CNHuBERTModel;
use crate::engine::roberta::RoBERTaModel;
use crate::engine::speaker::SpeakerEmbeddingModel;
use crate::engine::t2s::T2SModel;
use crate::engine::types::{InferenceRequest, InferenceResult, ModelVersion};
use crate::engine::vits_v1_v2::VitsV1V2Model;
use crate::text::{
    align_bert_to_phones, cleaned_text_to_sequence, segment_text, text_to_phonemes, BertTokenizer,
};

pub struct CustomModelInstance {
    pub version: ModelVersion,
    pub t2s: T2SModel,
    pub vits: Option<VitsV1V2Model>,
    pub cfm: Option<CfmV3V4Model>,
    pub sampling_rate: u32,
    pub sample_steps: usize,
}

pub struct ModelManager {
    cnhubert: CNHuBERTModel,
    roberta: RoBERTaModel,
    speaker: SpeakerEmbeddingModel,
    tokenizer: Option<BertTokenizer>,
    t2s_models: HashMap<ModelVersion, T2SModel>,
    vits_models: HashMap<ModelVersion, VitsV1V2Model>,
    cfm_models: HashMap<ModelVersion, CfmV3V4Model>,
    custom_models: HashMap<String, CustomModelInstance>,
    default_version: ModelVersion,
}

impl ModelManager {
    pub fn new(config: &AppConfig) -> Self {
        let default_version = ModelVersion::from_str_loose(&config.models.default_version);

        let cnhubert = CNHuBERTModel::new(&config.models.cnhubert_path);
        let roberta = RoBERTaModel::new(&config.models.bert_path);
        let speaker = SpeakerEmbeddingModel::new(&config.models.speaker_path);
        let tokenizer = match BertTokenizer::from_file(&config.models.bert_tokenizer_path) {
            Ok(tokenizer) => Some(tokenizer),
            Err(error) => {
                tracing::warn!(
                    path = %config.models.bert_tokenizer_path,
                    %error,
                    "Failed to load BERT tokenizer"
                );
                None
            }
        };

        let mut t2s_models = HashMap::new();
        let mut vits_models = HashMap::new();
        let mut cfm_models = HashMap::new();
        let mut custom_models = HashMap::new();

        // 1. Base Models (V1)
        t2s_models.insert(
            ModelVersion::V1,
            T2SModel::new(
                &config.models.v1.t2s_encoder_path,
                &config.models.v1.t2s_fsdec_path,
                &config.models.v1.t2s_sdec_path,
            ),
        );
        vits_models.insert(
            ModelVersion::V1,
            VitsV1V2Model::new(&config.models.v1.vits_path, config.models.v1.sampling_rate),
        );

        // 2. Base Models (V2)
        t2s_models.insert(
            ModelVersion::V2,
            T2SModel::new(
                &config.models.v2.t2s_encoder_path,
                &config.models.v2.t2s_fsdec_path,
                &config.models.v2.t2s_sdec_path,
            ),
        );
        vits_models.insert(
            ModelVersion::V2,
            VitsV1V2Model::new(&config.models.v2.vits_path, config.models.v2.sampling_rate),
        );

        // 3. Base Models (V2Pro)
        t2s_models.insert(
            ModelVersion::V2Pro,
            T2SModel::new(
                &config.models.v2_pro.t2s_encoder_path,
                &config.models.v2_pro.t2s_fsdec_path,
                &config.models.v2_pro.t2s_sdec_path,
            ),
        );
        vits_models.insert(
            ModelVersion::V2Pro,
            VitsV1V2Model::new(
                &config.models.v2_pro.vits_path,
                config.models.v2_pro.sampling_rate,
            ),
        );

        // 4. Base Models (V2ProPlus)
        t2s_models.insert(
            ModelVersion::V2ProPlus,
            T2SModel::new(
                &config.models.v2_pro_plus.t2s_encoder_path,
                &config.models.v2_pro_plus.t2s_fsdec_path,
                &config.models.v2_pro_plus.t2s_sdec_path,
            ),
        );
        vits_models.insert(
            ModelVersion::V2ProPlus,
            VitsV1V2Model::new(
                &config.models.v2_pro_plus.vits_path,
                config.models.v2_pro_plus.sampling_rate,
            ),
        );

        // 5. Base Models (V3)
        t2s_models.insert(
            ModelVersion::V3,
            T2SModel::new(
                &config.models.v3.t2s_encoder_path,
                &config.models.v3.t2s_fsdec_path,
                &config.models.v3.t2s_sdec_path,
            ),
        );
        cfm_models.insert(
            ModelVersion::V3,
            CfmV3V4Model::new(
                &config.models.v3.dit_path,
                &config.models.v3.vocoder_path,
                config.models.v3.sampling_rate,
                config.models.v3.sample_steps,
            ),
        );

        // 6. Base Models (V4)
        t2s_models.insert(
            ModelVersion::V4,
            T2SModel::new(
                &config.models.v4.t2s_encoder_path,
                &config.models.v4.t2s_fsdec_path,
                &config.models.v4.t2s_sdec_path,
            ),
        );
        cfm_models.insert(
            ModelVersion::V4,
            CfmV3V4Model::new(
                &config.models.v4.dit_path,
                &config.models.v4.vocoder_path,
                config.models.v4.sampling_rate,
                config.models.v4.sample_steps,
            ),
        );

        // 7. Custom Fine-tuned Models
        for (name, custom_cfg) in &config.models.custom {
            let ver = ModelVersion::from_str_loose(&custom_cfg.model_version);
            let sr = custom_cfg
                .sampling_rate
                .unwrap_or_else(|| ver.sampling_rate());
            let steps = custom_cfg
                .sample_steps
                .unwrap_or_else(|| ver.default_sample_steps());

            let dir = &custom_cfg.model_dir;
            let enc_path = if !custom_cfg.t2s_encoder_path.is_empty() {
                custom_cfg.t2s_encoder_path.clone()
            } else if !dir.is_empty() {
                format!("{}/t2s_encoder.onnx", dir)
            } else {
                String::new()
            };

            let fsdec_path = if !custom_cfg.t2s_fsdec_path.is_empty() {
                custom_cfg.t2s_fsdec_path.clone()
            } else if !dir.is_empty() {
                format!("{}/t2s_fsdec.onnx", dir)
            } else {
                String::new()
            };

            let sdec_path = if !custom_cfg.t2s_sdec_path.is_empty() {
                custom_cfg.t2s_sdec_path.clone()
            } else if !dir.is_empty() {
                format!("{}/t2s_sdec.onnx", dir)
            } else {
                String::new()
            };

            let vits_path = if !custom_cfg.vits_path.is_empty() {
                custom_cfg.vits_path.clone()
            } else if !dir.is_empty() {
                format!("{}/vits.onnx", dir)
            } else {
                String::new()
            };

            let dit_path = if !custom_cfg.dit_path.is_empty() {
                custom_cfg.dit_path.clone()
            } else if !dir.is_empty() {
                format!("{}/dit.onnx", dir)
            } else {
                String::new()
            };

            let vocoder_path = if !custom_cfg.vocoder_path.is_empty() {
                custom_cfg.vocoder_path.clone()
            } else if !dir.is_empty() {
                format!("{}/vocoder.onnx", dir)
            } else {
                String::new()
            };

            let t2s = T2SModel::new(&enc_path, &fsdec_path, &sdec_path);

            let (vits, cfm) = match ver {
                ModelVersion::V1
                | ModelVersion::V2
                | ModelVersion::V2Pro
                | ModelVersion::V2ProPlus => (Some(VitsV1V2Model::new(&vits_path, sr)), None),
                ModelVersion::V3 | ModelVersion::V4 => (
                    None,
                    Some(CfmV3V4Model::new(&dit_path, &vocoder_path, sr, steps)),
                ),
            };

            custom_models.insert(
                name.to_lowercase(),
                CustomModelInstance {
                    version: ver,
                    t2s,
                    vits,
                    cfm,
                    sampling_rate: sr,
                    sample_steps: steps,
                },
            );
        }

        Self {
            cnhubert,
            roberta,
            speaker,
            tokenizer,
            t2s_models,
            vits_models,
            cfm_models,
            custom_models,
            default_version,
        }
    }

    pub fn default_version(&self) -> ModelVersion {
        self.default_version
    }

    pub fn has_custom_model(&self, name: &str) -> bool {
        self.custom_models.contains_key(&name.to_lowercase())
    }

    pub fn has_model(&self, model_name: &str) -> bool {
        let name = model_name.trim().to_ascii_lowercase();
        self.custom_models.contains_key(&name) || self.base_version_for_name(&name).is_some()
    }

    pub fn list_custom_models(&self) -> Vec<(String, ModelVersion)> {
        self.custom_models
            .iter()
            .map(|(k, v)| (k.clone(), v.version))
            .collect()
    }

    fn base_version_for_name(&self, model_name: &str) -> Option<ModelVersion> {
        match model_name.trim().to_ascii_lowercase().as_str() {
            "gpt-sovits-v1" | "v1" => Some(ModelVersion::V1),
            "gpt-sovits-v2" | "v2" => Some(ModelVersion::V2),
            "gpt-sovits-v2pro" | "v2pro" => Some(ModelVersion::V2Pro),
            "gpt-sovits-v2proplus" | "v2proplus" | "v2pro+" => Some(ModelVersion::V2ProPlus),
            "gpt-sovits-v3" | "v3" => Some(ModelVersion::V3),
            "gpt-sovits-v4" | "v4" => Some(ModelVersion::V4),
            "gpt-4o-mini-tts" | "tts-1" | "tts-1-hd" | "gpt-sovits" => Some(self.default_version),
            _ => None,
        }
    }

    fn normalized_language(language: &str) -> String {
        language
            .trim()
            .to_ascii_lowercase()
            .trim_start_matches("all_")
            .to_string()
    }

    fn validate_language(language: &str) -> Result<()> {
        let normalized = Self::normalized_language(language);
        if matches!(normalized.as_str(), "zh" | "en") {
            Ok(())
        } else {
            Err(anyhow!(
                "Language '{}' is not supported by the Rust frontend; supported languages are zh and en",
                language
            ))
        }
    }

    fn extract_bert_features(
        &self,
        norm_text: &str,
        word2ph: &[usize],
        num_phones: usize,
        language: &str,
    ) -> Result<Array2<f32>> {
        let normalized_language = Self::normalized_language(language);
        if normalized_language == "en" {
            return Ok(Array2::zeros((num_phones, 1024)));
        }

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("BERT tokenizer is required for Chinese text"))?;
        // The upstream frontend splits mixed-language input before BERT. Keep
        // English phone spans at zero instead of aligning WordPiece tokens to
        // individual English characters.
        let is_chinese_bert_char = |ch: char| {
            ('\u{4e00}'..='\u{9fa5}').contains(&ch)
                || matches!(ch, '!' | '?' | '…' | ',' | '.' | '-')
        };
        let mut bert_text = String::new();
        let mut bert_word2ph = Vec::new();
        for (ch, &repeat_count) in norm_text.chars().zip(word2ph.iter()) {
            if is_chinese_bert_char(ch) {
                bert_text.push(ch);
                bert_word2ph.push(repeat_count);
            }
        }

        if bert_text.is_empty() {
            return Ok(Array2::zeros((num_phones, 1024)));
        }

        let (ids, mask, types) = tokenizer.encode(&bert_text)?;
        let char_bert = self.roberta.extract(&ids, &mask, &types)?;
        if char_bert.shape().get(1).copied() != Some(1024) {
            return Err(anyhow!(
                "BERT returned hidden size {:?}; expected 1024",
                char_bert.shape()
            ));
        }

        let bert_num_phones = bert_word2ph.iter().sum();
        let bert_phone_features =
            align_bert_to_phones(Some(&char_bert), &bert_word2ph, bert_num_phones);
        let mut phone_features = Array2::zeros((num_phones, 1024));
        let mut phone_offset = 0;
        let mut bert_phone_offset = 0;
        for (ch, &repeat_count) in norm_text.chars().zip(word2ph.iter()) {
            if repeat_count == 0 {
                continue;
            }

            if is_chinese_bert_char(ch) {
                for row in 0..repeat_count {
                    if phone_offset + row < num_phones && bert_phone_offset + row < bert_num_phones
                    {
                        let source = bert_phone_features.row(bert_phone_offset + row);
                        phone_features.row_mut(phone_offset + row).assign(&source);
                    }
                }
                bert_phone_offset += repeat_count;
            }
            phone_offset += repeat_count;
        }

        Ok(phone_features)
    }

    fn extract_speaker_embedding(
        &self,
        vits: &VitsV1V2Model,
        audio_16k: &[f32],
    ) -> Result<Array2<f32>> {
        if !vits.has_speaker_embedding_input() {
            return Err(anyhow!(
                "V2Pro/V2ProPlus VITS graph does not expose 'sv_emb'; re-export or patch the VITS ONNX model"
            ));
        }
        self.speaker.extract(audio_16k)
    }

    fn normalize_reference_audio(audio: &mut [f32]) {
        let max_abs = audio.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        if max_abs > 1.0 {
            let scale = 2.0_f32.min(max_abs);
            for sample in audio {
                *sample /= scale;
            }
        }
    }

    /// Full end-to-end TTS Synthesis pipeline by Model Identifier (custom model name or base model)
    pub fn synthesize_by_model(
        &self,
        req: &InferenceRequest,
        model_name: &str,
    ) -> Result<InferenceResult> {
        let name_lower = model_name.trim().to_ascii_lowercase();

        Self::validate_language(&req.text_lang)?;
        Self::validate_language(&req.prompt_lang)?;

        // 1. Check if it's a registered custom fine-tuned model
        if let Some(custom_model) = self.custom_models.get(&name_lower) {
            return self.synthesize_with_custom(req, custom_model);
        }

        // 2. Otherwise resolve base model version
        let base_version = self.base_version_for_name(&name_lower).ok_or_else(|| {
            anyhow!(
                "Model '{}' not found in base models or custom models",
                model_name
            )
        })?;

        self.synthesize(req, base_version)
    }

    /// Synthesize with a custom fine-tuned model instance (with text segmentation)
    fn synthesize_with_custom(
        &self,
        req: &InferenceRequest,
        custom: &CustomModelInstance,
    ) -> Result<InferenceResult> {
        let sym_version = custom.version.symbols_version();
        let target_sr = custom.sampling_rate;

        if req.ref_sr == 0 || req.ref_audio.is_empty() {
            return Err(anyhow!("A non-empty reference audio is required"));
        }
        let ref_duration = req.ref_audio.len() as f32 / req.ref_sr as f32;
        if !(3.0..=10.0).contains(&ref_duration) {
            return Err(anyhow!(
                "Reference audio must be between 3 and 10 seconds (got {:.3}s)",
                ref_duration
            ));
        }

        // 1. Prompt Text G2P & BERT
        let prompt_raw = req.prompt_text.trim();
        let prompt_text = if prompt_raw.is_empty() {
            if Self::normalized_language(&req.prompt_lang) == "en" {
                "Hello.".to_string()
            } else {
                "你好。".to_string()
            }
        } else if !prompt_raw.ends_with(['。', '.', '！', '!', '？', '?']) {
            let terminator = if Self::normalized_language(&req.prompt_lang) == "en" {
                '.'
            } else {
                '。'
            };
            format!("{}{}", prompt_raw, terminator)
        } else {
            prompt_raw.to_string()
        };

        let (ref_phones, ref_word2ph, ref_norm) =
            text_to_phonemes(&prompt_text, &req.prompt_lang, sym_version);
        let ref_seq = cleaned_text_to_sequence(&ref_phones, sym_version);

        let ref_bert = self.extract_bert_features(
            &ref_norm,
            &ref_word2ph,
            ref_phones.len(),
            &req.prompt_lang,
        )?;

        // 2. Reference Audio Preprocessing & SSL Extraction
        let ref_audio = req.ref_audio.clone();

        let mut ref_audio_16k = resample(&ref_audio, req.ref_sr as usize, 16000)?;
        let mut ref_audio_tgt_sr = resample(&ref_audio, req.ref_sr as usize, target_sr as usize)?;
        Self::normalize_reference_audio(&mut ref_audio_tgt_sr);
        Self::normalize_reference_audio(&mut ref_audio_16k);
        let sv_emb = if custom.version.uses_speaker_embedding() {
            let speaker_audio_16k = resample(&ref_audio_tgt_sr, target_sr as usize, 16000)?;
            let vits = custom
                .vits
                .as_ref()
                .ok_or_else(|| anyhow!("V2Pro/V2ProPlus custom model has no VITS synthesizer"))?;
            Some(self.extract_speaker_embedding(vits, &speaker_audio_16k)?)
        } else {
            None
        };
        // 0.3s zero padding at the end of prompt audio for boundary protection
        ref_audio_16k.resize(ref_audio_16k.len() + (16000.0 * 0.3) as usize, 0.0);
        let ssl_content = self.cnhubert.extract(&ref_audio_16k)?;

        // 3. Text Segmentation
        let segments = segment_text(&req.text, &req.text_split_method);
        let mut all_samples = Vec::new();

        let pause_samples = if req.fragment_interval > 0.0 {
            vec![0.0f32; (target_sr as f32 * req.fragment_interval) as usize]
        } else {
            Vec::new()
        };

        for (seg_idx, segment) in segments.iter().enumerate() {
            if seg_idx > 0 && !pause_samples.is_empty() {
                all_samples.extend_from_slice(&pause_samples);
            }

            let seg_raw = segment.trim();
            if seg_raw.is_empty() {
                continue;
            }
            let seg_clean = if !seg_raw.ends_with(['。', '.', '！', '!', '？', '?', '，', ','])
            {
                let terminator = if Self::normalized_language(&req.text_lang) == "en" {
                    '.'
                } else {
                    '。'
                };
                format!("{}{}", seg_raw, terminator)
            } else {
                seg_raw.to_string()
            };

            let (text_phones, text_word2ph, text_norm) =
                text_to_phonemes(&seg_clean, &req.text_lang, sym_version);
            let text_seq = cleaned_text_to_sequence(&text_phones, sym_version);

            let text_bert = self.extract_bert_features(
                &text_norm,
                &text_word2ph,
                text_phones.len(),
                &req.text_lang,
            )?;

            let pred_semantic = custom.t2s.generate(
                &ref_seq,
                &text_seq,
                &ref_bert,
                &text_bert,
                &ssl_content,
                req.top_k,
                req.top_p,
                req.temperature,
                req.repetition_penalty,
            )?;

            let seg_samples = if let Some(vits) = &custom.vits {
                vits.synthesize(
                    &text_seq,
                    &pred_semantic,
                    &ref_audio_tgt_sr,
                    sv_emb.as_ref(),
                )?
            } else if let Some(cfm) = &custom.cfm {
                cfm.synthesize(
                    &text_seq,
                    &pred_semantic,
                    &ref_audio_tgt_sr,
                    req.sample_steps,
                )?
            } else {
                return Err(anyhow!("Custom model has neither VITS nor CFM synthesizer"));
            };

            all_samples.extend(seg_samples);
        }

        // 4. Speed Adjustment on combined waveform
        let processed_samples = if (req.speed - 1.0).abs() > 1e-4 {
            adjust_speed(&all_samples, req.speed, target_sr)?
        } else {
            all_samples
        };

        if processed_samples.is_empty() {
            return Err(anyhow!("Synthesis produced no audio samples"));
        }

        Ok(InferenceResult {
            samples: processed_samples,
            sample_rate: target_sr,
        })
    }

    /// Base Model TTS Synthesis pipeline in pure Rust (with text segmentation)
    pub fn synthesize(
        &self,
        req: &InferenceRequest,
        version: ModelVersion,
    ) -> Result<InferenceResult> {
        let sym_version = version.symbols_version();
        let target_sr = version.sampling_rate();

        if req.ref_sr == 0 || req.ref_audio.is_empty() {
            return Err(anyhow!("A non-empty reference audio is required"));
        }
        let ref_duration = req.ref_audio.len() as f32 / req.ref_sr as f32;
        if !(3.0..=10.0).contains(&ref_duration) {
            return Err(anyhow!(
                "Reference audio must be between 3 and 10 seconds (got {:.3}s)",
                ref_duration
            ));
        }

        // 1. Reference / Prompt Text G2P & BERT
        let prompt_raw = req.prompt_text.trim();
        let prompt_text = if prompt_raw.is_empty() {
            if Self::normalized_language(&req.prompt_lang) == "en" {
                "Hello.".to_string()
            } else {
                "你好。".to_string()
            }
        } else if !prompt_raw.ends_with(['。', '.', '！', '!', '？', '?']) {
            let terminator = if Self::normalized_language(&req.prompt_lang) == "en" {
                '.'
            } else {
                '。'
            };
            format!("{}{}", prompt_raw, terminator)
        } else {
            prompt_raw.to_string()
        };

        let (ref_phones, ref_word2ph, ref_norm) =
            text_to_phonemes(&prompt_text, &req.prompt_lang, sym_version);
        let ref_seq = cleaned_text_to_sequence(&ref_phones, sym_version);

        let ref_bert = self.extract_bert_features(
            &ref_norm,
            &ref_word2ph,
            ref_phones.len(),
            &req.prompt_lang,
        )?;

        // 2. Reference Audio Preprocessing & SSL Extraction
        let ref_audio = req.ref_audio.clone();

        let mut ref_audio_16k = resample(&ref_audio, req.ref_sr as usize, 16000)?;
        let mut ref_audio_tgt_sr = resample(&ref_audio, req.ref_sr as usize, target_sr as usize)?;
        Self::normalize_reference_audio(&mut ref_audio_tgt_sr);
        Self::normalize_reference_audio(&mut ref_audio_16k);
        let sv_emb = if version.uses_speaker_embedding() {
            let speaker_audio_16k = resample(&ref_audio_tgt_sr, target_sr as usize, 16000)?;
            let vits = self
                .vits_models
                .get(&version)
                .ok_or_else(|| anyhow!("VITS model for {:?} not configured", version))?;
            Some(self.extract_speaker_embedding(vits, &speaker_audio_16k)?)
        } else {
            None
        };
        // 0.3s zero padding at the end of prompt audio for boundary protection
        ref_audio_16k.resize(ref_audio_16k.len() + (16000.0 * 0.3) as usize, 0.0);
        let ssl_content = self.cnhubert.extract(&ref_audio_16k)?;

        let t2s = self
            .t2s_models
            .get(&version)
            .ok_or_else(|| anyhow!("T2S model for {:?} not configured", version))?;

        // 3. Text Segmentation
        let segments = segment_text(&req.text, &req.text_split_method);
        let mut all_samples = Vec::new();

        let pause_samples = if req.fragment_interval > 0.0 {
            vec![0.0f32; (target_sr as f32 * req.fragment_interval) as usize]
        } else {
            Vec::new()
        };

        for (seg_idx, segment) in segments.iter().enumerate() {
            if seg_idx > 0 && !pause_samples.is_empty() {
                all_samples.extend_from_slice(&pause_samples);
            }

            let seg_raw = segment.trim();
            if seg_raw.is_empty() {
                continue;
            }
            let seg_clean = if !seg_raw.ends_with(['。', '.', '！', '!', '？', '?', '，', ','])
            {
                let terminator = if Self::normalized_language(&req.text_lang) == "en" {
                    '.'
                } else {
                    '。'
                };
                format!("{}{}", seg_raw, terminator)
            } else {
                seg_raw.to_string()
            };

            let (text_phones, text_word2ph, text_norm) =
                text_to_phonemes(&seg_clean, &req.text_lang, sym_version);
            let text_seq = cleaned_text_to_sequence(&text_phones, sym_version);

            let text_bert = self.extract_bert_features(
                &text_norm,
                &text_word2ph,
                text_phones.len(),
                &req.text_lang,
            )?;

            let pred_semantic = t2s.generate(
                &ref_seq,
                &text_seq,
                &ref_bert,
                &text_bert,
                &ssl_content,
                req.top_k,
                req.top_p,
                req.temperature,
                req.repetition_penalty,
            )?;

            let seg_samples = match version {
                ModelVersion::V1
                | ModelVersion::V2
                | ModelVersion::V2Pro
                | ModelVersion::V2ProPlus => {
                    let vits = self
                        .vits_models
                        .get(&version)
                        .ok_or_else(|| anyhow!("VITS model for {:?} not configured", version))?;
                    vits.synthesize(
                        &text_seq,
                        &pred_semantic,
                        &ref_audio_tgt_sr,
                        sv_emb.as_ref(),
                    )?
                }
                ModelVersion::V3 | ModelVersion::V4 => {
                    let cfm = self
                        .cfm_models
                        .get(&version)
                        .ok_or_else(|| anyhow!("CFM model for {:?} not configured", version))?;
                    cfm.synthesize(
                        &text_seq,
                        &pred_semantic,
                        &ref_audio_tgt_sr,
                        req.sample_steps,
                    )?
                }
            };

            all_samples.extend(seg_samples);
        }

        // 4. Post-processing: Speed Adjustment
        let processed_samples = if (req.speed - 1.0).abs() > 1e-4 {
            adjust_speed(&all_samples, req.speed, target_sr)?
        } else {
            all_samples
        };

        if processed_samples.is_empty() {
            return Err(anyhow!("Synthesis produced no audio samples"));
        }

        Ok(InferenceResult {
            samples: processed_samples,
            sample_rate: target_sr,
        })
    }
}
