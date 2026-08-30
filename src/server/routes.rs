use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use futures_util::stream;
use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::audio::{encode_audio, load_wav, AudioFormat};
use crate::engine::{InferenceRequest, ModelManager};
use crate::server::error::AppError;
use crate::server::schema::{
    ErrorResponse, HealthResponse, ModelListResponse, ModelObject, SpeechRequest, StreamFormat,
    VoiceListResponse, VoiceObject,
};
use crate::voice::VoiceManager;

#[derive(Clone)]
pub struct AppState {
    pub model_manager: Arc<ModelManager>,
    pub voice_manager: Arc<VoiceManager>,
    pub semaphore: Arc<Semaphore>,
}

/// Handler for POST /audio/speech and POST /v1/audio/speech
#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    tag = "Audio",
    summary = "Create speech from input text",
    description = "Synthesizes speech audio from the given text using OpenAI-compatible parameters, with GPT-SoVITS zero-shot voice cloning and text segmentation support.",
    request_body(
        content = SpeechRequest,
        description = "TTS parameters including model ID, text input, voice preset or dynamic voice config, speed, and output audio format",
        content_type = "application/json"
    ),
    security(
        ("bearerAuth" = [])
    ),
    responses(
        (status = 200, description = "Synthesized audio file (binary stream)", content_type = "audio/mpeg"),
        (status = 400, description = "Bad Request (empty input, speed out of bounds, etc.)", body = ErrorResponse),
        (status = 401, description = "Unauthorized (invalid or missing API key)", body = ErrorResponse),
        (status = 404, description = "Not Found (model or voice not found)", body = ErrorResponse),
        (status = 500, description = "Internal Server Error during inference", body = ErrorResponse)
    )
)]
pub async fn create_speech(
    State(state): State<AppState>,
    Json(payload): Json<SpeechRequest>,
) -> Result<Response, AppError> {
    // 1. Validate Input Length (1..4096)
    let input_len = payload.input.chars().count();
    if input_len == 0 {
        return Err(AppError::BadRequest(
            "Input text cannot be empty.".to_string(),
            Some("input"),
            Some("empty_input"),
        ));
    }
    if input_len > 4096 {
        return Err(AppError::BadRequest(
            "Input text exceeds maximum length of 4096 characters.".to_string(),
            Some("input"),
            Some("input_too_long"),
        ));
    }

    // 2. Validate Speed (0.25..4.0)
    let speed = payload.speed.unwrap_or(1.0);
    if !(0.25..=4.0).contains(&speed) {
        return Err(AppError::BadRequest(
            "Speed must be between 0.25 and 4.0.".to_string(),
            Some("speed"),
            Some("invalid_speed"),
        ));
    }

    let response_format = payload.response_format.unwrap_or(AudioFormat::Mp3);
    let stream_format = payload.stream_format.unwrap_or(StreamFormat::Audio);

    // 3. Resolve Voice Preset / Dynamic Voice Object
    let voice_preset = state
        .voice_manager
        .resolve_voice(&payload.voice)
        .map_err(|e| {
            AppError::NotFound(
                format!("Invalid voice: {}", e),
                Some("voice"),
                Some("invalid_voice"),
            )
        })?;

    // 4. Load Reference Audio if specified
    let (ref_audio, ref_sr) = if !voice_preset.ref_audio_path.is_empty() {
        if let Ok((samples, sr)) = load_wav(&voice_preset.ref_audio_path) {
            (samples, sr)
        } else {
            (Vec::new(), 32000)
        }
    } else {
        (Vec::new(), 32000)
    };

    // 5. Build Inference Request
    let infer_req = InferenceRequest {
        text: payload.input,
        text_lang: voice_preset.text_lang.clone(),
        prompt_text: voice_preset.prompt_text.clone(),
        prompt_lang: voice_preset.prompt_lang.clone(),
        ref_audio,
        ref_sr,
        text_split_method: voice_preset.text_split_method.clone(),
        fragment_interval: voice_preset.fragment_interval,
        top_k: voice_preset.top_k,
        top_p: voice_preset.top_p,
        temperature: voice_preset.temperature,
        repetition_penalty: voice_preset.repetition_penalty,
        speed,
        sample_steps: 32,
    };

    // 6. Acquire concurrency permit
    let _permit = state
        .semaphore
        .try_acquire()
        .map_err(|_| AppError::RateLimit("Server concurrency limit reached. Please retry shortly.".to_string()))?;

    // 7. Run Inference with selected model (custom fine-tuned model or base model)
    let model_mgr = state.model_manager.clone();
    let model_name = payload.model.clone();

    let infer_result = tokio::task::spawn_blocking(move || {
        model_mgr.synthesize_by_model(&infer_req, &model_name)
    })
    .await
    .map_err(|e| AppError::Internal(format!("Task execution failed: {}", e)))?
    .map_err(|e| {
        let err_str = e.to_string();
        if err_str.contains("not found") {
            AppError::NotFound(err_str, Some("model"), Some("model_not_found"))
        } else {
            AppError::Internal(format!("TTS synthesis error: {}", err_str))
        }
    })?;

    // 8. Audio Encoding
    let audio_bytes = encode_audio(
        &infer_result.samples,
        infer_result.sample_rate,
        response_format,
    )
    .map_err(|e| AppError::Internal(format!("Audio encoding failed: {}", e)))?;

    // 9. Return response based on stream_format
    match stream_format {
        StreamFormat::Audio => {
            let mut headers = HeaderMap::new();
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static(response_format.content_type()),
            );
            headers.insert(
                header::CONTENT_LENGTH,
                HeaderValue::from_str(&audio_bytes.len().to_string()).unwrap(),
            );
            headers.insert(
                header::CACHE_CONTROL,
                HeaderValue::from_static("no-cache, no-store, must-revalidate"),
            );

            Ok((StatusCode::OK, headers, audio_bytes).into_response())
        }
        StreamFormat::Sse => {
            // Format audio chunk as Server-Sent Events stream
            let mut headers = HeaderMap::new();
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream"),
            );
            headers.insert(
                header::CACHE_CONTROL,
                HeaderValue::from_static("no-cache"),
            );
            headers.insert(
                header::CONNECTION,
                HeaderValue::from_static("keep-alive"),
            );

            let chunk_size = 8192;
            let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = audio_bytes
                .chunks(chunk_size)
                .map(|c| {
                    let sse_event = format!("data: {}\n\n", hex::encode(c));
                    Ok(bytes::Bytes::from(sse_event))
                })
                .collect();

            let stream = stream::iter(chunks);
            Ok((StatusCode::OK, headers, Body::from_stream(stream)).into_response())
        }
    }
}

/// Helper hex encoding
mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

/// Handler for GET /models and GET /v1/models
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "Models",
    summary = "List available TTS models",
    description = "Lists official base models (e.g. gpt-sovits-v1..v4) and registered custom fine-tuned models.",
    security(
        ("bearerAuth" = [])
    ),
    responses(
        (status = 200, description = "List of models", body = ModelListResponse)
    )
)]
pub async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut data = Vec::new();

    // 1. Official Base Models
    let base_models = vec![
        "gpt-4o-mini-tts",
        "tts-1",
        "tts-1-hd",
        "gpt-sovits",
        "gpt-sovits-v1",
        "gpt-sovits-v2",
        "gpt-sovits-v2pro",
        "gpt-sovits-v2proplus",
        "gpt-sovits-v3",
        "gpt-sovits-v4",
    ];

    for id in base_models {
        data.push(ModelObject {
            id: id.to_string(),
            object: "model".to_string(),
            created: now,
            owned_by: "official-base".to_string(),
        });
    }

    // 2. Custom Fine-tuned Models
    for (custom_id, _ver) in state.model_manager.list_custom_models() {
        data.push(ModelObject {
            id: custom_id,
            object: "model".to_string(),
            created: now,
            owned_by: "custom-finetuned".to_string(),
        });
    }

    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

/// Handler for GET /voices and GET /v1/voices
#[utoipa::path(
    get,
    path = "/v1/voices",
    tag = "Voices",
    summary = "List available voice presets",
    description = "Lists all voice presets defined in voices.toml.",
    security(
        ("bearerAuth" = [])
    ),
    responses(
        (status = 200, description = "List of voice presets", body = VoiceListResponse)
    )
)]
pub async fn list_voices(State(state): State<AppState>) -> Json<VoiceListResponse> {
    let raw_voices = state.voice_manager.list_voices();
    let data = raw_voices
        .into_iter()
        .map(|(name, preset)| VoiceObject {
            id: name.clone(),
            name,
            model_version: preset.model_version,
            prompt_lang: preset.prompt_lang,
            text_lang: preset.text_lang,
        })
        .collect();

    Json(VoiceListResponse {
        object: "list".to_string(),
        data,
    })
}

/// Handler for GET /health
#[utoipa::path(
    get,
    path = "/health",
    tag = "System",
    summary = "Service health check",
    description = "Returns system status and active capacity.",
    responses(
        (status = 200, description = "Service is operational", body = HealthResponse)
    )
)]
pub async fn health_check() -> (StatusCode, &'static str) {
    (StatusCode::OK, "{\"status\":\"ok\",\"version\":\"0.1.0\"}")
}
