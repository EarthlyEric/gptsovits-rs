use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use serde_json::json;
use std::sync::Arc;
use tower::ServiceExt;

use gptsovits_rs::config::{AppConfig, CustomModelConfig};
use gptsovits_rs::engine::{ModelManager, ModelVersion};
use gptsovits_rs::server::create_router;
use gptsovits_rs::voice::VoiceManager;

fn setup_test_app() -> axum::Router {
    let mut config = AppConfig::default();
    config.server.api_key = "test-secret-token".to_string();

    // Register a custom fine-tuned model
    config.models.custom.insert(
        "sandrone".to_string(),
        CustomModelConfig {
            model_version: "v2ProPlus".to_string(),
            model_dir: "models/sandrone".to_string(),
            sampling_rate: Some(32000),
            sample_steps: Some(32),
            ..Default::default()
        },
    );

    let model_manager = Arc::new(ModelManager::new(&config));
    let voice_manager = Arc::new(VoiceManager::default());

    create_router(&config, model_manager, voice_manager)
}

#[tokio::test]
async fn test_health_check_endpoint() {
    let app = setup_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    assert!(body_str.contains("status"));
}

#[tokio::test]
async fn test_auth_failure_returns_401_openai_format() {
    let app = setup_test_app();

    let req_body = json!({
        "model": "gpt-sovits-v2",
        "input": "測試認證失敗",
        "voice": "alloy"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json_val: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json_val["error"]["code"], "invalid_api_key");
    assert_eq!(json_val["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn test_create_speech_success_all_base_versions() {
    let versions = [
        ("gpt-sovits-v1", ModelVersion::V1),
        ("gpt-sovits-v2", ModelVersion::V2),
        ("gpt-sovits-v2pro", ModelVersion::V2Pro),
        ("gpt-sovits-v2proplus", ModelVersion::V2ProPlus),
        ("gpt-sovits-v3", ModelVersion::V3),
        ("gpt-sovits-v4", ModelVersion::V4),
    ];

    for (model_name, _ver) in versions {
        let app = setup_test_app();

        let req_body = json!({
            "model": model_name,
            "input": "今天天氣真好，歡迎使用純 Rust GPT-SoVITS 推論伺服器！",
            "voice": "alloy",
            "response_format": "wav",
            "speed": 1.2
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/speech")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(header::AUTHORIZATION, "Bearer test-secret-token")
                    .body(Body::from(req_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        if status != StatusCode::OK {
            let body_str = String::from_utf8_lossy(&body);
            panic!("Failed for model {}: status={}, error={}", model_name, status, body_str);
        }
        assert_eq!(status, StatusCode::OK);
        assert!(!body.is_empty());
        assert_eq!(&body[0..4], b"RIFF");
    }
}

#[tokio::test]
async fn test_create_speech_with_custom_model_specified() {
    let app = setup_test_app();

    // Call with custom model name "sandrone"
    let req_body = json!({
        "model": "sandrone",
        "input": "這是使用自訂微調模型 Sandrone 進行的語音合成測試。",
        "voice": "sandrone",
        "response_format": "wav",
        "speed": 1.0
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer test-secret-token")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    if status != StatusCode::OK {
        let body_str = String::from_utf8_lossy(&body);
        panic!("Failed for custom model sandrone: status={}, error={}", status, body_str);
    }
    assert_eq!(status, StatusCode::OK);
    assert!(!body.is_empty());
    assert_eq!(&body[0..4], b"RIFF");
}

#[tokio::test]
async fn test_unknown_model_returns_404() {
    let app = setup_test_app();

    let req_body = json!({
        "model": "nonexistent-character-model",
        "input": "測試不存在的模型",
        "voice": "alloy"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer test-secret-token")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json_val: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json_val["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn test_create_speech_with_dynamic_voice_object() {
    let app = setup_test_app();

    let req_body = json!({
        "model": "gpt-sovits-v2",
        "input": "你好，這是動態 Voice 物件測試。",
        "voice": {
            "ref_audio_path": "",
            "prompt_text": "我是動態自訂聲音",
            "prompt_lang": "zh",
            "text_lang": "zh",
            "model_version": "v2",
            "top_k": 10,
            "temperature": 0.8
        },
        "response_format": "pcm"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer test-secret-token")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/octet-stream"
    );

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert!(!body.is_empty());
}

#[tokio::test]
async fn test_empty_input_returns_400() {
    let app = setup_test_app();

    let req_body = json!({
        "model": "gpt-sovits-v2",
        "input": "",
        "voice": "alloy"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer test-secret-token")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json_val: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json_val["error"]["code"], "empty_input");
}

#[tokio::test]
async fn test_invalid_speed_returns_400() {
    let app = setup_test_app();

    let req_body = json!({
        "model": "gpt-sovits-v2",
        "input": "測試無效語速",
        "voice": "alloy",
        "speed": 5.0
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer test-secret-token")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json_val: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json_val["error"]["code"], "invalid_speed");
}

#[tokio::test]
async fn test_models_and_voices_listing_with_custom_models() {
    let app = setup_test_app();

    // Models listing
    let res_models = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header(header::AUTHORIZATION, "Bearer test-secret-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res_models.status(), StatusCode::OK);
    let body = to_bytes(res_models.into_body(), usize::MAX).await.unwrap();
    let json_val: serde_json::Value = serde_json::from_slice(&body).unwrap();

    let models_data = json_val["data"].as_array().unwrap();
    
    // Check official base models
    let has_base = models_data.iter().any(|m| m["id"] == "gpt-sovits-v2" && m["owned_by"] == "official-base");
    assert!(has_base);

    // Check custom fine-tuned model
    let has_custom = models_data.iter().any(|m| m["id"] == "sandrone" && m["owned_by"] == "custom-finetuned");
    assert!(has_custom);

    // Voices listing
    let res_voices = app
        .oneshot(
            Request::builder()
                .uri("/v1/voices")
                .header(header::AUTHORIZATION, "Bearer test-secret-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res_voices.status(), StatusCode::OK);
}
