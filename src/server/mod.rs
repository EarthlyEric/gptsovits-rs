pub mod auth;
pub mod error;
pub mod openapi;
pub mod routes;
pub mod schema;

use axum::middleware::from_fn;
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_scalar::{Scalar, Servable as ScalarServable};
use utoipa_swagger_ui::SwaggerUi;

use crate::config::AppConfig;
use crate::engine::ModelManager;
use crate::server::auth::auth_middleware;
use crate::server::openapi::ApiDoc;
use crate::server::routes::{create_speech, health_check, list_models, list_voices, AppState};
use crate::voice::VoiceManager;

pub fn create_router(
    config: &AppConfig,
    model_manager: Arc<ModelManager>,
    voice_manager: Arc<VoiceManager>,
) -> Router {
    let api_key = config.server.api_key.clone();
    let max_concurrency = config.server.max_concurrency.max(1);
    let semaphore = Arc::new(Semaphore::new(max_concurrency));

    let state = AppState {
        model_manager,
        voice_manager,
        semaphore,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let api_routes = Router::new()
        .route("/audio/speech", post(create_speech))
        .route("/v1/audio/speech", post(create_speech))
        .route("/models", get(list_models))
        .route("/v1/models", get(list_models))
        .route("/voices", get(list_voices))
        .route("/v1/voices", get(list_voices))
        .route_layer(from_fn(move |req, next| {
            let key = api_key.clone();
            async move { auth_middleware(key, req, next).await }
        }));

    Router::new()
        .route("/health", get(health_check))
        .merge(SwaggerUi::new("/swagger-ui").url("/openapi.json", ApiDoc::openapi()))
        .merge(Scalar::with_url("/docs", ApiDoc::openapi()))
        .merge(api_routes)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
