use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use crate::server::error::AppError;

pub async fn auth_middleware(
    configured_api_key: String,
    req: Request,
    next: Next,
) -> Result<Response, AppError> {
    if configured_api_key.trim().is_empty() {
        // Authentication disabled
        return Ok(next.run(req).await);
    }

    if let Some(auth_header) = req.headers().get(axum::http::header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                if token.trim() == configured_api_key.trim() {
                    return Ok(next.run(req).await);
                }
            }
        }
    }

    Err(AppError::Unauthorized(
        "Incorrect API key provided. You can find your API key at https://platform.openai.com/account/api-keys.".to_string(),
    ))
}
