use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use crate::server::schema::ErrorResponse;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Bad Request: {0}")]
    BadRequest(String, Option<&'static str>, Option<&'static str>),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Not Found: {0}")]
    NotFound(String, Option<&'static str>, Option<&'static str>),

    #[error("Rate Limit Exceeded: {0}")]
    RateLimit(String),

    #[error("Internal Server Error: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, err_resp) = match self {
            AppError::BadRequest(msg, param, code) => (
                StatusCode::BAD_REQUEST,
                ErrorResponse::new(msg, "invalid_request_error", param, code),
            ),
            AppError::Unauthorized(msg) => (
                StatusCode::UNAUTHORIZED,
                ErrorResponse::new(
                    msg,
                    "invalid_request_error",
                    None,
                    Some("invalid_api_key"),
                ),
            ),
            AppError::NotFound(msg, param, code) => (
                StatusCode::NOT_FOUND,
                ErrorResponse::new(msg, "invalid_request_error", param, code),
            ),
            AppError::RateLimit(msg) => (
                StatusCode::TOO_MANY_REQUESTS,
                ErrorResponse::new(
                    msg,
                    "rate_limit_error",
                    None,
                    Some("rate_limit_exceeded"),
                ),
            ),
            AppError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                ErrorResponse::new(msg, "api_error", None, Some("internal_error")),
            ),
        };

        (status, Json(err_resp)).into_response()
    }
}
