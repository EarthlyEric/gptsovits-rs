use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use serde_json::json;

/// Returns the OpenAPI 3.1.0 specification JSON string
pub fn generate_openapi_spec() -> serde_json::Value {
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "GPT-SoVITS Pure Rust Inference Engine & OpenAI TTS API",
            "version": "0.1.0",
            "description": "High-performance pure Rust inference engine and 100% OpenAI-compatible TTS server for GPT-SoVITS (supporting v1, v2, v2Pro, v2ProPlus, v3, and v4 models).",
            "license": {
                "name": "MIT or Apache-2.0"
            }
        },
        "servers": [
            {
                "url": "/",
                "description": "Current Server Instance"
            }
        ],
        "paths": {
            "/v1/audio/speech": {
                "post": {
                    "summary": "Create speech from input text",
                    "description": "Generates audio from the input text using OpenAI-compatible parameters, with GPT-SoVITS zero-shot voice cloning and text segmentation support.",
                    "operationId": "createSpeech",
                    "security": [
                        {
                            "bearerAuth": []
                        }
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/CreateSpeechRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "The synthesized audio file in the requested format or streamed chunks.",
                            "content": {
                                "audio/mpeg": {
                                    "schema": {
                                        "type": "string",
                                        "format": "binary"
                                    }
                                },
                                "audio/wav": {
                                    "schema": {
                                        "type": "string",
                                        "format": "binary"
                                    }
                                },
                                "audio/opus": {
                                    "schema": {
                                        "type": "string",
                                        "format": "binary"
                                    }
                                },
                                "audio/aac": {
                                    "schema": {
                                        "type": "string",
                                        "format": "binary"
                                    }
                                },
                                "audio/flac": {
                                    "schema": {
                                        "type": "string",
                                        "format": "binary"
                                    }
                                },
                                "application/octet-stream": {
                                    "schema": {
                                        "type": "string",
                                        "format": "binary"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad Request (e.g. empty input, input exceeds 4096 characters, invalid speed).",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        },
                        "401": {
                            "description": "Unauthorized (missing or invalid API bearer token).",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Not Found (requested model or voice not configured).",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Internal Server Error during inference or audio encoding.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/audio/speech": {
                "post": {
                    "summary": "Create speech (alternative path without /v1 prefix)",
                    "description": "Direct alias for /v1/audio/speech.",
                    "operationId": "createSpeechAlt",
                    "security": [
                        {
                            "bearerAuth": []
                        }
                    ],
                    "requestBody": {
                        "$ref": "#/paths/~1v1~1audio~1speech/post/requestBody"
                    },
                    "responses": {
                        "$ref": "#/paths/~1v1~1audio~1speech/post/responses"
                    }
                }
            },
            "/v1/models": {
                "get": {
                    "summary": "List available TTS models",
                    "description": "Lists official base models (e.g. gpt-sovits-v1..v4) and custom fine-tuned models registered in config.toml.",
                    "operationId": "listModels",
                    "security": [
                        {
                            "bearerAuth": []
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of available models",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ModelListResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/v1/voices": {
                "get": {
                    "summary": "List available voice presets",
                    "description": "Lists all voice presets defined in voices.toml with their configured languages and parameters.",
                    "operationId": "listVoices",
                    "security": [
                        {
                            "bearerAuth": []
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of available voice presets",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/VoiceListResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "summary": "Health and status check",
                    "description": "Returns the server health status and active worker capacity.",
                    "operationId": "healthCheck",
                    "responses": {
                        "200": {
                            "description": "Server is running normally",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {
                                                "type": "string",
                                                "example": "ok"
                                            },
                                            "engine": {
                                                "type": "string",
                                                "example": "gptsovits-rs"
                                            },
                                            "version": {
                                                "type": "string",
                                                "example": "0.1.0"
                                            },
                                            "available_permits": {
                                                "type": "integer",
                                                "example": 8
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "API Key",
                    "description": "Enter your API key configured in config.toml (or leave empty if api_key is disabled)"
                }
            },
            "schemas": {
                "CreateSpeechRequest": {
                    "type": "object",
                    "required": ["model", "input", "voice"],
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "Model ID to synthesize speech (e.g. gpt-sovits-v2, gpt-sovits-v4, or custom model name like sandrone)",
                            "example": "gpt-sovits-v2"
                        },
                        "input": {
                            "type": "string",
                            "description": "The text to generate audio for. Maximum length is 4096 characters.",
                            "example": "先帝創業未半而中道崩殂，今天下三分，益州疲弊。"
                        },
                        "voice": {
                            "description": "The voice to use for generation. Can be a string identifier (e.g. 'default', 'sandrone') or a dynamic custom voice object for zero-shot cloning.",
                            "oneOf": [
                                {
                                    "type": "string",
                                    "example": "default"
                                },
                                {
                                    "$ref": "#/components/schemas/DynamicVoiceObject"
                                }
                            ]
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Optional instructions controlling speaking style, tone, or emotion.",
                            "example": "Speak in a gentle and polite tone."
                        },
                        "response_format": {
                            "type": "string",
                            "enum": ["mp3", "opus", "aac", "flac", "wav", "pcm"],
                            "default": "mp3",
                            "description": "The audio format to return."
                        },
                        "speed": {
                            "type": "number",
                            "minimum": 0.25,
                            "maximum": 4.0,
                            "default": 1.0,
                            "description": "The speed of the generated audio. Select values from 0.25 to 4.0. 1.0 is default."
                        },
                        "stream_format": {
                            "type": "string",
                            "enum": ["audio", "sse"],
                            "default": "audio",
                            "description": "Streaming delivery format (audio stream or Server-Sent Events)."
                        }
                    }
                },
                "DynamicVoiceObject": {
                    "type": "object",
                    "properties": {
                        "ref_audio_path": {
                            "type": "string",
                            "description": "Local path to 3~10s reference audio wav file for voice cloning",
                            "example": "voices/sandrone/ref.wav"
                        },
                        "prompt_text": {
                            "type": "string",
                            "description": "Transcript of the reference audio for cross-lingual / zero-shot synthesis",
                            "example": "我是「木偶」桑多涅。"
                        },
                        "prompt_lang": {
                            "type": "string",
                            "description": "Language of prompt_text (zh, en, ja, ko, yue, auto)",
                            "example": "zh"
                        },
                        "text_lang": {
                            "type": "string",
                            "description": "Language of input text (zh, en, ja, ko, yue, auto)",
                            "example": "zh"
                        },
                        "model_version": {
                            "type": "string",
                            "description": "Model architecture version (v1, v2, v2Pro, v2ProPlus, v3, v4)",
                            "example": "v2ProPlus"
                        },
                        "text_split_method": {
                            "type": "string",
                            "enum": ["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"],
                            "default": "cut5",
                            "description": "Text segmentation strategy: cut0(no cut), cut1(every 4 sentences), cut2(every ~50 chars), cut3(Chinese period), cut4(English period), cut5(all punctuations, lowest latency)"
                        },
                        "fragment_interval": {
                            "type": "number",
                            "default": 0.2,
                            "description": "Silence pause between segmented sentences (seconds)"
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 15,
                            "description": "Top-K sampling parameter"
                        },
                        "top_p": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Top-P sampling parameter"
                        },
                        "temperature": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Sampling temperature"
                        },
                        "repetition_penalty": {
                            "type": "number",
                            "default": 1.35,
                            "description": "Repetition penalty for autoregressive semantic token generation"
                        }
                    }
                },
                "ModelObject": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "example": "gpt-sovits-v2"
                        },
                        "object": {
                            "type": "string",
                            "example": "model"
                        },
                        "created": {
                            "type": "integer",
                            "example": 1700000000
                        },
                        "owned_by": {
                            "type": "string",
                            "example": "official-base"
                        }
                    }
                },
                "ModelListResponse": {
                    "type": "object",
                    "properties": {
                        "object": {
                            "type": "string",
                            "example": "list"
                        },
                        "data": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/ModelObject"
                            }
                        }
                    }
                },
                "VoiceObject": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "example": "default"
                        },
                        "name": {
                            "type": "string",
                            "example": "Default Chinese Preset"
                        },
                        "model_version": {
                            "type": "string",
                            "example": "v2"
                        },
                        "prompt_lang": {
                            "type": "string",
                            "example": "zh"
                        },
                        "text_lang": {
                            "type": "string",
                            "example": "zh"
                        }
                    }
                },
                "VoiceListResponse": {
                    "type": "object",
                    "properties": {
                        "object": {
                            "type": "string",
                            "example": "list"
                        },
                        "data": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/VoiceObject"
                            }
                        }
                    }
                },
                "ErrorDetail": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "example": "Input text cannot be empty."
                        },
                        "type": {
                            "type": "string",
                            "example": "invalid_request_error"
                        },
                        "param": {
                            "type": "string",
                            "nullable": true,
                            "example": "input"
                        },
                        "code": {
                            "type": "string",
                            "nullable": true,
                            "example": "empty_input"
                        }
                    }
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "$ref": "#/components/schemas/ErrorDetail"
                        }
                    }
                }
            }
        }
    })
}

/// Handler for GET /openapi.json
pub async fn openapi_json() -> impl IntoResponse {
    let spec = generate_openapi_spec();
    (
        [(header::CONTENT_TYPE, "application/json; charset=utf-8")],
        spec.to_string(),
    )
}

/// Handler for GET /docs (Scalar Interactive Documentation)
pub async fn scalar_docs() -> impl IntoResponse {
    let html = r#"<!doctype html>
<html>
  <head>
    <title>GPT-SoVITS API Documentation</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎙️</text></svg>">
    <style>
      body {
        margin: 0;
        padding: 0;
      }
    </style>
  </head>
  <body>
    <script
      id="api-reference"
      data-url="/openapi.json"
      data-configuration='{"theme":"purple","showSidebar":true,"darkMode":true}'
      src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>"#;

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        Html(html),
    )
}

/// Handler for GET /swagger-ui (Swagger UI Interactive Documentation)
pub async fn swagger_ui() -> Response {
    let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GPT-SoVITS API - Swagger UI</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>">
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin: 0; background: #fafafa; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
    window.onload = function() {
        window.ui = SwaggerUIBundle({
            url: "/openapi.json",
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIStandalonePreset
            ],
            layout: "BaseLayout"
        });
    };
    </script>
</body>
</html>"#;

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        Html(html),
    )
        .into_response()
}
