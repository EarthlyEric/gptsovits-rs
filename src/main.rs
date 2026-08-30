use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use gptsovits_rs::config::AppConfig;
use gptsovits_rs::engine::ModelManager;
use gptsovits_rs::server::create_router;
use gptsovits_rs::voice::VoiceManager;

#[derive(Parser, Debug)]
#[command(
    name = "gptsovits-rs",
    author = "earthlyeric6",
    version = "0.1.0",
    about = "Pure Rust High-Performance GPT-SoVITS Inference Engine & OpenAI TTS API Server (v1/v2/v2Pro/v2ProPlus/v3/v4)"
)]
struct Args {
    /// Path to configuration file (TOML)
    #[arg(short = 'c', long = "config", default_value = "config.toml")]
    config: String,

    /// Override server bind address
    #[arg(short = 'a', long = "bind")]
    bind_address: Option<String>,

    /// Override server port
    #[arg(short = 'p', long = "port")]
    port: Option<u16>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize Tracing Subscriber
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "gptsovits_rs=info,tower_http=info,axum=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    info!("Loading configuration from: {}", args.config);

    // 2. Load Configuration
    let mut config = AppConfig::load(&args.config).unwrap_or_else(|e| {
        error!("Failed to load config (using default): {}", e);
        AppConfig::default()
    });

    // Apply CLI overrides
    if let Some(bind) = args.bind_address {
        config.server.bind_address = bind;
    }
    if let Some(port) = args.port {
        config.server.port = port;
    }

    info!("Initializing GPT-SoVITS Pure Rust Inference Engine...");
    info!("Runtime device target: {}", config.runtime.device);

    // 3. Load Voice Manager
    let voice_manager = Arc::new(
        VoiceManager::from_file(&config.server.voices_config).unwrap_or_else(|e| {
            error!(
                "Failed to load voices config ({}), using fallback presets: {}",
                config.server.voices_config, e
            );
            VoiceManager::default()
        }),
    );

    // 4. Load Model Manager
    let model_manager = Arc::new(ModelManager::new(&config));
    info!(
        "Engine initialized. Default model version: {:?}",
        model_manager.default_version()
    );

    // 5. Build Axum Router
    let app = create_router(&config, model_manager, voice_manager);

    let addr_str = format!("{}:{}", config.server.bind_address, config.server.port);
    let addr: SocketAddr = addr_str.parse().expect("Invalid socket address format");

    info!("OpenAI-compatible TTS Server listening on http://{}", addr);
    info!(
        "Speech endpoint available at POST http://{}/v1/audio/speech and http://{}/audio/speech",
        addr, addr
    );

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown cleanly.");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Ctrl+C signal received, initiating graceful shutdown...");
        },
        _ = terminate => {
            info!("SIGTERM signal received, initiating graceful shutdown...");
        },
    }
}
