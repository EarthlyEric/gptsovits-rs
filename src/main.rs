use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use gptsovits_rs::config::{AppConfig, RuntimeConfig};
use gptsovits_rs::engine::ModelManager;
use gptsovits_rs::server::create_router;
use gptsovits_rs::voice::VoiceManager;

fn add_library_roots(roots: &mut Vec<PathBuf>, value: Option<std::ffi::OsString>) {
    let Some(value) = value else {
        return;
    };
    if value.is_empty() {
        return;
    }
    roots.extend(std::env::split_paths(&value));
}

fn cuda_library_roots(runtime: &RuntimeConfig) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    add_library_roots(&mut roots, Some(runtime.cuda_lib_dir.clone().into()));
    add_library_roots(&mut roots, Some(runtime.cudnn_lib_dir.clone().into()));
    add_library_roots(&mut roots, std::env::var_os("ORT_CUDA_LIB_DIR"));
    add_library_roots(&mut roots, std::env::var_os("ORT_CUDNN_LIB_DIR"));
    add_library_roots(&mut roots, std::env::var_os("LD_LIBRARY_PATH"));

    for base in [
        std::env::var_os("CUDA_HOME").map(PathBuf::from),
        std::env::var_os("CUDA_PATH").map(PathBuf::from),
    ]
    .into_iter()
    .flatten()
    {
        roots.push(base.clone());
        roots.push(base.join("lib64"));
        roots.push(base.join("lib"));
        roots.push(base.join("targets/x86_64-linux/lib"));
    }

    roots.sort();
    roots.dedup();
    roots
}

fn preload_library(name: &str, roots: &[PathBuf]) -> anyhow::Result<()> {
    for root in roots {
        let path = root.join(name);
        if path.is_file() {
            ort::util::preload_dylib(&path).map_err(|error| {
                anyhow::anyhow!("failed to preload {} from {}: {}", name, path.display(), error)
            })?;
            return Ok(());
        }
    }

    ort::util::preload_dylib(name).map_err(|error| {
        anyhow::anyhow!(
            "failed to preload {} from the dynamic linker search path: {}",
            name,
            error
        )
    })
}

fn preload_cuda_dependencies(runtime: &RuntimeConfig) -> anyhow::Result<()> {
    // ort's CUDA 13 distribution keeps these dependencies outside the provider .so.
    const REQUIRED_LIBRARIES: &[&str] = &[
        "libcudart.so.13",
        "libcublasLt.so.13",
        "libcublas.so.13",
        "libcurand.so.10",
    ];
    let roots = cuda_library_roots(runtime);
    let mut missing = Vec::new();

    for library in REQUIRED_LIBRARIES {
        if let Err(error) = preload_library(library, &roots) {
            tracing::debug!(library, %error, "CUDA dependency was not found");
            missing.push(*library);
        }
    }

    if !missing.is_empty() {
        anyhow::bail!(
            "CUDAExecutionProvider dependencies are missing: {}. Set runtime.cuda_lib_dir or ORT_CUDA_LIB_DIR to the CUDA 13 library directory",
            missing.join(", ")
        );
    }

    if let Err(error) = preload_library("libcudnn.so.9", &roots) {
        tracing::debug!(%error, "cuDNN library was not preloaded; ONNX Runtime may not require it");
    }
    Ok(())
}

fn initialize_onnx_runtime(runtime: &RuntimeConfig) -> anyhow::Result<()> {
    let device = runtime.device.trim().to_ascii_lowercase();
    let intra_threads = runtime.intra_threads.max(1);
    let inter_threads = runtime.inter_threads.max(1);

    let thread_options = ort::environment::GlobalThreadPoolOptions::default()
        .with_intra_threads(intra_threads)?
        .with_inter_threads(inter_threads)?
        .with_spin_control(false)?;

    let builder = ort::init()
        .with_telemetry(false)
        .with_global_thread_pool(thread_options);

    match device.as_str() {
        "cuda" => {
            preload_cuda_dependencies(runtime)?;
            let cuda_provider = ort::ep::CUDA::default()
                .with_device_id(runtime.cuda_device_id)
                .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                .with_conv_max_workspace(false)
                .with_tf32(true);

            if !ort::ep::ExecutionProvider::is_available(&cuda_provider)? {
                anyhow::bail!(
                    "CUDAExecutionProvider is not available in this ONNX Runtime build"
                );
            }

            if !builder
                .with_execution_providers([cuda_provider.build().error_on_failure()])
                .commit()
            {
                anyhow::bail!("ONNX Runtime environment was already initialized");
            }
            info!(
                device_id = runtime.cuda_device_id,
                intra_threads,
                inter_threads,
                "ONNX Runtime CUDAExecutionProvider registered with shared global thread pool"
            );
        }
        "cpu" => {
            if !builder.commit() {
                anyhow::bail!("ONNX Runtime environment was already initialized");
            }
            info!(
                intra_threads,
                inter_threads,
                "ONNX Runtime CPUExecutionProvider selected with shared global thread pool"
            );
        }
        _ => anyhow::bail!(
            "Unsupported runtime.device '{}'; expected 'cuda' or 'cpu'",
            device
        ),
    }

    info!("ONNX Runtime build: {}", ort::info());
    Ok(())
}

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

    // Configure the process-global ONNX Runtime environment before creating any session.
    initialize_onnx_runtime(&config.runtime)?;

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
