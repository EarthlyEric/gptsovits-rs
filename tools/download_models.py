#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub>=0.23.0",
#     "requests>=2.31.0",
#     "tqdm>=4.65.0",
# ]
# ///
"""
GPT-SoVITS Pretrained & Fine-tuned Models Downloader
Downloads official pretrained model weights from HuggingFace, HF-Mirror, or ModelScope for fresh environments.
Supports: Base, v1, v2, v2Pro, v2ProPlus, v3, v4, and custom character models (e.g. sandrone).
"""

import os
import sys
import argparse
import subprocess
import requests
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
default_target_dir = os.path.join(project_root, "GPT-SoVITS", "GPT_SoVITS", "pretrained_models")

# Model file mappings: (repo_id, rel_url, rel_dest)
MODELS_MANIFEST = {
    "base": [
        ("lj1995/GPT-SoVITS", "chinese-hubert-base/config.json", "chinese-hubert-base/config.json"),
        ("lj1995/GPT-SoVITS", "chinese-hubert-base/pytorch_model.bin", "chinese-hubert-base/pytorch_model.bin"),
        ("lj1995/GPT-SoVITS", "chinese-hubert-base/preprocessor_config.json", "chinese-hubert-base/preprocessor_config.json"),
        ("lj1995/GPT-SoVITS", "chinese-roberta-wwm-ext-large/config.json", "chinese-roberta-wwm-ext-large/config.json"),
        ("lj1995/GPT-SoVITS", "chinese-roberta-wwm-ext-large/pytorch_model.bin", "chinese-roberta-wwm-ext-large/pytorch_model.bin"),
        ("lj1995/GPT-SoVITS", "chinese-roberta-wwm-ext-large/tokenizer.json", "chinese-roberta-wwm-ext-large/tokenizer.json"),
    ],
    "v1": [
        ("lj1995/GPT-SoVITS", "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"),
        ("lj1995/GPT-SoVITS", "s2G488k.pth", "s2G488k.pth"),
    ],
    "v2": [
        ("lj1995/GPT-SoVITS", "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
        ("lj1995/GPT-SoVITS", "gsv-v2final-pretrained/s2G2333k.pth", "gsv-v2final-pretrained/s2G2333k.pth"),
    ],
    "v2pro": [
        ("lj1995/GPT-SoVITS", "v2Pro/s2Gv2Pro.pth", "v2Pro/s2Gv2Pro.pth"),
        ("lj1995/GPT-SoVITS", "v2Pro/s2Gv2ProPlus.pth", "v2Pro/s2Gv2ProPlus.pth"),
        ("lj1995/GPT-SoVITS", "sv/pretrained_eres2netv2w24s4ep4.ckpt", "sv/pretrained_eres2netv2w24s4ep4.ckpt"),
    ],
    "v3": [
        ("lj1995/GPT-SoVITS", "s1v3.ckpt", "s1v3.ckpt"),
        ("lj1995/GPT-SoVITS", "s2Gv3.pth", "s2Gv3.pth"),
    ],
    "v4": [
        ("lj1995/GPT-SoVITS", "s1v3.ckpt", "s1v3.ckpt"),
        ("lj1995/GPT-SoVITS", "gsv-v4-pretrained/s2Gv4.pth", "gsv-v4-pretrained/s2Gv4.pth"),
        ("lj1995/GPT-SoVITS", "gsv-v4-pretrained/vocoder.pth", "gsv-v4-pretrained/vocoder.pth"),
    ],
    "sandrone": [
        ("EarthlyEric6/Sandrone_gptsovits", "GPT_weights_v2ProPlus/Sandrone_v2proplus-e15.ckpt", "sandrone/Sandrone_v2proplus-e15.ckpt"),
        ("EarthlyEric6/Sandrone_gptsovits", "SoVITS_weights_v2ProPlus/Sandrone_v2proplus_e8_s1368.pth", "sandrone/Sandrone_v2proplus_e8_s1368.pth"),
        ("EarthlyEric6/Sandrone_gptsovits", "sandrone_ref.wav", "__VOICES__/sandrone/ref.wav"),
        ("EarthlyEric6/Sandrone_gptsovits", "sandrone_ref_text.txt", "__VOICES__/sandrone/ref.txt"),
    ],
}

def download_file(url, target_path, resume=True):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    headers = {}
    temp_path = target_path + ".tmp"
    existing_bytes = 0

    if resume and os.path.exists(temp_path):
        existing_bytes = os.path.getsize(temp_path)
        headers["Range"] = f"bytes={existing_bytes}-"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        # Check if already completed
        if response.status_code == 416: # Range not satisfiable
            if os.path.exists(temp_path):
                os.replace(temp_path, target_path)
                return True

        if response.status_code not in (200, 206):
            print(f"[!] Error: Server returned status {response.status_code} for {url}")
            return False

        total_size = int(response.headers.get("content-length", 0)) + existing_bytes
        desc = os.path.basename(target_path)

        mode = "ab" if existing_bytes > 0 else "wb"
        with open(temp_path, mode) as f, tqdm(
            desc=desc,
            total=total_size,
            initial=existing_bytes,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        os.replace(temp_path, target_path)
        return True
    except Exception as e:
        print(f"[!] Download failed for {url}: {e}")
        return False

def get_base_url(source, repo="lj1995/GPT-SoVITS"):
    if source == "hf-mirror":
        return f"https://hf-mirror.com/{repo}/resolve/main"
    elif source == "modelscope" and repo == "lj1995/GPT-SoVITS":
        return "https://www.modelscope.cn/api/v1/models/iic/GPT-SoVITS/repo?Revision=master&FilePath="
    else:
        return f"https://huggingface.co/{repo}/resolve/main"

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Pretrained & Fine-tuned Models Downloader")
    parser.add_argument(
        "--version",
        type=str,
        default="all",
        choices=["all", "base", "v1", "v2", "v2pro", "v2proplus", "v3", "v4", "sandrone"],
        help="Target model version / character to download (default: all)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "hf-mirror", "modelscope"],
        help="Download source / mirror (default: huggingface)",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=default_target_dir,
        help=f"Target directory for pretrained models (default: {default_target_dir})",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Automatically trigger ONNX export after download completes",
    )

    args = parser.parse_args()

    # Determine which categories to download
    categories = []
    if args.version == "all":
        categories = list(MODELS_MANIFEST.keys())
    elif args.version == "base":
        categories = ["base"]
    elif args.version in ("v2proplus", "v2pro"):
        categories = ["base", "v2", "v2pro"]
    elif args.version in ("v3", "v4"):
        categories = ["base", "v3", args.version]
    elif args.version == "sandrone":
        categories = ["base", "sandrone"]
    else:
        categories = ["base", args.version]

    # Collect files
    files_to_download = []
    for cat in categories:
        if cat in MODELS_MANIFEST:
            files_to_download.extend(MODELS_MANIFEST[cat])

    print("==========================================================")
    print("  GPT-SoVITS Model Downloader (uv compatible)")
    print(f"  Target Version: {args.version}")
    print(f"  Source Mirror:  {args.source}")
    print(f"  Target Path:    {args.target_dir}")
    print(f"  Total Files:    {len(files_to_download)}")
    print("==========================================================")

    os.makedirs(args.target_dir, exist_ok=True)

    success_count = 0
    for repo_id, rel_url, rel_dest in files_to_download:
        if rel_dest.startswith("__VOICES__/"):
            voice_rel = rel_dest.replace("__VOICES__/", "")
            dest_path = os.path.join(project_root, "voices", voice_rel)
        else:
            dest_path = os.path.join(args.target_dir, rel_dest)
        
        # Check if file already exists
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10:
            print(f"[✓] Already exists: {os.path.basename(dest_path)}")
            success_count += 1
            continue

        base_url = get_base_url(args.source, repo_id)
        if args.source == "modelscope" and repo_id == "lj1995/GPT-SoVITS":
            url = f"{base_url}{rel_url}"
        else:
            url = f"{base_url}/{rel_url}?download=true"

        print(f"\n[*] Downloading: {os.path.basename(dest_path)} (from {repo_id})")
        if download_file(url, dest_path):
            success_count += 1
        else:
            print(f"[!] Warning: Could not download {rel_dest}")

    print("\n==========================================================")
    print(f"  Download Finished: {success_count}/{len(files_to_download)} files ready.")
    print("==========================================================")

    # Optional: trigger ONNX export
    if args.export_onnx:
        import shutil
        exporter_script = os.path.join(script_dir, "onnx_exporter.py")
        uv_cmd = shutil.which("uv")
        base_cmd = [uv_cmd, "run", exporter_script] if uv_cmd else [sys.executable, exporter_script]
        base_cmd.extend([
            "--cnhubert-path", os.path.join(args.target_dir, "chinese-hubert-base"),
            "--bert-path", os.path.join(args.target_dir, "chinese-roberta-wwm-ext-large"),
            "--output-dir", "models",
        ])

        if args.version.lower() == "sandrone":
            print(f"\n[*] Triggering ONNX export for Sandrone custom fine-tuned model...")
            gpt_p = os.path.join(args.target_dir, "sandrone", "Sandrone_v2proplus-e15.ckpt")
            sovits_p = os.path.join(args.target_dir, "sandrone", "Sandrone_v2proplus_e8_s1368.pth")
            cmd = list(base_cmd)
            cmd.extend([
                "--version", "v2ProPlus",
                "--custom-name", "sandrone",
                "--gpt-path", gpt_p,
                "--sovits-path", sovits_p,
            ])
            subprocess.run(cmd)
        else:
            raw_ver = "v2" if args.version in ("all", "base") else args.version
            canonical_map = {
                "v1": "v1",
                "v2": "v2",
                "v2pro": "v2Pro",
                "v2proplus": "v2ProPlus",
                "v3": "v3",
                "v4": "v4",
            }
            export_ver = canonical_map.get(raw_ver.lower(), raw_ver)
            print(f"\n[*] Triggering ONNX export for version: {export_ver}...")
            cmd = list(base_cmd)
            cmd.extend(["--version", export_ver])

            version_checkpoints = {
                "v1": ("s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", "s2G488k.pth"),
                "v2": ("gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "gsv-v2final-pretrained/s2G2333k.pth"),
                "v2pro": ("gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "v2Pro/s2Gv2Pro.pth"),
                "v2proplus": ("gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "v2Pro/s2Gv2ProPlus.pth"),
                "v3": ("s1v3.ckpt", "s2Gv3.pth"),
                "v4": ("s1v3.ckpt", "gsv-v4-pretrained/s2Gv4.pth"),
            }
            if export_ver.lower() in version_checkpoints:
                gpt_rel, sovits_rel = version_checkpoints[export_ver.lower()]
                gpt_p = os.path.join(args.target_dir, gpt_rel)
                sovits_p = os.path.join(args.target_dir, sovits_rel)
                if os.path.exists(gpt_p) and os.path.exists(sovits_p):
                    cmd.extend(["--gpt-path", gpt_p, "--sovits-path", sovits_p])

            subprocess.run(cmd)

if __name__ == "__main__":
    main()
