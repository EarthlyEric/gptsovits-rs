#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.1.0",
#     "torchaudio>=2.1.0",
#     "transformers>=4.40.0",
#     "onnx>=1.15.0",
#     "onnxruntime>=1.17.0",
#     "scipy>=1.11.0",
#     "numpy>=1.24.0",
#     "soundfile>=0.12.0",
#     "librosa>=0.10.0",
#     "pypinyin>=0.50.0",
#     "cn2an>=0.5.0",
#     "jieba>=0.42.0",
#     "pyyaml>=6.0",
#     "tqdm>=4.65.0",
#     "einops>=0.7.0",
# ]
# ///
"""
GPT-SoVITS to ONNX Exporter Tool (Runnable via `uv run tools/onnx_exporter.py`)
Exports PyTorch weights (.ckpt and .pth) to ONNX models for high-performance pure Rust inference.
Supports: v1, v2, v2Pro, v2ProPlus, v3, v4
"""

import os
import sys
import argparse
import shutil
import json

# Add GPT-SoVITS directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
gpt_sovits_dir = os.path.join(project_root, "GPT-SoVITS")
sys.path.insert(0, gpt_sovits_dir)
sys.path.insert(0, os.path.join(gpt_sovits_dir, "GPT_SoVITS"))

try:
    import torch
    import torchaudio
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except ImportError:
    print("Warning: PyTorch/Transformers not installed in this environment. Run with `uv run tools/onnx_exporter.py`.")

def export_cnhubert(cnhubert_path, output_path):
    print(f"[*] Exporting CNHuBERT SSL Model from {cnhubert_path} to {output_path}...")
    from feature_extractor import cnhubert
    cnhubert.cnhubert_base_path = cnhubert_path
    ssl_model = cnhubert.get_model()

    class SSLWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, ref_audio_16k):
            return self.model.model(ref_audio_16k)["last_hidden_state"].transpose(1, 2)

    wrapper = SSLWrapper(ssl_model).eval()
    dummy_audio = torch.randn(1, 16000 * 3) # 3 seconds

    torch.onnx.export(
        wrapper,
        (dummy_audio,),
        output_path,
        input_names=["ref_audio_16k"],
        output_names=["ssl_content"],
        dynamic_axes={
            "ref_audio_16k": {1: "audio_length"},
            "ssl_content": {2: "ssl_length"},
        },
        opset_version=17,
        verbose=False,
    )
    print(f"[+] CNHuBERT exported successfully: {output_path}")

def export_roberta(bert_path, output_path, output_tok_path):
    print(f"[*] Exporting Chinese-RoBERTa Model from {bert_path} to {output_path}...")
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    model = AutoModelForMaskedLM.from_pretrained(bert_path).eval()

    # Save tokenizer.json
    tokenizer.save_pretrained(os.path.dirname(output_tok_path))
    if hasattr(tokenizer, "save_vocabulary"):
        tokenizer.save_vocabulary(os.path.dirname(output_tok_path))

    class BertWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
            )
            # Hidden state at layer -3
            return outputs.hidden_states[-3]

    wrapper = BertWrapper(model)
    dummy_ids = torch.randint(0, 1000, (1, 32), dtype=torch.long)
    dummy_mask = torch.ones((1, 32), dtype=torch.long)
    dummy_types = torch.zeros((1, 32), dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask, dummy_types),
        output_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
            "token_type_ids": {1: "seq_len"},
            "hidden_states": {1: "seq_len"},
        },
        opset_version=17,
        verbose=False,
    )
    print(f"[+] RoBERTa exported successfully: {output_path}")

def export_t2s(gpt_path, vits_path, output_dir, version="v2"):
    print(f"[*] Exporting T2S AR Models ({version}) from {gpt_path}...")
    from onnx_export import T2SModel, VitsModel

    vits = VitsModel(vits_path)
    gpt = T2SModel(gpt_path, vits)

    ref_seq = torch.randint(0, 300, (1, 20), dtype=torch.long)
    text_seq = torch.randint(0, 300, (1, 30), dtype=torch.long)
    ref_bert = torch.randn((20, 1024), dtype=torch.float32)
    text_bert = torch.randn((30, 1024), dtype=torch.float32)
    ssl_content = torch.randn((1, 768, 50), dtype=torch.float32)

    enc_path = os.path.join(output_dir, "t2s_encoder.onnx")
    fsdec_path = os.path.join(output_dir, "t2s_fsdec.onnx")
    sdec_path = os.path.join(output_dir, "t2s_sdec.onnx")

    # 1. T2S Encoder
    torch.onnx.export(
        gpt.onnx_encoder,
        (ref_seq, text_seq, ref_bert, text_bert, ssl_content),
        enc_path,
        input_names=["ref_seq", "text_seq", "ref_bert", "text_bert", "ssl_content"],
        output_names=["x", "prompts"],
        dynamic_axes={
            "ref_seq": {1: "ref_length"},
            "text_seq": {1: "text_length"},
            "ref_bert": {0: "ref_length"},
            "text_bert": {0: "text_length"},
            "ssl_content": {2: "ssl_length"},
            "x": {1: "x_length"},
            "prompts": {1: "prompts_length"},
        },
        opset_version=17,
        verbose=False,
    )
    print(f"[+] Exported t2s_encoder.onnx: {enc_path}")

    # 2. First stage decoder
    x, prompts = gpt.onnx_encoder(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
    torch.onnx.export(
        gpt.first_stage_decoder,
        (x, prompts),
        fsdec_path,
        input_names=["x", "prompts"],
        output_names=["y", "k", "v", "y_emb", "x_example"],
        dynamic_axes={
            "x": {1: "x_length"},
            "prompts": {1: "prompts_length"},
        },
        opset_version=17,
        verbose=False,
    )
    print(f"[+] Exported t2s_fsdec.onnx: {fsdec_path}")

    # 3. Stage decoder
    y, k, v, y_emb, x_example = gpt.first_stage_decoder(x, prompts)
    torch.onnx.export(
        gpt.stage_decoder,
        (y, k, v, y_emb, x_example),
        sdec_path,
        input_names=["iy", "ik", "iv", "iy_emb", "ix_example"],
        output_names=["y", "k", "v", "y_emb", "logits", "samples"],
        dynamic_axes={
            "iy": {1: "iy_length"},
            "ik": {1: "ik_length"},
            "iv": {1: "iv_length"},
            "iy_emb": {1: "iy_emb_length"},
            "ix_example": {1: "ix_example_length"},
        },
        opset_version=17,
        verbose=False,
    )
    print(f"[+] Exported t2s_sdec.onnx: {sdec_path}")

def export_vits(vits_path, output_dir, version="v2"):
    print(f"[*] Exporting VITS Synthesizer ({version}) from {vits_path}...")
    from onnx_export import VitsModel

    vits = VitsModel(vits_path)
    vits_onnx_path = os.path.join(output_dir, "vits.onnx")

    text_seq = torch.randint(0, 300, (1, 30), dtype=torch.long)
    pred_semantic = torch.randint(0, 1024, (1, 1, 60), dtype=torch.long)
    ref_audio = torch.randn((1, 32000 * 3), dtype=torch.float32)

    torch.onnx.export(
        vits,
        (text_seq, pred_semantic, ref_audio),
        vits_onnx_path,
        input_names=["text_seq", "pred_semantic", "ref_audio"],
        output_names=["audio"],
        dynamic_axes={
            "text_seq": {1: "text_length"},
            "pred_semantic": {2: "pred_length"},
            "ref_audio": {1: "audio_length"},
            "audio": {1: "out_length"},
        },
        opset_version=17,
        verbose=False,
    )
    print(f"[+] Exported vits.onnx: {vits_onnx_path}")

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS to ONNX Exporter for Rust Inference")
    parser.add_argument("--gpt-path", type=str, help="Path to GPT/T2S checkpoint (.ckpt)")
    parser.add_argument("--sovits-path", type=str, help="Path to SoVITS checkpoint (.pth)")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4"], help="Model version")
    parser.add_argument("--cnhubert-path", type=str, default="GPT_SoVITS/pretrained_models/chinese-hubert-base", help="CNHuBERT directory")
    parser.add_argument("--bert-path", type=str, default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", help="Chinese RoBERTa directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for ONNX models")

    args = parser.parse_args()

    out_version_dir = os.path.join(args.output_dir, args.version)
    os.makedirs(out_version_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "chinese-hubert-base"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "chinese-roberta-wwm-ext-large"), exist_ok=True)

    print("==========================================================")
    print("  GPT-SoVITS Pure Rust ONNX Model Exporter (uv compatible)")
    print(f"  Target Version: {args.version}")
    print(f"  Output Directory: {out_version_dir}")
    print("==========================================================")

    # 1. CNHuBERT
    cnhubert_out = os.path.join(args.output_dir, "chinese-hubert-base", "cnhubert.onnx")
    if os.path.exists(args.cnhubert_path) and not os.path.exists(cnhubert_out):
        export_cnhubert(args.cnhubert_path, cnhubert_out)

    # 2. RoBERTa
    bert_out = os.path.join(args.output_dir, "chinese-roberta-wwm-ext-large", "bert.onnx")
    tok_out = os.path.join(args.output_dir, "chinese-roberta-wwm-ext-large", "tokenizer.json")
    if os.path.exists(args.bert_path) and not os.path.exists(bert_out):
        export_roberta(args.bert_path, bert_out, tok_out)

    # 3. T2S & VITS
    if args.gpt_path and args.sovits_path:
        export_t2s(args.gpt_path, args.sovits_path, out_version_dir, args.version)
        export_vits(args.sovits_path, out_version_dir, args.version)

    print("\n[✓] Export completed successfully!")
    print(f"Update your config.toml to point to models under '{args.output_dir}'")

if __name__ == "__main__":
    main()
