#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "onnx>=1.15.0",
# ]
# ///
"""Expose the trained V2Pro/V2ProPlus speaker embedding input in an old VITS graph."""

import argparse
import os
import tempfile

import onnx
from onnx import TensorProto, helper


def patch_model(path):
    model = onnx.load(path)
    if any(value.name == "sv_emb" for value in model.graph.input):
        print(f"[=] Already exposes sv_emb: {path}")
        return

    constant_outputs = {
        output
        for node in model.graph.node
        if "/sv_emb/Constant" in node.name
        for output in node.output
    }
    if not constant_outputs:
        raise ValueError(f"Speaker embedding constant not found: {path}")

    replacements = 0
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in constant_outputs:
                node.input[index] = "sv_emb"
                replacements += 1
    if replacements == 0:
        raise ValueError(f"Speaker embedding constant is unused: {path}")

    kept_nodes = [node for node in model.graph.node if "/sv_emb/Constant" not in node.name]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    model.graph.input.append(
        helper.make_tensor_value_info("sv_emb", TensorProto.FLOAT, [1, 20480])
    )
    onnx.checker.check_model(model)

    directory = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".onnx", delete=False) as temp:
        temporary_path = temp.name
    try:
        onnx.save(model, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    print(f"[+] Patched {path}: replacements={replacements}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="VITS ONNX files to patch in place")
    args = parser.parse_args()
    for path in args.models:
        patch_model(path)


if __name__ == "__main__":
    main()
