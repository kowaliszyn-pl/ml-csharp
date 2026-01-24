#!/usr/bin/env python3
"""Utility helpers for downloading GPT-2 checkpoints and exporting weights.

The exported binary format is consumed by the C# `Gpt2Model` implementation
and stores every tensor in little-endian row-major order.
"""

import argparse
import json
import os
import re
import struct
from typing import Dict, Iterable, Tuple

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

from encoder import get_encoder

MAGIC = b"GPT2WEIGHTS"
FORMAT_VERSION = 1
MODEL_SIZES = ["124M", "355M", "774M", "1558M"]


def download_gpt2_files(model_size: str, model_dir: str) -> None:
    assert model_size in MODEL_SIZES
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        response = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        response.raise_for_status()

        path = os.path.join(model_dir, filename)
        file_size = int(response.headers.get("content-length", 0))
        chunk_size = 1000
        with open(path, "wb") as handle, tqdm(
            ncols=100,
            desc=f"Fetching {filename}",
            total=file_size,
            unit_scale=True,
            unit="b",
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                handle.write(chunk)
                progress.update(len(chunk))


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path: str, hparams: Dict) -> Dict:
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name[len("model/") :]
        if name.startswith("h"):
            match = re.match(r"h([0-9]+)/(.*)", name)
            block_index = int(match[1])
            sub_name = match[2]
            set_in_nested_dict(params["blocks"][block_index], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def load_encoder_hparams_and_params(model_size: str, models_dir: str):
    assert model_size in MODEL_SIZES

    model_dir = os.path.join(models_dir, model_size)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not tf_ckpt_path:
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2_files(model_size, model_dir)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    encoder = get_encoder(model_size, models_dir)
    hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

    return encoder, hparams, params


def iter_named_tensors(params: Dict) -> Iterable[Tuple[str, np.ndarray]]:
    yield "token_embeddings", params["wte"]
    yield "positional_embeddings", params["wpe"]

    for index, block in enumerate(params["blocks"]):
        prefix = f"blocks.{index}"
        attn = block["attn"]
        mlp = block["mlp"]

        yield f"{prefix}.attn.qkv.weight", attn["c_attn"]["w"]
        yield f"{prefix}.attn.qkv.bias", attn["c_attn"]["b"]
        yield f"{prefix}.attn.out.weight", attn["c_proj"]["w"]
        yield f"{prefix}.attn.out.bias", attn["c_proj"]["b"]

        yield f"{prefix}.mlp.up.weight", mlp["c_fc"]["w"]
        yield f"{prefix}.mlp.up.bias", mlp["c_fc"]["b"]
        yield f"{prefix}.mlp.down.weight", mlp["c_proj"]["w"]
        yield f"{prefix}.mlp.down.bias", mlp["c_proj"]["b"]

        yield f"{prefix}.ln1.gamma", block["ln_1"]["g"]
        yield f"{prefix}.ln1.beta", block["ln_1"]["b"]
        yield f"{prefix}.ln2.gamma", block["ln_2"]["g"]
        yield f"{prefix}.ln2.beta", block["ln_2"]["b"]

    yield "final_layer_norm.gamma", params["ln_f"]["g"]
    yield "final_layer_norm.beta", params["ln_f"]["b"]


def export_gpt2_weights(params: Dict, output_path: str) -> int:
    tensors = list(iter_named_tensors(params))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", FORMAT_VERSION))
        handle.write(struct.pack("<I", len(tensors)))
        for name, array in tensors:
            write_tensor(handle, name, array)

    return len(tensors)


def write_tensor(handle, name: str, array: np.ndarray) -> None:
    data = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    if data.ndim == 0:
        data = data.reshape(1)

    encoded_name = name.encode("utf-8")
    handle.write(struct.pack("<I", len(encoded_name)))
    handle.write(encoded_name)
    handle.write(struct.pack("<I", data.ndim))
    for dim in data.shape:
        handle.write(struct.pack("<I", int(dim)))
    handle.write(data.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GPT-2 weights for the C# runtime.")
    parser.add_argument("--model-size", default="124M", choices=MODEL_SIZES)
    parser.add_argument("--models-dir", default="models", help="Directory with downloaded GPT-2 checkpoints.")
    parser.add_argument("--output", required=True, help="Output path for the binary weight file.")

    args = parser.parse_args()
    _, hparams, params = load_encoder_hparams_and_params(args.model_size, args.models_dir)
    tensor_count = export_gpt2_weights(params, args.output)
    print(f"Exported {tensor_count} tensors to {args.output}")


if __name__ == "__main__":
    main()
