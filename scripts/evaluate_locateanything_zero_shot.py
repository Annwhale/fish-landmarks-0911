# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

POINT_PATTERN = re.compile(r"<box><(\d+)><(\d+)></box>")


def parse_point(answer: str, width: int, height: int) -> tuple[float, float] | None:
    match = POINT_PATTERN.search(answer)
    if match is None:
        return None
    x, y = int(match.group(1)), int(match.group(2))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x / 1000.0 * width, y / 1000.0 * height


def prepare_inputs(processor, image: Image.Image, prompt: str, device: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = processor.process_vision_info(messages)
    return processor(
        text=[text], images=images, videos=videos, return_tensors="pt"
    ).to(device)


def cluster_bootstrap_mean_ci(
    frame: pd.DataFrame,
    value_column: str,
    cluster_column: str = "benchmark_id",
    seed: int = 20260825,
    replicates: int = 2000,
) -> dict[str, float | int]:
    clean = frame[[cluster_column, value_column]].copy()
    clean[value_column] = pd.to_numeric(clean[value_column], errors="coerce")
    clean = clean[np.isfinite(clean[value_column])]
    if clean.empty:
        return {"n": 0, "clusters": 0, "mean": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan")}
    cluster_values = {
        cluster: group[value_column].to_numpy(dtype=np.float64)
        for cluster, group in clean.groupby(cluster_column, sort=True)
    }
    clusters = sorted(cluster_values)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values = np.concatenate([cluster_values[cluster] for cluster in sampled])
        means[index] = values.mean()
    return {
        "n": int(len(clean)),
        "clusters": int(len(clusters)),
        "mean": float(clean[value_column].mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def summarize(frame: pd.DataFrame, protocol: dict) -> dict:
    available = frame[frame["output_available"]].copy()
    pck_columns = ["pck_0_02", "pck_0_05", "pck_0_10"]
    result = {
        "protocol_name": protocol["protocol_name"],
        "model_id": protocol["model_id"],
        "images": int(frame["benchmark_id"].nunique()),
        "queries": int(len(frame)),
        "outputs": int(frame["output_available"].sum()),
        "output_rate": float(frame["output_available"].mean()),
        "complete_11_point_images": int(
            frame.groupby("benchmark_id")["output_available"].all().sum()
        ),
        "complete_11_point_image_rate": float(
            frame.groupby("benchmark_id")["output_available"].all().mean()
        ),
        "nme_detected_only": cluster_bootstrap_mean_ci(
            available, "nme_role1_role6"
        ),
        "runtime_seconds": float(frame["runtime_seconds"].sum()),
        "per_role": {},
        "metric_definitions": protocol["metrics"],
        "claim_boundary": protocol["claim_boundary"],
    }
    for column in pck_columns:
        result[f"{column}_all_queries"] = cluster_bootstrap_mean_ci(frame, column)
        result[f"{column}_detected_only"] = cluster_bootstrap_mean_ci(available, column)
    for role_id, group in frame.groupby("role_id", sort=True):
        detected = group[group["output_available"]]
        result["per_role"][str(int(role_id))] = {
            "landmark_name": str(group["landmark_name"].iloc[0]),
            "queries": int(len(group)),
            "outputs": int(group["output_available"].sum()),
            "output_rate": float(group["output_available"].mean()),
            "nme_detected_only": cluster_bootstrap_mean_ci(
                detected, "nme_role1_role6"
            ),
            "pck_0_05_all_queries": cluster_bootstrap_mean_ci(group, "pck_0_05"),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LocateAnything-3B as a zero-shot text-conditioned fish landmark localiser."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/path/to/LocateAnything-3B"),
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("data/experiments/protocols/locateanything_3b_zero_shot"),
    )
    parser.add_argument(
        "--keypoints",
        type=Path,
        default=Path("data/benchmark_v1/tables/benchmark_keypoints.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/evaluation/locateanything_3b_zero_shot"),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Runtime feasibility subset only; omit for the frozen full protocol.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Recompute the summary from an existing query_predictions.csv without loading the model.",
    )
    args = parser.parse_args()

    protocol = json.loads((args.protocol_dir / "protocol.json").read_text(encoding="utf-8"))
    if args.summarize_only:
        result_frame = pd.read_csv(args.output / "query_predictions.csv")
        summary = summarize(result_frame, protocol)
        summary["runtime_subset_max_images"] = None
        existing_path = args.output / "summary.json"
        if existing_path.exists():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if "software" in existing:
                summary["software"] = existing["software"]
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    manifest = pd.read_csv(args.protocol_dir / "sample_manifest.csv")
    if args.max_images is not None:
        manifest = manifest.head(args.max_images).copy()
    target_frame = pd.read_csv(args.keypoints)
    target_lookup = {
        benchmark_id: group.sort_values("canonical_id")
        for benchmark_id, group in target_frame.groupby("benchmark_id", sort=False)
    }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(20260825)
    torch.cuda.manual_seed_all(20260825)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, fix_mistral_regex=True
    )
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, fix_mistral_regex=True
    )
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    rows: list[dict] = []
    prompt_map = {int(key): value for key, value in protocol["landmark_prompts"].items()}
    args.output.mkdir(parents=True, exist_ok=True)

    for image_index, sample in manifest.reset_index(drop=True).iterrows():
        benchmark_id = str(sample["benchmark_id"])
        image_path = Path(str(sample["image_relative_path"]))
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        target = target_lookup[benchmark_id]
        target_points = target[["x", "y"]].to_numpy(dtype=np.float64)
        normalizer = float(np.linalg.norm(target_points[0] - target_points[5]))

        for role_id, description in prompt_map.items():
            prompt = protocol["prompt_template"].format(landmark_description=description)
            inputs = prepare_inputs(processor, image, prompt, device)
            start = time.perf_counter()
            with torch.inference_mode():
                response = model.generate(
                    pixel_values=inputs["pixel_values"].to(dtype),
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws"),
                    tokenizer=tokenizer,
                    max_new_tokens=protocol["inference"]["max_new_tokens"],
                    use_cache=True,
                    generation_mode=protocol["inference"]["generation_mode"],
                    temperature=protocol["inference"]["temperature"],
                    repetition_penalty=protocol["inference"]["repetition_penalty"],
                    verbose=False,
                )
            runtime = time.perf_counter() - start
            point = parse_point(response, width, height)
            target_x, target_y = target_points[role_id - 1]
            if point is None:
                pred_x = pred_y = error_px = nme = np.nan
                output_available = False
            else:
                pred_x, pred_y = point
                error_px = float(np.hypot(pred_x - target_x, pred_y - target_y))
                nme = error_px / max(normalizer, 1e-12)
                output_available = True
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "source_groups": sample["source_groups"],
                    "taxon_candidate": sample["taxon_candidate"],
                    "origin_label": sample["origin_label"],
                    "role_id": role_id,
                    "landmark_name": str(target.iloc[role_id - 1]["canonical_name_provisional"]),
                    "prompt": prompt,
                    "target_x": target_x,
                    "target_y": target_y,
                    "prediction_x": pred_x,
                    "prediction_y": pred_y,
                    "output_available": output_available,
                    "error_px": error_px,
                    "nme_role1_role6": nme,
                    "pck_0_02": float(output_available and nme <= 0.02),
                    "pck_0_05": float(output_available and nme <= 0.05),
                    "pck_0_10": float(output_available and nme <= 0.10),
                    "runtime_seconds": runtime,
                    "raw_response": response,
                }
            )
        partial = pd.DataFrame(rows)
        partial.to_csv(args.output / "query_predictions.partial.csv", index=False)
        print(f"[{image_index + 1}/{len(manifest)}] {benchmark_id}", flush=True)

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(args.output / "query_predictions.csv", index=False)
    summary = summarize(result_frame, protocol)
    summary["runtime_subset_max_images"] = args.max_images
    summary["software"] = {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
