# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


LANDMARK_PROMPTS = {
    1: "the frontmost visible tip of the fish snout or closed mouth",
    2: "the point on the dorsal body outline at the fish nape, immediately behind the head",
    3: "the anterior attachment point where the dorsal fin begins on the fish body",
    4: "the posterior attachment point where the dorsal fin ends on the fish body",
    5: "the upper attachment point where the caudal fin meets the caudal peduncle",
    6: "the midpoint at the posterior end of the caudal peduncle, immediately before the caudal fin",
    7: "the lower attachment point where the caudal fin meets the caudal peduncle",
    8: "the anterior attachment point where the anal fin begins on the fish body",
    9: "the anterior attachment point where the pelvic fin begins on the fish body",
    10: "the base where the pectoral fin attaches to the fish body, just behind the gill cover",
    11: "the posterior visible margin of the fish operculum or gill cover",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a deterministic LocateAnything-3B zero-shot evaluation protocol."
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("data/benchmark_v1/tables/benchmark_samples.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/protocols/locateanything_3b_zero_shot"),
    )
    parser.add_argument("--per-source-group", type=int, default=3)
    args = parser.parse_args()

    frame = pd.read_csv(args.samples)
    eligible = frame[
        (frame["standard_split"] == "test")
        & frame["technical_validation_eligible"].astype(bool)
        & frame["origin_label"].isin(["farmed", "wild"])
    ].copy()
    eligible = eligible.sort_values(["source_groups", "benchmark_id"], kind="stable")

    selected = (
        eligible.groupby("source_groups", sort=True, group_keys=False)
        .head(args.per_source_group)
        .copy()
    )
    selected["selection_rank_within_source_group"] = selected.groupby(
        "source_groups", sort=True
    ).cumcount() + 1
    selected["image_relative_path"] = selected["benchmark_id"].map(
        lambda value: f"data/benchmark_v1/images/original/{value}.jpg"
    )

    columns = [
        "benchmark_id",
        "source_groups",
        "taxon_candidate",
        "origin_label",
        "standard_split",
        "selection_rank_within_source_group",
        "image_relative_path",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    selected[columns].to_csv(args.output / "sample_manifest.csv", index=False)

    protocol = {
        "protocol_name": "locateanything_3b_zero_shot_original_image",
        "prepared_date": "2026-08-25",
        "model_id": "nvidia/LocateAnything-3B",
        "model_release_date": "2026-05-26",
        "model_license": "NVIDIA License for non-commercial academic research",
        "representation": "unaligned original images",
        "data_partition": "locked standard test only",
        "selection": {
            "rule": "lexicographically first benchmark IDs within each source_groups value",
            "per_source_group": args.per_source_group,
            "selected_images": int(len(selected)),
            "source_group_values": int(selected["source_groups"].nunique()),
            "excluded_origin_labels": ["conflicted"],
            "purpose": "bounded zero-shot semantic-grounding stress test; not model selection",
        },
        "prompt_template": "Point to: {landmark_description} on the single fish.",
        "landmark_prompts": {str(key): value for key, value in LANDMARK_PROMPTS.items()},
        "inference": {
            "generation_mode": "hybrid",
            "temperature": 0.0,
            "top_p": None,
            "repetition_penalty": 1.1,
            "max_new_tokens": 128,
            "precision": "bfloat16",
            "batch_size": 1,
            "coordinate_parser": "first <box><x><y></box> point; integer coordinates scaled from [0,1000]",
        },
        "metrics": {
            "normalizer": "Euclidean distance between target roles 1 and 6 in the original image",
            "missing_output": "counted as incorrect for all-output PCK and excluded from detected-only NME",
            "reporting": "per-query output rate, per-role NME/PCK, and complete-11-point image rate",
            "uncertainty": "2,000 percentile bootstrap resamples clustered by image; queries within an image stay together",
        },
        "claim_boundary": (
            "The model is evaluated without task-specific training. Prompt semantics inherit the "
            "project ontology's provisional status. Results assess zero-shot text-conditioned "
            "localisation and do not validate anatomical truth or establish a supervised SOTA claim."
        ),
        "redistribution": "model weights are excluded from all project release and submission archives",
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
