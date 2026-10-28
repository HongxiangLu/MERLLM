"""CASME3: full-clip native Qwen2.5-VL, AU-description -> emotion SFT.

Standalone replacement for the uploaded two-frame LAQ/action-token script.
Python >= 3.10; tested APIs: transformers 4.51.3, peft 0.15.2.
Targets are natural-language action descriptions followed by the emotion.
AU annotations are NEVER included in the inference prompt.
"""
import argparse
import contextlib
import gc
import json
import os
import random
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The original external models.Text dictionary was not attached.
# Built-in wording; --au-description-json can supply the original dictionary.
# action_units_nouns = {
#     1: "inner brow raising", 2: "outer brow raising", 4: "brow lowering",
#     5: "upper lid raising", 6: "cheek raising", 7: "lid tightening",
#     9: "nose wrinkling", 10: "upper lip raising", 11: "nasolabial furrow deepening",
#     12: "lip corner pulling", 13: "sharp lip pulling", 14: "dimpling",
#     15: "lip corner depressing", 16: "lower lip depressing", 17: "chin raising",
#     18: "lip puckering", 20: "lip stretching", 22: "lip funneling",
#     23: "lip tightening", 24: "lip pressing", 25: "lips parting",
#     26: "jaw dropping", 27: "mouth stretching", 28: "lip sucking",
#     41: "lid drooping", 42: "eye slit narrowing", 43: "eye closure",
#     44: "squinting", 45: "blinking", 46: "winking",
# }

action_units_nouns = {
    1: "raised inner eyebrows",
    2: "raised outer eyebrows",
    4: "lowered eyebrows",
    5: "raised upper eyelids",
    6: "raised cheeks",
    7: "tightened eyelids",
    9: "wrinkled nose",
    10: "raised upper lip",
    12: "a smile",  # pulled lip corners
    14: "tightened lip corners",
    15: "lowered lip corners",  # 有15不看12
    16: "lowered lower lip",  # opening the mouth
    17: "raised chin",
    18: "contracted lips",
    20: "stretched lips",
    23: "tightened lips",
    24: "pressed lips",
    25: "slightly opened mouth",
    26: "dropped jaw",
    28: "sucked-in lips",
    100: "raised eyebrows",
    43: "closed eyes",
    38: "dilated nostrils",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/home/yicheng/CASME3_ME_all/")
    p.add_argument("--label-file", default="cas(me)3_part_A_ME_label_JpgIndex_v1.xls")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--output-dir", default="./checkpoints/casme3_qwen25vl_au_sft")
    p.add_argument("--loso", default="CASME3/spNO.1")
    p.add_argument("--run-all-loso", action="store_true")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-frames", type=int, default=12)
    p.add_argument("--sampled-fps", type=float, default=2.0,
                   help="Virtual time scale of sampled frames, not the original camera FPS.")
    p.add_argument("--au-description-json", default=None)
    p.add_argument("--max-train-iters", type=int, default=6000,
                   help="Number of microbatches, as in the original script; <=0 means one data pass.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr-lora", type=float, default=2e-5)
    p.add_argument("--lr-vision", type=float, default=1e-5)
    p.add_argument("--lr-merger", type=float, default=5e-5)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--freeze-vision-encoder", action="store_true",
                   help="Freeze native vision blocks; the native merger still trains.")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Quantize language layers only; requires CUDA and bitsandbytes.")
    p.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--best-metric", choices=["acc", "uf1", "uar"], default="acc")
    p.add_argument("--no-balanced-sampling", dest="balanced_sampling", action="store_false")
    p.set_defaults(balanced_sampling=True)
    p.add_argument("--class-loss-power", type=float, default=1.0)
    p.add_argument("--class-token-loss-weight", type=float, default=2.0)
    p.add_argument("--adam-eps", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--skip-nonfinite-batches", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nan-stop-patience", type=int, default=3)
    p.add_argument("--empty-cache-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every-iters", type=int, default=200)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--max-eval-batches", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--repetition-penalty", type=float, default=1.08)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    p.add_argument("--loso-shard-rank", type=int, default=int(os.environ.get("LOSO_SHARD_RANK", 0)))
    p.add_argument("--loso-shard-world-size", type=int, default=int(os.environ.get("LOSO_SHARD_WORLD_SIZE", 1)))
    p.add_argument("--gpu-ids", default="")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--perfect-val-acc-stop", type=float, default=1.0)
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--checkpoint-dir", default=None,
                   help="A NEW native-video checkpoint for --eval-only, not an old LAQ checkpoint.")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.num_frames < 2 or a.num_frames % 2 or a.image_size < 28 or a.image_size % 28:
        p.error("num-frames must be even and >=2; image-size must be a positive multiple of 28.")
    if min(a.batch_size, a.grad_accum_steps, a.lora_rank, a.max_new_tokens) < 1:
        p.error("Batch size, accumulation steps, rank and max-new-tokens must be positive.")
    if a.eval_batch_size is None:
        a.eval_batch_size = a.batch_size
    if a.eval_batch_size < 1 or a.num_workers < 0 or a.sampled_fps <= 0:
        p.error("Invalid evaluation batch size, workers or FPS.")
    if a.class_token_loss_weight <= 0 or a.max_grad_norm <= 0:
        p.error("class-token-loss-weight and max-grad-norm must be positive.")
    if bool(a.checkpoint_dir) != bool(a.eval_only):
        p.error("Use --checkpoint-dir together with --eval-only.")
    if a.eval_only and a.run_all_loso:
        p.error("Evaluate a checkpoint on its own --loso subject, not --run-all-loso.")
    return a


CASME3_LOSO = [
    "CASME3/spNO.1", "CASME3/spNO.10", "CASME3/spNO.11", "CASME3/spNO.12",
    "CASME3/spNO.13", "CASME3/spNO.138", "CASME3/spNO.139", "CASME3/spNO.14",
    "CASME3/spNO.142", "CASME3/spNO.143", "CASME3/spNO.144", "CASME3/spNO.145",
    "CASME3/spNO.146", "CASME3/spNO.147", "CASME3/spNO.148", "CASME3/spNO.149",
    "CASME3/spNO.15", "CASME3/spNO.150", "CASME3/spNO.151", "CASME3/spNO.152",
    "CASME3/spNO.153", "CASME3/spNO.154", "CASME3/spNO.155", "CASME3/spNO.156",
    "CASME3/spNO.157", "CASME3/spNO.158", "CASME3/spNO.159", "CASME3/spNO.160",
    "CASME3/spNO.161", "CASME3/spNO.162", "CASME3/spNO.163", "CASME3/spNO.165",
    "CASME3/spNO.166", "CASME3/spNO.167", "CASME3/spNO.168", "CASME3/spNO.169",
    "CASME3/spNO.17", "CASME3/spNO.170", "CASME3/spNO.171", "CASME3/spNO.172",
    "CASME3/spNO.173", "CASME3/spNO.174", "CASME3/spNO.175", "CASME3/spNO.176",
    "CASME3/spNO.177", "CASME3/spNO.178", "CASME3/spNO.179", "CASME3/spNO.180",
    "CASME3/spNO.181", "CASME3/spNO.182", "CASME3/spNO.183", "CASME3/spNO.184",
    "CASME3/spNO.185", "CASME3/spNO.186", "CASME3/spNO.187", "CASME3/spNO.188",
    "CASME3/spNO.189", "CASME3/spNO.190", "CASME3/spNO.192", "CASME3/spNO.193",
    "CASME3/spNO.194", "CASME3/spNO.195", "CASME3/spNO.196", "CASME3/spNO.197",
    "CASME3/spNO.198", "CASME3/spNO.2", "CASME3/spNO.200", "CASME3/spNO.201",
    "CASME3/spNO.202", "CASME3/spNO.203", "CASME3/spNO.204", "CASME3/spNO.206",
    "CASME3/spNO.207", "CASME3/spNO.208", "CASME3/spNO.209", "CASME3/spNO.210",
    "CASME3/spNO.211", "CASME3/spNO.212", "CASME3/spNO.213", "CASME3/spNO.214",
    "CASME3/spNO.215", "CASME3/spNO.216", "CASME3/spNO.217", "CASME3/spNO.3",
    "CASME3/spNO.39", "CASME3/spNO.4", "CASME3/spNO.40", "CASME3/spNO.41",
    "CASME3/spNO.42", "CASME3/spNO.5", "CASME3/spNO.6", "CASME3/spNO.7",
    "CASME3/spNO.77", "CASME3/spNO.8", "CASME3/spNO.9",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_to_emotion(label_all):
    if label_all == "happy":
        return 0, "positive"
    if label_all == "surprise":
        return 1, "surprise"
    if label_all in ["disgust", "anger", "fear", "sad"]:
        return 2, "negative"
    return None, None


def au_to_description(label_au):
    numbers = []
    for part in str(label_au).split("+"):
        match = re.search(r"\d+", part)
        if match:
            num = int(match.group())
            if num in action_units_nouns:
                numbers.append(num)

    numbers = sorted(set(numbers))
    descriptions = []
    if 1 in numbers and 2 in numbers:
        descriptions.append("brow raising")
        numbers = [n for n in numbers if n not in [1, 2]]
    if 25 in numbers and 26 in numbers:
        descriptions.append("mouth opening")
        numbers = [n for n in numbers if n not in [25, 26]]
    for num in numbers:
        descriptions.append(action_units_nouns[num])
    if not descriptions:
        descriptions.append("subtle facial muscle movements")
    return " and ".join(descriptions)


def parse_emotion_prediction(text):
    text = text.lower()
    if "surprise" in text or "surprised" in text:
        return 1
    if "positive" in text or "happy" in text or "happiness" in text:
        return 0
    if (
        "negative" in text
        or "disgust" in text
        or "anger" in text
        or "angry" in text
        or "fear" in text
        or "sad" in text
        or "sadness" in text
    ):
        return 2
    return -1


def build_answer_from_template(au_text, emotion, randomize=True):
    templates = [
        f"The facial actions include {au_text}. Therefore, the micro-expression category is {emotion}.",
        f"Observed facial actions include {au_text}. This pattern corresponds to a {emotion} micro-expression.",
        f"The face shows {au_text}. These facial movements indicate a {emotion} micro-expression.",
        # f"The visible facial action is {au_text}. The most likely micro-expression category is {emotion}.",
        # f"There is evidence of {au_text}. Based on these actions, the micro-expression is {emotion}.",
        # f"Facial muscle activity includes {au_text}. This expression should be classified as {emotion}.",
    ]
    if randomize:
        return random.choice(templates)
    return templates[0]


def compute_metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.float64)
    class_cm = cm[:3, :3]
    total = cm[:3, :].sum()
    correct = np.trace(class_cm)
    acc = correct / max(total, 1.0)

    recalls = []
    f1s = []
    for cls_idx in range(3):
        tp = class_cm[cls_idx, cls_idx]
        row_sum = cm[cls_idx, :].sum()
        col_sum = cm[:3, cls_idx].sum()
        recall = tp / row_sum if row_sum > 0 else 0.0
        precision = tp / col_sum if col_sum > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        recalls.append(recall)
        f1s.append(f1)

    return {
        "acc": float(acc),
        "uf1": float(np.mean(f1s)),
        "uar": float(np.mean(recalls)),
        "per_class_f1": [float(x) for x in f1s],
        "per_class_recall": [float(x) for x in recalls],
        "total": int(total),
        "correct": int(correct),
    }


def save_eval_result(output_dir, loso_name, checkpoint_label, result):
    result_dir = os.path.join(output_dir, loso_name.replace("/", "_"), "best_validation")
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, "confusion_matrix_4x4.npy"), result["confusion_matrix"])
    np.savetxt(
        os.path.join(result_dir, "confusion_matrix_4x4.csv"),
        result["confusion_matrix"],
        fmt="%d",
        delimiter=",",
    )
    with open(os.path.join(result_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for item in result["samples"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "loso": loso_name,
        "best_iter": checkpoint_label,
        "metrics": result["metrics"],
        "unknown": result["unknown"],
        "num_samples": len(result["targets"]),
        "confusion_matrix_labels": ["positive", "surprise", "negative", "unknown_pred"],
    }
    with open(os.path.join(result_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def aggregate_saved_results(output_dir):
    total_cm = np.zeros((4, 4), dtype=np.int64)
    loso_results = []
    for name in sorted(os.listdir(output_dir)):
        result_dir = os.path.join(output_dir, name, "best_validation")
        cm_path = os.path.join(result_dir, "confusion_matrix_4x4.npy")
        summary_path = os.path.join(result_dir, "summary.json")
        if not os.path.exists(cm_path):
            continue

        cm = np.load(cm_path).astype(np.int64)
        total_cm += cm
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            loso_results.append(summary)
        else:
            loso_results.append({
                "loso": name,
                "best_iter": None,
                "metrics": compute_metrics_from_confusion(cm),
                "unknown": int(cm[:3, 3].sum()),
                "num_samples": int(cm[:3, :].sum()),
            })

    total_metrics = compute_metrics_from_confusion(total_cm)
    aggregate_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)
    np.save(os.path.join(aggregate_dir, "total_confusion_matrix_4x4.npy"), total_cm)
    np.savetxt(
        os.path.join(aggregate_dir, "total_confusion_matrix_4x4.csv"),
        total_cm,
        fmt="%d",
        delimiter=",",
    )
    aggregate_summary = {
        "metrics_from_total_confusion_matrix": total_metrics,
        "confusion_matrix_labels": ["positive", "surprise", "negative", "unknown_pred"],
        "total_confusion_matrix": total_cm.tolist(),
        "loso_results": loso_results,
    }
    with open(os.path.join(aggregate_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2, ensure_ascii=False)

    print("Total confusion_matrix_4x4 rows=true cols=pred [positive,surprise,negative,unknown]:")
    print(total_cm)
    print(
        f"Overall from saved best CMs: ACC={total_metrics['acc'] * 100:.3f}% "
        f"UF1={total_metrics['uf1'] * 100:.3f}% UAR={total_metrics['uar'] * 100:.3f}% "
        f"correct={total_metrics['correct']}/{total_metrics['total']}"
    )
    return aggregate_summary


def parse_gpu_ids(gpu_ids):
    if not gpu_ids:
        return []
    return [item.strip() for item in gpu_ids.split(",") if item.strip()]


def launch_gpu_workers(args):
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if len(gpu_ids) <= 1:
        return False
    if not args.run_all_loso:
        raise ValueError("--gpu-ids multi-worker launch is intended for --run-all-loso.")

    base_cmd = [sys.executable, os.path.abspath(__file__)]
    skip_next = False
    passthrough = []
    skip_options = {
        "--gpu-ids",
        "--loso-shard-rank",
        "--loso-shard-world-size",
        "--device",
    }
    flag_options = {"--worker", "--aggregate-only"}
    for idx, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg in skip_options:
            skip_next = True
            continue
        if any(arg.startswith(option + "=") for option in skip_options):
            continue
        if arg in flag_options:
            continue
        passthrough.append(arg)

    processes = []
    world_size = len(gpu_ids)
    for rank, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["LOSO_SHARD_RANK"] = str(rank)
        env["LOSO_SHARD_WORLD_SIZE"] = str(world_size)
        cmd = base_cmd + passthrough + [
            "--worker",
            "--device", "cuda:0",
            "--loso-shard-rank", str(rank),
            "--loso-shard-world-size", str(world_size),
        ]
        print(f"Launching worker {rank}/{world_size} on GPU {gpu_id}: {' '.join(cmd)}")
        processes.append(subprocess.Popen(cmd, env=env))

    failed = []
    for rank, process in enumerate(processes):
        return_code = process.wait()
        if return_code != 0:
            failed.append((rank, return_code))

    if failed:
        raise RuntimeError(f"GPU worker(s) failed: {failed}")

    aggregate_saved_results(args.output_dir)
    return True


def compute_class_weights(labels, num_classes=3, power=1.0):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    weights = safe_counts.sum() / (num_classes * safe_counts)
    weights = np.power(weights, power)
    weights = weights / max(weights.mean(), 1e-8)
    return counts.astype(np.int64), weights.astype(np.float32)


def build_train_sampler(dataset, args):
    labels = dataset.get_labels()
    if args.max_train_iters > 0:
        num_samples = args.max_train_iters * args.batch_size
    else:
        num_samples = len(dataset)

    if args.balanced_sampling:
        _, class_weights = compute_class_weights(labels, power=args.class_loss_power)
        sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.double)
        return data.WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

    if args.max_train_iters > 0:
        return data.RandomSampler(dataset, replacement=True, num_samples=num_samples)

    return None


def move_to_device(inputs, device):
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def find_subsequence_positions(sequence, pattern):
    if not pattern or len(pattern) > len(sequence):
        return []
    matches = []
    width = len(pattern)
    for start in range(0, len(sequence) - width + 1):
        if sequence[start:start + width] == pattern:
            matches.extend(range(start, start + width))
    return matches


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def normalize_subject(value):
    text = str(value).strip()
    match = re.search(r"spNO\.(\d+)", text, flags=re.I)
    if match:
        return f"spNO.{int(match.group(1))}"
    number = float(text)
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"Invalid subject {value!r}")
    return f"spNO.{int(number)}"


@lru_cache(maxsize=8192)
def sample_clip_paths(directory, onset, offset, count):
    files = {}
    for path in Path(directory).iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        numbers = re.findall(r"\d+", path.stem)
        if not numbers:
            continue
        index = int(numbers[-1])
        if onset <= index <= offset:
            if index in files:
                raise ValueError(f"Duplicate frame {index} in {directory}")
            files[index] = str(path)
    if not files:
        raise FileNotFoundError(f"No frames in [{onset},{offset}]: {directory}")
    indices = np.array(sorted(files))
    targets = np.linspace(onset, offset, count)
    right = np.searchsorted(indices, targets).clip(0, len(indices) - 1)
    left = (right - 1).clip(0, len(indices) - 1)
    nearest = np.where(abs(indices[left] - targets) <= abs(indices[right] - targets), left, right)
    return tuple(files[int(indices[i])] for i in nearest)


class CASME3Dataset(data.Dataset):
    def __init__(self, data_root, label_file, phase, num_loso, image_size, num_frames=16):
        self.phase, self.image_size, self.num_frames = phase, image_size, num_frames
        df = pd.read_csv(label_file) if str(label_file).lower().endswith(".csv") else pd.read_excel(label_file)
        if df.shape[1] < 8:
            raise ValueError("Expected at least 8 annotation columns and a header row.")
        held_out = normalize_subject(num_loso)
        self.samples = []
        for index, row in df.iterrows():
            if row.isna().all():
                continue
            try:
                subject = normalize_subject(row.iloc[0])
                if (subject != held_out) != (phase == "train"):
                    continue
                onset_raw, offset_raw = float(row.iloc[2]), float(row.iloc[4])
                if not onset_raw.is_integer() or not offset_raw.is_integer():
                    raise ValueError("Non-integer onset/offset")
                onset, offset = int(onset_raw), int(offset_raw)
                if onset == 0:
                    continue
                if onset < 0 or offset < onset:
                    raise ValueError("Invalid onset/offset")
                label_id, emotion = label_to_emotion(str(row.iloc[7]).strip().lower())
                if label_id is None:
                    continue
                value = row.iloc[1]
                clip = str(int(value)) if isinstance(value, (float, np.floating)) and float(value).is_integer() else str(value).strip()
                clip = clip.replace("\\", "/")
                root = Path(data_root)
                candidates = [root / subject / clip, root / "CASME3" / subject / clip, root / clip]
                directory = next((p for p in candidates if p.is_dir()), None)
                if directory is None:
                    raise FileNotFoundError(f"Clip directory not found: {candidates}")
                paths = sample_clip_paths(str(directory.resolve()), onset, offset, num_frames)
                self.samples.append(dict(paths=paths, label=label_id, emotion=emotion,
                    au_text=au_to_description(row.iloc[5]), sample_id=f"{subject}/{clip}"))
            except Exception as exc:
                raise ValueError(f"Annotation row {index + 2}: {exc}") from exc

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [int(sample["label"]) for sample in self.samples]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []
        for path in sample["paths"]:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                frames.append(np.array(image, dtype=np.uint8))
        return dict(video=np.stack(frames), label=sample["label"], emotion=sample["emotion"],
                    sample_id=sample["sample_id"], answer=build_answer_from_template(
                        sample["au_text"], sample["emotion"], randomize=self.phase == "train"))


def collate_fn(batch):
    return dict(videos=[item["video"] for item in batch],
                labels=torch.tensor([item["label"] for item in batch], dtype=torch.long),
                emotions=[item["emotion"] for item in batch],
                answers=[item["answer"] for item in batch],
                sample_ids=[item["sample_id"] for item in batch])


def build_instruction():
    return (
        "A complete facial micro-expression clip is provided as sampled video frames. "
        "Describe the facial action. These facial actions correspond to which micro-expression? "
        "Choose one category: positive, negative, or surprise. "
        "First describe the observed facial movements, then state the final category."
    )


def make_messages(answer=None):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": "already_sampled_clip"},
        {"type": "text", "text": build_instruction()},
    ]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def get_qwenvl_model(model):
    return model.get_base_model() if isinstance(model, PeftModel) else model


def get_embedding_device(model):
    return get_qwenvl_model(model).get_input_embeddings().weight.device


def build_sft_batch(batch, processor, model, args, include_answers=True):
    prompt = processor.apply_chat_template(make_messages(), tokenize=False, add_generation_prompt=True)
    prompts = [prompt] * len(batch["videos"])
    common = dict(videos=batch["videos"], fps=args.sampled_fps, padding=True,
                  return_tensors="pt", do_resize=False, input_data_format="channels_last")
    # Native processor expands video placeholders BEFORE prompt lengths are measured.
    prompt_inputs = processor(text=prompts, **common)
    device = get_embedding_device(model)
    if not include_answers:
        return move_to_device(prompt_inputs, device), None, None
    texts = [processor.apply_chat_template(make_messages(answer), tokenize=False,
                                          add_generation_prompt=False) for answer in batch["answers"]]
    inputs = processor(text=texts, **common)
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    labels = torch.full_like(ids, -100)
    weights = torch.ones_like(ids, dtype=torch.float32)
    for b in range(len(texts)):
        prompt_ids = prompt_inputs["input_ids"][b][prompt_inputs["attention_mask"][b].bool()]
        positions = mask[b].nonzero(as_tuple=True)[0]
        sequence = ids[b, positions]
        length = len(prompt_ids)
        if length >= len(sequence) or not torch.equal(sequence[:length], prompt_ids):
            raise ValueError("Assistant boundary mismatch; check tokenizer/chat template versions.")
        answer_positions = positions[length:]
        labels[b, answer_positions] = ids[b, answer_positions]
        weight = float(args.runtime_class_loss_weights[int(batch["labels"][b])])
        weights[b, answer_positions] = weight
        # Match both bare and whitespace-prefixed class tokens (BPE tokenization differs).
        answer_ids = sequence[length:].tolist()
        class_positions = set()
        for word in (batch["emotions"][b], " " + batch["emotions"][b]):
            pattern = processor.tokenizer.encode(word, add_special_tokens=False)
            class_positions.update(find_subsequence_positions(answer_ids, pattern))
        if not class_positions:
            raise ValueError("Emotion token not found in the target answer.")
        for relative in class_positions:
            weights[b, answer_positions[relative]] *= args.class_token_loss_weight
    return move_to_device(inputs, device), labels.to(device), weights.to(device)


def weighted_lm_loss(logits, labels, loss_weights):
    valid = labels[:, 1:].ne(-100)
    if not valid.any():
        raise ValueError("No supervised answer tokens.")
    # Supports packed answer-only logits and ordinary full-sequence logits.
    selected = logits if logits.ndim == 2 else logits[:, :-1, :][valid]
    targets = labels[:, 1:][valid]
    weights = loss_weights[:, 1:][valid].float()
    losses = nn.functional.cross_entropy(selected.float(), targets, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def forward_sft_loss(model, inputs, labels, weights):
    valid = labels[:, 1:].ne(-100)
    # Project only the hidden states that predict supervised answer tokens.
    # Exactly the same answer CE; avoids full-vocabulary logits for video/prompt tokens.
    def select_answer_states(_module, values):
        return (values[0][:, :-1, :][valid],) + values[1:]
    handle = get_qwenvl_model(model).get_output_embeddings().register_forward_pre_hook(select_answer_states)
    try:
        output = model(**inputs, use_cache=False, return_dict=True)
        return weighted_lm_loss(output.logits, labels, weights)
    finally:
        # Generation must always use the unmodified native LM head.
        handle.remove()


def compute_dtype(args):
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]


def amp_context(args):
    if args.precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch.device(args.device).type, dtype=compute_dtype(args))


def build_model_and_processor(args):
    source = args.checkpoint_dir or args.model_name
    processor = AutoProcessor.from_pretrained(source, revision=args.revision)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    kwargs = dict(revision=args.revision, torch_dtype=compute_dtype(args),
                  attn_implementation="sdpa", low_cpu_mem_usage=True)
    if args.load_in_4bit:
        if torch.device(args.device).type != "cuda":
            raise ValueError("4-bit loading requires CUDA in this script.")
        kwargs.update(device_map={"": args.device}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(args), llm_int8_skip_modules=["visual", "lm_head"]))
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, **kwargs)
    base.config.use_cache = False
    if args.load_in_4bit:
        if any("4bit" in type(m).__name__.lower() for m in base.visual.modules()):
            raise RuntimeError("The native visual encoder must not be quantized for full fine-tuning.")
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    if args.checkpoint_dir:
        model = PeftModel.from_pretrained(base, args.checkpoint_dir, is_trainable=False)
    else:
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", target_modules=r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"))
    visual = get_qwenvl_model(model).visual
    for p in visual.parameters():
        p.requires_grad_(not args.freeze_vision_encoder and not args.eval_only)
    for p in visual.merger.parameters():
        p.requires_grad_(not args.eval_only)
    # Keep all saved visual weights FP32 on reload, including evaluation.
    if args.checkpoint_dir:
        payload = torch.load(Path(args.checkpoint_dir) / "visual.pt", map_location="cpu", weights_only=True)
        if payload["scope"] == "full":
            visual.float().load_state_dict(payload["state_dict"], strict=True)
        else:
            visual.merger.float().load_state_dict(payload["state_dict"], strict=True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if not args.load_in_4bit:
        model.to(args.device)
    if args.gradient_checkpointing and not args.eval_only:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    return model, processor


def make_optimizer(model, args):
    visual = get_qwenvl_model(model).visual
    merger = [p for p in visual.merger.parameters() if p.requires_grad]
    merger_ids = {id(p) for p in merger}
    vision = [p for p in visual.parameters() if p.requires_grad and id(p) not in merger_ids]
    visual_ids = {id(p) for p in visual.parameters()}
    lora = [p for p in model.parameters() if p.requires_grad and id(p) not in visual_ids]
    groups = [{"params": lora, "lr": args.lr_lora}, {"params": merger, "lr": args.lr_merger}]
    if vision:
        groups.append({"params": vision, "lr": args.lr_vision})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay, eps=args.adam_eps)


def save_checkpoint(model, processor, args, loso_name, checkpoint_label):
    path = Path(args.output_dir) / loso_name.replace("/", "_") / str(checkpoint_label)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    visual = get_qwenvl_model(model).visual
    scope = "merger" if args.freeze_vision_encoder else "full"
    module = visual.merger if scope == "merger" else visual
    torch.save(dict(scope=scope, state_dict={k: v.detach().cpu() for k, v in module.state_dict().items()}), path / "visual.pt")
    metadata = dict(args=vars(args), loso=loso_name, checkpoint_label=checkpoint_label,
                    instruction=build_instruction(), au_descriptions=action_units_nouns,
                    base_commit=getattr(get_qwenvl_model(model).config, "_commit_hash", None))
    (path / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_one_epoch(loader, model, processor, optimizer, args, eval_callback=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.device(args.device).type == "cuda" and args.precision == "fp16")
    params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    total_iters = min(len(loader), args.max_train_iters) if args.max_train_iters > 0 else len(loader)
    total_loss, total_steps, skipped, failures, pending = 0.0, 0, 0, 0, 0
    reason, last_iter = "completed", 0
    next_eval = args.eval_every_iters if args.eval_every_iters > 0 else None
    for step, batch in enumerate(loader, 1):
        if step > total_iters:
            break
        last_iter = step
        loss = inputs = labels = weights = None
        try:
            inputs, labels, weights = build_sft_batch(batch, processor, model, args)
            with amp_context(args):
                loss = forward_sft_loss(model, inputs, labels, weights)
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError("Nonfinite SFT loss")
            scaler.scale(loss / args.grad_accum_steps).backward()
            pending += 1
            value = loss.item()
            del loss, inputs, labels, weights
            loss = inputs = labels = weights = None
            did_step = pending == args.grad_accum_steps or step == total_iters
            if did_step:
                scaler.unscale_(optimizer)
                # Correctly average an incomplete final accumulation window.
                if pending < args.grad_accum_steps:
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(args.grad_accum_steps / pending)
                norm = nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                if not torch.isfinite(norm):
                    scaler.update()
                    raise FloatingPointError("Nonfinite gradient norm")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            failures = 0
            total_loss += value
            total_steps += 1
            if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
                print(f"iter={step}/{total_iters} loss={total_loss/total_steps:.4f} sft={value:.4f} skipped={skipped}", flush=True)
            if args.empty_cache_every > 0 and step % args.empty_cache_every == 0:
                cleanup_cuda()
            if eval_callback and next_eval is not None and did_step and step >= next_eval:
                try:
                    stop = eval_callback(step, total_loss / total_steps)
                finally:
                    model.train()
                while next_eval <= step:
                    next_eval += args.eval_every_iters
                if stop:
                    reason = "metric_early_stop"
                    break
        except (FloatingPointError, torch.OutOfMemoryError) as exc:
            if not args.skip_nonfinite_batches:
                raise
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            skipped += 1
            failures += 1
            loss = inputs = labels = weights = None
            print(f"iter={step} skipped: {type(exc).__name__}: {exc}", flush=True)
            cleanup_cuda()
            if args.nan_stop_patience > 0 and failures >= args.nan_stop_patience:
                reason = "unstable"
                break
    optimizer.zero_grad(set_to_none=True)
    if total_steps == 0:
        raise RuntimeError("No successful training batches; reduce memory usage or check the data.")
    return total_loss / total_steps, reason, last_iter


@torch.no_grad()
def evaluate(loader, model, processor, args, split_name="val"):
    model.eval()
    preds, targets, samples = [], [], []
    for step, batch in enumerate(loader):
        if args.max_eval_batches > 0 and step >= args.max_eval_batches:
            break
        inputs, _, _ = build_sft_batch(batch, processor, model, args, include_answers=False)
        with amp_context(args):
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, use_cache=True, repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                pad_token_id=processor.tokenizer.pad_token_id)
        input_len = inputs["input_ids"].shape[1]
        texts = processor.tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)
        batch_preds = [parse_emotion_prediction(text) for text in texts]
        batch_targets = batch["labels"].tolist()
        preds.extend(batch_preds)
        targets.extend(batch_targets)
        for sample_id, gt, pred, answer, text in zip(batch["sample_ids"], batch_targets, batch_preds, batch["answers"], texts):
            samples.append(dict(sample_id=sample_id, target=int(gt), prediction=int(pred), answer=answer, generated=text))
            if len(samples) <= 3:
                print(f"[{split_name}] GT={gt} PRED={pred}\nGT  : {answer}\nPRED: {text}\n", flush=True)
    if not targets:
        raise ValueError("Empty evaluation set.")
    safe_preds = [pred if pred >= 0 else 3 for pred in preds]
    cm = confusion_matrix(targets, safe_preds, labels=[0, 1, 2, 3])
    metrics = compute_metrics_from_confusion(cm)
    known_pairs = [(p, t) for p, t in zip(preds, targets) if p >= 0]
    known_acc = accuracy_score([t for _, t in known_pairs], [p for p, _ in known_pairs]) if known_pairs else 0.0
    unknown = sum(p < 0 for p in preds)
    print(f"{split_name}: acc={metrics['acc']*100:.3f}% known_acc={known_acc*100:.3f}% "
          f"UF1={metrics['uf1']*100:.3f}% UAR={metrics['uar']*100:.3f}% unknown={unknown}/{len(preds)}")
    print("confusion_matrix_4x4 [positive,surprise,negative,unknown]:\n", cm)
    return dict(metrics=metrics, confusion_matrix=cm.astype(np.int64), preds=preds,
                targets=targets, samples=samples, unknown=int(unknown))


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def run_one_loso(args, loso_name):
    set_seed(args.seed)
    train_ds = CASME3Dataset(args.data_root, args.label_file, "train", loso_name, args.image_size, args.num_frames)
    val_ds = CASME3Dataset(args.data_root, args.label_file, "test", loso_name, args.image_size, args.num_frames)
    if not len(val_ds) or (not args.eval_only and not len(train_ds)):
        raise ValueError(f"Empty LOSO split: {loso_name}; train={len(train_ds)} val={len(val_ds)}")
    print(f"Subject {loso_name}: train={len(train_ds)} val={len(val_ds)}", flush=True)
    counts, weights = compute_class_weights(train_ds.get_labels(), power=args.class_loss_power)
    args.runtime_class_loss_weights = weights.tolist()
    print("Train counts:", counts.tolist(), "class weights:", weights.tolist(), "balanced sampling:", args.balanced_sampling)
    common = dict(num_workers=args.num_workers, pin_memory=torch.device(args.device).type == "cuda",
                  collate_fn=collate_fn, worker_init_fn=seed_worker)
    val_loader = data.DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, **common)
    model, processor = build_model_and_processor(args)
    if args.eval_only:
        result = evaluate(val_loader, model, processor, args)
        save_eval_result(args.output_dir, loso_name, "eval_only", result)
        del model, processor
        cleanup_cuda()
        return dict(loso=loso_name, best_iter="eval_only", **result)
    sampler = build_train_sampler(train_ds, args)
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None,
                                   sampler=sampler, **common)
    optimizer = make_optimizer(model, args)
    best_result, best_step, bad_evals, last_eval = None, -1, 0, -1
    log_path = Path(args.output_dir) / loso_name.replace("/", "_") / "evaluations.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    def run_eval_at_iter(global_iter, train_loss):
        nonlocal best_result, best_step, bad_evals, last_eval
        result = evaluate(val_loader, model, processor, args)
        last_eval = global_iter
        improved = best_result is None or result["metrics"][args.best_metric] > best_result["metrics"][args.best_metric]
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(iteration=global_iter, train_loss=train_loss,
                                         selected=improved, **result["metrics"])) + "\n")
        if improved:
            best_result, best_step, bad_evals = result, global_iter, 0
            label = f"iter_{global_iter}"
            if args.save_checkpoints:
                save_checkpoint(model, processor, args, loso_name, label)
            save_eval_result(args.output_dir, loso_name, label, result)
            print(f"new best: {loso_name} iter={global_iter} {args.best_metric}={result['metrics'][args.best_metric]:.6f}")
            return args.perfect_val_acc_stop > 0 and result["metrics"]["acc"] >= args.perfect_val_acc_stop
        bad_evals += 1
        return args.early_stop_patience > 0 and bad_evals >= args.early_stop_patience

    train_loss, reason, last_iter = train_one_epoch(train_loader, model, processor, optimizer, args, run_eval_at_iter)
    if reason == "unstable" and best_result is None:
        raise RuntimeError(f"{loso_name}: unstable training before any usable evaluation.")
    if reason == "completed" and last_eval != last_iter:
        run_eval_at_iter(last_iter, train_loss)
    print(f"Finished {loso_name}: stop={reason} best_iter={best_step}")
    result = dict(loso=loso_name, best_iter=best_step, metrics=best_result["metrics"],
                  confusion_matrix=best_result["confusion_matrix"], unknown=best_result["unknown"],
                  num_samples=len(best_result["targets"]))
    del model, processor, optimizer, train_loader, val_loader
    cleanup_cuda()
    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.aggregate_only:
        aggregate_saved_results(args.output_dir)
        return
    if not args.worker and launch_gpu_workers(args):
        return
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if len(gpu_ids) == 1 and not args.worker:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        args.device = "cuda:0"
    if args.checkpoint_dir:
        metadata = json.loads((Path(args.checkpoint_dir) / "run_config.json").read_text(encoding="utf-8"))
        # Restore model/input settings while retaining current data and output paths.
        for key in ("model_name", "revision", "num_frames", "image_size", "sampled_fps", "freeze_vision_encoder"):
            setattr(args, key, metadata["args"][key])
        args.revision = metadata.get("base_commit") or args.revision
        args.loso = metadata["loso"]
        action_units_nouns.clear()
        action_units_nouns.update({int(k): v for k, v in metadata["au_descriptions"].items()})
    elif args.au_description_json:
        mapping = json.loads(Path(args.au_description_json).read_text(encoding="utf-8"))
        action_units_nouns.clear()
        action_units_nouns.update({int(k): str(v) for k, v in mapping.items()})
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Use --device cpu for a small diagnostic run.")
        torch.cuda.set_device(device)
    if args.precision == "auto":
        args.precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "fp32"
    if device.type != "cuda" and args.precision == "fp16":
        raise ValueError("Use fp32 or bf16 on CPU.")
    set_seed(args.seed)
    loso_list = CASME3_LOSO if args.run_all_loso else ["CASME3/" + normalize_subject(args.loso)]
    if args.loso_shard_world_size < 1 or not 0 <= args.loso_shard_rank < args.loso_shard_world_size:
        raise ValueError("Invalid LOSO shard configuration.")
    loso_list = loso_list[args.loso_shard_rank::args.loso_shard_world_size]
    all_results = [run_one_loso(args, name) for name in loso_list]
    if not args.worker:
        aggregate_saved_results(args.output_dir)
    if len(all_results) > 1:
        for name in ("acc", "uf1", "uar"):
            print(f"Mean fold {name}: {100*np.mean([r['metrics'][name] for r in all_results]):.3f}%")


if __name__ == "__main__":
    main()
