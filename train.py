"""基于 Qwen2.5-VL 的 CASME3 微表情动作描述与情绪监督微调（SFT）。

处理流程：
    1. 读取 CSV/Excel 标注，按留一被试法（LOSO）划分训练与评估数据。
    2. 在 onset～offset 内选择原始帧号最接近均匀目标位置的裁剪图像。
    3. 将 AU 转为英文动作描述，与三类情绪组合成监督答案。
    4. 使用原生视频输入训练语言 LoRA，以及配置允许的视觉参数。
    5. 生成文本并解析情绪，保存每折结果、检查点及跨折汇总。

数据集提供 AU 和情绪标注，英文答案由本脚本的词典和模板生成。
推理提示不包含真实 AU 或情绪；输入是帧图像而不是视频文件。
本脚本保留 onset=0 跳过规则，与裁剪脚本的起始帧调整策略相互独立。
Python >= 3.10；原脚本注明使用过的 API 版本为 transformers 4.51.3、peft 0.15.2。
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

# 原外部 models.Text 词典未随脚本提供。
# 默认使用下方内置描述，可通过 --au-description-json 替换为自定义词典。
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
    12: "a smile",  # AU12 表示嘴角上提，此处使用简化的英文描述。
    14: "tightened lip corners",
    15: "lowered lip corners",  # AU15 表示嘴角下拉；当前代码不据此排除 AU12。
    16: "lowered lower lip",  # AU16 表示下唇下拉。
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
    """解析路径、模型、训练、评估和多 GPU 调度参数，并校验参数组合。"""
    p = argparse.ArgumentParser(description=__doc__)
    # 输入输出及基础模型：model-name 可为远程仓库 ID 或完整的本地模型目录。
    p.add_argument("--data-root", default="/home/yicheng/CASME3_ME_all/")
    p.add_argument("--label-file", default="cas(me)3_part_A_ME_label_JpgIndex_v1.xls")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--output-dir", default="./checkpoints/casme3_qwen25vl_au_sft")
    p.add_argument("--loso", default="CASME3/spNO.1")
    p.add_argument("--run-all-loso", action="store_true")
    # 视频输入：空间尺寸须为 28 的倍数，帧数须为不小于 2 的偶数。
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-frames", type=int, default=12)
    p.add_argument("--sampled-fps", type=float, default=2.0,
                   help="Virtual time scale of sampled frames, not the original camera FPS.")
    p.add_argument("--au-description-json", default=None)
    # 训练预算按微批次计；有效累计批量通常为 batch-size × grad-accum-steps。
    p.add_argument("--max-train-iters", type=int, default=6000,
                   help="Number of microbatches, as in the original script; <=0 means one data pass.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    # 不同可训练模块使用不同学习率；LoRA 的 rank、alpha 和 dropout 控制适配器。
    p.add_argument("--lr-lora", type=float, default=2e-5)
    p.add_argument("--lr-vision", type=float, default=1e-5)
    p.add_argument("--lr-merger", type=float, default=5e-5)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # 显存配置会改变训练方案：4-bit 作用于语言层，视觉主体冻结不包含 merger。
    p.add_argument("--freeze-vision-encoder", action="store_true",
                   help="Freeze native vision blocks; the native merger still trains.")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Quantize language layers only; requires CUDA and bitsandbytes.")
    p.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--best-metric", choices=["acc", "uf1", "uar"], default="acc")
    # 类别不平衡处理包含采样权重及回答损失权重，两者默认同时启用。
    p.add_argument("--no-balanced-sampling", dest="balanced_sampling", action="store_false")
    p.set_defaults(balanced_sampling=True)
    p.add_argument("--class-loss-power", type=float, default=1.0)
    p.add_argument("--class-token-loss-weight", type=float, default=2.0)
    p.add_argument("--adam-eps", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # 此开关也控制训练循环中的 OOM；诊断时可用 --no-skip-nonfinite-batches 直接报错。
    p.add_argument("--skip-nonfinite-batches", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nan-stop-patience", type=int, default=3)
    p.add_argument("--empty-cache-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    # 评估在完成梯度累计更新后触发；max-eval-batches=0 表示不限制评估批数。
    p.add_argument("--eval-every-iters", type=int, default=200)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--max-eval-batches", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--repetition-penalty", type=float, default=1.08)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    # 多 GPU 分片的是被试折列表，每个工作进程独立训练，不合并显存。
    p.add_argument("--loso-shard-rank", type=int, default=int(os.environ.get("LOSO_SHARD_RANK", 0)))
    p.add_argument("--loso-shard-world-size", type=int, default=int(os.environ.get("LOSO_SHARD_WORLD_SIZE", 1)))
    p.add_argument("--gpu-ids", default="")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--perfect-val-acc-stop", type=float, default=1.0)
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    # 检查点路径仅支持配合 eval-only 评估，不提供训练状态恢复。
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


# 预设的留出被试列表；不自动排除情绪筛选后无有效样本的被试。
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
    """设置 Python、NumPy 和 PyTorch 随机种子，降低重复实验的随机差异。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_to_emotion(label_all):
    """将六种原始情绪映射为三类；其他标签返回 (None, None)。
    
    类别编号固定为 0=positive、1=surprise、2=negative。"""
    if label_all == "happy":
        return 0, "positive"
    if label_all == "surprise":
        return 1, "surprise"
    if label_all in ["disgust", "anger", "fear", "sad"]:
        return 2, "negative"
    return None, None


def au_to_description(label_au):
    """将 AU 标注转换为英文动作描述，作为监督答案的一部分。
    
    按加号拆分并提取数字，忽略词典中不存在的 AU；不保留左右侧或强度信息。
    组合 AU1+2 和 AU25+26 使用合并描述；无已知 AU 时使用通用描述。"""
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
    """从生成文本中按关键词解析三类情绪，无法识别时返回 -1。
    
    检查优先级为 surprise、positive、negative；这是子串匹配，
    不具备否定或多类别冲突消解能力，也不是独立的分类输出层。"""
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
    """由动作描述和情绪构造英文监督答案。
    
    训练时随机选择模板；评估时固定使用第一个模板作为参考答案。"""
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
    """从 4×4 混淆矩阵计算准确率、宏平均 F1（UF1）和宏平均召回率（UAR）。
    
    行是真实类别，列是预测类别，顺序为 positive、surprise、negative、unknown。
    unknown 预测计入错误和召回率分母；真实标签仅有前三类。
    没有真实样本的类别召回率记为 0，宏平均仍固定对三类求平均。"""
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
    """保存一折的混淆矩阵、逐样本预测及指标摘要。
    
    best_validation 目录中的同名结果文件会覆盖，表示当前选中的评估结果。"""
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
    """累加输出目录中各折已保存的混淆矩阵，再计算总体指标。
    
    这里使用目录中已有的结果，不保证所有预定折都已完成；
    先合并混淆矩阵得到的指标与各折指标的简单平均并不等价。"""
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
    """将逗号分隔的 GPU 编号解析成列表；空字符串表示不启用显式调度。"""
    if not gpu_ids:
        return []
    return [item.strip() for item in gpu_ids.split(",") if item.strip()]


def launch_gpu_workers(args):
    """每张指定 GPU 启动一个独立进程，分配不同 LOSO 折并等待结束。
    
    这是折之间的并行，不是 DDP 或模型切分；每张卡都需容纳完整的一折模型。
    返回 True 表示父进程已完成调度及汇总，无需继续本进程的训练流程。"""
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
    """按类别频数的倒数构造权重，再取幂并归一化到均值为 1。
    
    空类别用频数 1 防止除零；power=0 时所有权重相同。"""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    weights = safe_counts.sum() / (num_classes * safe_counts)
    weights = np.power(weights, power)
    weights = weights / max(weights.mean(), 1e-8)
    return counts.astype(np.int64), weights.astype(np.float32)


def build_train_sampler(dataset, args):
    """根据迭代预算和类别权重构造训练采样器。
    
    指定正的 max_train_iters 时抽取“迭代数×批量大小”个样本，允许重复。
    均衡采样始终有放回；即使预算不为正，也不保证每条样本恰好出现一次。"""
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
    """将输入字典中的顶层张量移到指定设备，其他值原样保留。"""
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def find_subsequence_positions(sequence, pattern):
    """返回序列中所有匹配子序列覆盖的位置，供情绪 token 加权使用。"""
    if not pattern or len(pattern) > len(sequence):
        return []
    matches = []
    width = len(pattern)
    for start in range(0, len(sequence) - width + 1):
        if sequence[start:start + width] == pattern:
            matches.extend(range(start, start + width))
    return matches


def cleanup_cuda():
    """触发垃圾回收并释放 CUDA 缓存中的空闲块，不释放仍被引用的张量。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def normalize_subject(value):
    """将带前缀的被试名或整数编号统一为 spNO.N。"""
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
    """在标注闭区间内均匀设置目标位置，并选择最近的实际图像帧。
    
    Args:
        directory: 直接存放裁剪图像的片段目录，不含 color 子目录。
        onset: 起始帧编号，包含该端点。
        offset: 结束帧编号，包含该端点。
        count: 需要返回的帧数，允许重复选择同一图像。
    
    Returns:
        按目标时间位置排列的路径元组，长度为 count。
    
    Raises:
        FileNotFoundError: 目录不存在或区间内没有可用图像。
        ValueError: 不同图像文件被解析为相同帧号。
    
    文件名必须保留原始帧号；本函数缓存结果，运行中修改目录不会自动刷新缓存。"""
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
    # 目标位置仍以标注帧号定义，不按“现有文件的列表位置”均分。
    indices = np.array(sorted(files))
    targets = np.linspace(onset, offset, count)
    right = np.searchsorted(indices, targets).clip(0, len(indices) - 1)
    left = (right - 1).clip(0, len(indices) - 1)
    # 在左右候选中选最近者；等距选左侧，越界目标选端点，允许重复采样。
    nearest = np.where(abs(indices[left] - targets) <= abs(indices[right] - targets), left, right)
    return tuple(files[int(indices[i])] for i in nearest)


class CASME3Dataset(data.Dataset):
    """读取 CASME3 标注及裁剪图像，为一折训练或评估构造样本。
    
    输入是已经提取并裁剪的帧图像，不在此处解码视频或检测人脸。
    按表格位置读取：被试、片段、onset、apex、offset、AU、客观类别、情绪；
    apex 和客观类别不参与当前采样和目标构造。"""
    def __init__(self, data_root, label_file, phase, num_loso, image_size, num_frames=16):
        """按留出被试划分数据，筛选标签并预先确定每条标注的采样路径。"""
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
                # train 仅保留非留出被试，其他 phase 仅保留留出被试。
                if (subject != held_out) != (phase == "train"):
                    continue
                onset_raw, offset_raw = float(row.iloc[2]), float(row.iloc[4])
                if not onset_raw.is_integer() or not offset_raw.is_integer():
                    raise ValueError("Non-integer onset/offset")
                onset, offset = int(onset_raw), int(offset_raw)
                # 保留训练侧既有筛选规则，不因裁剪侧改为从第 1 帧开始而自动改变。
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
                # 兼容三种目录布局；片段目录下应直接存放裁剪图像。
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
        """返回筛选后的标注样本数，不是图像总数。"""
        return len(self.samples)

    def get_labels(self):
        """返回类别编号列表，供类别权重和均衡采样器使用。"""
        return [int(sample["label"]) for sample in self.samples]

    def __getitem__(self, idx):
        """读取采样帧并构造监督答案。
        
        video 为 uint8 数组，形状是 (帧数, 高, 宽, 3)，颜色通道为 RGB。
        此处只缩放图像，模型需要的归一化及视觉 patch 组织由 processor 完成。"""
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
    """整理批次，视频保留为数组列表，类别标签转换为 long 张量。"""
    return dict(videos=[item["video"] for item in batch],
                labels=torch.tensor([item["label"] for item in batch], dtype=torch.long),
                emotions=[item["emotion"] for item in batch],
                answers=[item["answer"] for item in batch],
                sample_ids=[item["sample_id"] for item in batch])


def build_instruction():
    """构造只含任务要求的英文提示词，不包含该样本的 AU 或真实情绪。"""
    return (
        "A complete facial micro-expression clip is provided as sampled video frames. "
        "Describe the facial action. These facial actions correspond to which micro-expression? "
        "Choose one category: positive, negative, or surprise. "
        "First describe the observed facial movements, then state the final category."
    )


def make_messages(answer=None):
    """构造原生视频聊天消息；训练时追加答案，推理时仅保留用户消息。
    
    video 字段是模板占位标记；真实帧数组通过 processor 的 videos 参数传入。"""
    messages = [{"role": "user", "content": [
        {"type": "video", "video": "already_sampled_clip"},
        {"type": "text", "text": build_instruction()},
    ]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def get_qwenvl_model(model):
    """从 PEFT 包装中取出基础 Qwen 模型；未包装时直接返回模型。"""
    return model.get_base_model() if isinstance(model, PeftModel) else model


def get_embedding_device(model):
    """取得输入词嵌入所在设备，用于放置模型输入张量。"""
    return get_qwenvl_model(model).get_input_embeddings().weight.device


def build_sft_batch(batch, processor, model, args, include_answers=True):
    """将帧数组和聊天文本转换为模型输入及回答区间的监督标签。
    
    Args:
        batch: collate_fn 返回的视频、答案、情绪和类别信息。
        processor: Qwen 原生多模态处理器。
        model: 用于确定输入设备的模型。
        args: 输入帧率、类别权重和情绪 token 权重等配置。
        include_answers: 为 False 时仅构造推理提示，不输入真实答案。
    
    Returns:
        (inputs, labels, weights)。训练时 labels 和 weights 形状均为 (B, L)，
        仅回答位置参与损失；推理时后两项为 None。B 为批量大小，L 为填充后长度。"""
    prompt = processor.apply_chat_template(make_messages(), tokenize=False, add_generation_prompt=True)
    prompts = [prompt] * len(batch["videos"])
    common = dict(videos=batch["videos"], fps=args.sampled_fps, padding=True,
                  return_tensors="pt", do_resize=False, input_data_format="channels_last")
    # 先由原生处理器展开视频占位 token，再计算提示长度，避免视觉 token 数量造成边界偏移。
    prompt_inputs = processor(text=prompts, **common)
    device = get_embedding_device(model)
    if not include_answers:
        return move_to_device(prompt_inputs, device), None, None
    texts = [processor.apply_chat_template(make_messages(answer), tokenize=False,
                                          add_generation_prompt=False) for answer in batch["answers"]]
    inputs = processor(text=texts, **common)
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    # -100 表示忽略位置：填充、用户提示和视频占位都不作为预测目标。
    labels = torch.full_like(ids, -100)
    weights = torch.ones_like(ids, dtype=torch.float32)
    for b in range(len(texts)):
        prompt_ids = prompt_inputs["input_ids"][b][prompt_inputs["attention_mask"][b].bool()]
        # 通过 attention_mask 去除左侧填充，比较完整文本与纯提示的 token 前缀。
        positions = mask[b].nonzero(as_tuple=True)[0]
        sequence = ids[b, positions]
        length = len(prompt_ids)
        if length >= len(sequence) or not torch.equal(sequence[:length], prompt_ids):
            raise ValueError("Assistant boundary mismatch; check tokenizer/chat template versions.")
        answer_positions = positions[length:]
        labels[b, answer_positions] = ids[b, answer_positions]
        weight = float(args.runtime_class_loss_weights[int(batch["labels"][b])])
        # 回答先赋予类别权重，再对情绪词覆盖的 token 乘以额外权重。
        weights[b, answer_positions] = weight
        # 同时匹配带前导空格和不带空格的情绪编码，兼容不同的子词切分结果。
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
    """计算回答 token 的加权因果语言模型交叉熵。
    
    前一位置预测后一 token，因此标签向右移一位进行对齐。
    损失按有效 token 权重总和归一化；单样本批次的统一类别权重会在分子分母抵消。"""
    valid = labels[:, 1:].ne(-100)
    if not valid.any():
        raise ValueError("No supervised answer tokens.")
    # 兼容仅含回答位置的二维 logits，以及包含完整序列的三维 logits。
    selected = logits if logits.ndim == 2 else logits[:, :-1, :][valid]
    targets = labels[:, 1:][valid]
    weights = loss_weights[:, 1:][valid].float()
    losses = nn.functional.cross_entropy(selected.float(), targets, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def forward_sft_loss(model, inputs, labels, weights):
    """仅对预测监督答案所需的隐藏状态执行词表投影，减少 logits 显存。"""
    valid = labels[:, 1:].ne(-100)
    # 仅投影有效回答 token 对应的前一位置隐藏状态。
    # 保留相同的回答交叉熵目标，避免为视频和提示位置分配完整词表 logits。
    def select_answer_states(_module, values):
        """在 LM head 前截取有效预测位置，输出形状从 (B, L, H) 变为 (N, H)。"""
        return (values[0][:, :-1, :][valid],) + values[1:]
    handle = get_qwenvl_model(model).get_output_embeddings().register_forward_pre_hook(select_answer_states)
    try:
        output = model(**inputs, use_cache=False, return_dict=True)
        return weighted_lm_loss(output.logits, labels, weights)
    finally:
        # 无论前向是否成功，都移除钩子，确保后续生成使用完整的原生 LM head。
        handle.remove()


def compute_dtype(args):
    """将已解析的精度名称转换为 torch.dtype；auto 在 main 中预先解析。"""
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]


def amp_context(args):
    """创建自动混合精度上下文；FP32 模式使用空上下文。"""
    if args.precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch.device(args.device).type, dtype=compute_dtype(args))


def build_model_and_processor(args):
    """加载基础模型和处理器，配置量化、LoRA、视觉参数及梯度检查点。
    
    默认训练语言注意力的 LoRA、视觉主体和 merger；冻结视觉主体时 merger 仍训练。
    评估检查点时另行恢复适配器和视觉权重，基础模型仍从 model_name 加载。"""
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
        # NF4 双重量化；显式排除视觉模块和输出头，并在加载后检查视觉模块。
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
    # merger 连接视觉表示与语言表示；训练模式下即使冻结视觉主体也更新它。
    for p in visual.merger.parameters():
        p.requires_grad_(not args.eval_only)
    # 恢复视觉权重时使用 FP32，评估模式也保持这一加载规则。
    if args.checkpoint_dir:
        payload = torch.load(Path(args.checkpoint_dir) / "visual.pt", map_location="cpu", weights_only=True)
        if payload["scope"] == "full":
            visual.float().load_state_dict(payload["state_dict"], strict=True)
        else:
            visual.merger.float().load_state_dict(payload["state_dict"], strict=True)
    # 可训练权重保持 FP32，前向计算仍可由 autocast 使用低精度。
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
    """将 LoRA、视觉 merger 和未冻结的视觉主体分组，使用各自学习率。"""
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
    """保存 PEFT 适配器、处理器、视觉权重和运行配置。
    
    不保存优化器、梯度缩放器或完整训练状态，因此不是断点续训快照；
    重新加载时还需要原始基础模型。冻结视觉主体时仅另存 merger 权重。"""
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
    """执行一次 loader 遍历，包含梯度累计、周期评估和异常处理。
    
    loader 可由有放回采样器扩展，因此函数名中的 epoch 不一定是数据集的一遍。
    迭代数按微批次计，优化器更新数受 grad_accum_steps 影响。
    返回成功微批次的平均损失、停止原因和最后尝试的迭代编号。"""
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
            # 按累计步数缩放损失，使多个微批次的梯度形成平均值。
            scaler.scale(loss / args.grad_accum_steps).backward()
            pending += 1
            value = loss.item()
            del loss, inputs, labels, weights
            loss = inputs = labels = weights = None
            did_step = pending == args.grad_accum_steps or step == total_iters
            if did_step:
                scaler.unscale_(optimizer)
                # 最后一个累计窗口若不足设定步数，按实际微批次数校正梯度平均值。
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
            # 只在优化器更新后评估，避免在未完成的累计窗口中切换训练/评估。
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
            # 丢弃当前累计窗口的梯度，防止异常结果污染后续参数更新。
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
    """在无梯度模式下生成答案，解析情绪并保存逐样本结果。
    
    真实答案仅用于结果对照，不传入推理提示。max_eval_batches>0 时只评估部分批次。"""
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
        # generate 返回提示和续写的拼接结果，只解码新生成的答案。
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
    # 无法解析的预测进入第 4 列；主指标将其计错，known_acc 仅是可解析子集指标。
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
    """根据 DataLoader 分配的 PyTorch 种子初始化子进程的 Python/NumPy 随机数。"""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def run_one_loso(args, loso_name):
    """为一个留出被试重新构建数据集和模型，完成训练或仅评估。
    
    当前将留出被试同时用于周期评估及最佳检查点选择，
    所得最佳分数并非独立于模型选择的最终测试结果。"""
    set_seed(args.seed)
    train_ds = CASME3Dataset(args.data_root, args.label_file, "train", loso_name, args.image_size, args.num_frames)
    val_ds = CASME3Dataset(args.data_root, args.label_file, "test", loso_name, args.image_size, args.num_frames)
    if not len(val_ds) or (not args.eval_only and not len(train_ds)):
        raise ValueError(f"Empty LOSO split: {loso_name}; train={len(train_ds)} val={len(val_ds)}")
    print(f"Subject {loso_name}: train={len(train_ds)} val={len(val_ds)}", flush=True)
    # 类别权重只由训练被试计算，避免使用留出被试的标签分布。
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
        """评估当前权重，记录指标，在严格改善时保存结果并判断是否提前停止。"""
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
    """解析配置，选择设备，分配 LOSO 折并组织训练或结果汇总。"""
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
        # 从检查点恢复模型及输入设置；数据路径和输出路径仍使用本次命令行配置。
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
    # 按步长分配不同折，每个工作进程领取互不重叠的被试子列表。
    loso_list = loso_list[args.loso_shard_rank::args.loso_shard_world_size]
    all_results = [run_one_loso(args, name) for name in loso_list]
    if not args.worker:
        aggregate_saved_results(args.output_dir)
    if len(all_results) > 1:
        for name in ("acc", "uf1", "uar"):
            print(f"Mean fold {name}: {100*np.mean([r['metrics'][name] for r in all_results]):.3f}%")


if __name__ == "__main__":
    main()
