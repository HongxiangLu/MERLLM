#!/usr/bin/env bash
# 在一张 GPU 上逐折运行全部有效被试；每折重新加载基础模型。
# 使用当前已激活的训练环境，DATA_ROOT、LABEL_FILE 必须指向服务器实际数据。
set -euo pipefail

export WORK="${WORK:-/data02/hxlu/MERLLM}"
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-$WORK/code/train.py}"
export MODEL_DIR="${MODEL_DIR:-$WORK/models/Qwen2.5-VL-7B-Instruct}"
export DATA_ROOT="${DATA_ROOT:?请先设置 DATA_ROOT 为裁剪图像根目录}"
export LABEL_FILE="${LABEL_FILE:?请先设置 LABEL_FILE 为 v1 标注文件路径}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export RUN_DIR="${RUN_DIR:-$WORK/checkpoints/full_loso_$(date +%Y%m%d_%H%M%S)}"

# 每次实验使用独立目录，避免旧结果混入汇总；本启动脚本不提供断点恢复。
if [[ -e "$RUN_DIR" ]]; then
    echo "输出目录已存在，请指定新的 RUN_DIR：$RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR/logs"
cp "$TRAIN_SCRIPT" "$RUN_DIR/train_snapshot.py"
python -m pip freeze > "$RUN_DIR/environment.txt"
echo "本次结果目录：$RUN_DIR"

# 复用训练脚本的数据筛选和采样逻辑；虚拟留出被试 0 使 train 包含全部真实被试。
# 此步骤不加载模型，会提前发现目录缺失或整个区间没有可用帧的问题。
python - <<'PY' | tee "$RUN_DIR/logs/preflight.log"
import importlib.util
import os
from collections import Counter
from pathlib import Path

spec = importlib.util.spec_from_file_location("casme3_train", os.environ["TRAIN_SCRIPT"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
dataset = module.CASME3Dataset(
    os.environ["DATA_ROOT"], os.environ["LABEL_FILE"],
    phase="train", num_loso="spNO.0", image_size=224, num_frames=12,
)
counts = Counter(sample["sample_id"].split("/")[0] for sample in dataset.samples)
if len(counts) < 2:
    raise RuntimeError("有效被试不足两名，无法进行 LOSO。")
subjects = sorted(counts, key=lambda name: int(name.split(".")[-1]))
Path(os.environ["RUN_DIR"], "subjects.txt").write_text(
    "\n".join(subjects) + "\n", encoding="utf-8"
)
print(f"有效样本：{len(dataset)}；有效被试/折数：{len(subjects)}")
for subject in subjects:
    print(f"{subject}: 评估 {counts[subject]}，训练 {len(dataset) - counts[subject]}")
PY

TRAIN_ARGS=(
    --data-root "$DATA_ROOT"
    --label-file "$LABEL_FILE"
    --model-name "$MODEL_DIR"
    --output-dir "$RUN_DIR"
    --load-in-4bit
    --batch-size 1 --eval-batch-size 1 --grad-accum-steps 16
    --num-workers 0
    --no-skip-nonfinite-batches
    --early-stop-patience 0 --perfect-val-acc-stop 0
)

while IFS= read -r subject; do
    echo "$(date -Is) 开始：$subject"
    python -u "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}" --loso "$subject" \
        2>&1 | tee "$RUN_DIR/logs/$subject.log"
    # 返回码为 0 还不足以证明跑满预算，额外确认本折正常完成。
    grep -Fq "Finished CASME3/$subject: stop=completed" "$RUN_DIR/logs/$subject.log"
    printf '%s\n' "$subject" >> "$RUN_DIR/completed_subjects.txt"
done < "$RUN_DIR/subjects.txt"

python -u "$TRAIN_SCRIPT" --aggregate-only --output-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/logs/aggregate.log"
echo "$(date -Is) 全部有效折已完成：$RUN_DIR"