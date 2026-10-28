"""根据 CASME3 Part A 微表情标注，批量裁剪图像中的人脸区域。

处理流程：
    1. 读取 v1 XLS 标签表的第一个工作表，逐行获取被试、片段及帧区间。
    2. 在每个微表情区间的 onset 帧上使用 MTCNN 检测人脸。
    3. 使用同一检测框逐帧裁剪 onset 到 offset 的全部可用帧（包含两个端点）。
    4. 每帧裁剪后立即保存并释放图像资源，按「被试/片段/帧编号.jpg」组织结果。
    5. 向终端和输出目录中的日志文件写入异常明细与本次运行汇总。

输入目录结构：原始数据根目录/被试编号/片段编号的小写形式/color/帧编号.jpg。
输出目录结构：/home/yicheng/CASME3_ME_all/被试编号/片段编号/帧编号.jpg。
运行前需修改下方输入、标注和输出路径，并准备可用的 CUDA 环境。
"""

import logging
import os
from contextlib import closing
from datetime import datetime

import xlrd
from facenet_pytorch import MTCNN
from PIL import Image


def crop_safe(im, box):
    """将检测框限制在图像边界内，再返回裁剪后的图像。

    Args:
        im: 待裁剪的 PIL 图像。
        box: 检测框坐标 (x1, y1, x2, y2)，分别表示左上角和右下角。

    Returns:
        裁剪后的 PIL 图像；边界修正后宽度或高度非正时返回 None。
        主流程会记录无效裁剪框并报错终止。
    """
    w, h = im.size
    # 将浮点检测坐标转换为整数，并限制在有效图像范围内。
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return im.crop((x1, y1, x2, y2))


# 原始数据根目录和 v1 标签路径；XLS 文件由 xlrd 读取。
path = '/home/yicheng/CASME3/part_A/data/CASME3/part_A/data/part_A'
annotation_path = os.path.join(
    '/home/yicheng/CASME3/part_A/annotation',
    'cas(me)3_part_A_ME_label_JpgIndex_v1.xls',
)

# 日志逐条写入文件，方便在 SSH 终端查看，也便于运行后追溯。
output_root = '/home/yicheng/CASME3_ME_all'
os.makedirs(output_root, exist_ok=True)
log_path = os.path.join(output_root, f"crop_{datetime.now():%Y%m%d_%H%M%S_%f}.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.info("标注文件=%s；输出目录=%s；日志=%s", annotation_path, output_root, log_path)

# 在 GPU 上初始化人脸检测器；下方通过 detect 获取检测框。
mtcnn = MTCNN(keep_all=False, device="cuda")

workbook = xlrd.open_workbook(annotation_path)
sheet = workbook.sheet_by_index(0)
num_rows = sheet.nrows
# “成功”要求该区间所有预期帧都已保存；缺帧但完成保存的区间单独统计。
# 帧数仅统计本次已开始处理且 onset 非零的区间，不代表目录中的历史文件数。
stats = dict(success=0, partial=0, failed=0, skipped=0, expected=0, saved=0)
started = 0

# 第 0 行为表头；每条数据行对应一个带 onset/apex/offset 标注的区间。
try:
    for row_idx in range(1, num_rows):
        try:
            started += 1
            subject_id = session_id = "未知"
            frame_path = ""
            stage = "读取标注"
            row = sheet.row_values(row_idx)

            # 第 1、2 列为被试和片段编号；输入目录中的片段编号转为小写。
            subject_id = str(row[0])
            session_id = str(row[1])
            image_dir = os.path.join(path, subject_id, session_id.lower(), 'color')

            # 第 3～5 列为起始帧、峰值帧和结束帧，数值对应原始图像文件名。
            onset = int(row[2])
            apex = int(row[3])
            offset = int(row[4])
            if onset == 0:
                stats["skipped"] += 1
                logger.warning("跳过：onset=0；标注行=%d；被试=%s；片段=%s",
                               row_idx + 1, subject_id, session_id)
                continue
            expected = offset - onset + 1
            stats["expected"] += expected
            saved_before = stats["saved"]

            # 构造闭区间 [onset, offset] 内所有预期图像路径。
            img_path = []
            for i in range(onset, offset + 1):
                img_num = str(i) + '.jpg'
                img_path.append(os.path.join(image_dir, img_num))

            # 参考帧仅用于检测；离开 with 后释放图像及文件资源。
            stage = "读取 onset 图像（缺失或损坏时终止）"
            frame_path = img_path[0]
            with closing(Image.open(frame_path)) as img:
                img.load()
                img_mode = img.mode
                stage = "检测人脸"
                face1, _ = mtcnn.detect(img)

            if face1 is None:
                stats["skipped"] += 1
                logger.warning("未检测到人脸；标注行=%d；被试=%s；片段=%s；文件=%s",
                               row_idx + 1, subject_id, session_id, frame_path)
                continue

            # 整段复用参考帧的第一个检测框。
            box = face1[0]
            stage = "创建输出目录"
            frame_path = ""
            output_dir = os.path.join(output_root, subject_id, session_id)
            os.makedirs(output_dir, exist_ok=True)

            # 逐帧裁剪、保存并释放内存，不缓存整个区间的裁剪图。
            # 保留从 onset 连续编号的规则；缺帧时仍可能与原编号不一致。
            num = onset
            for next_img_path in img_path:
                frame_path = next_img_path
                if not os.path.exists(next_img_path):
                    logger.warning("缺失帧；标注行=%d；被试=%s；片段=%s；文件=%s",
                                   row_idx + 1, subject_id, session_id, frame_path)
                    continue

                stage = "读取区间图像（损坏时终止）"
                with closing(Image.open(next_img_path)) as next_img:
                    next_img.load()
                    stage = "裁剪图像"
                    cropped = crop_safe(next_img, box)
                    if cropped is None:
                        raise ValueError(f"无效裁剪框：{box}")
                    # 裁剪图和颜色转换结果也显式关闭，包括保存失败的情况。
                    with closing(cropped):
                        with closing(cropped.convert(img_mode)) as converted:
                            stage = "保存图像"
                            frame_path = os.path.join(output_dir, str(num) + '.jpg')
                            converted.save(frame_path)
                            stats["saved"] += 1
                num += 1

            saved = stats["saved"] - saved_before
            if saved == expected:
                stats["success"] += 1
            else:
                stats["partial"] += 1
                logger.warning("区间不完整；标注行=%d；被试=%s；片段=%s；预期=%d；保存=%d",
                               row_idx + 1, subject_id, session_id, expected, saved)
        except Exception:
            stats["failed"] += 1
            logger.exception("区间失败；标注行=%d；被试=%s；片段=%s；阶段=%s；文件=%s",
                             row_idx + 1, subject_id, session_id, stage, frame_path)
            raise
finally:
    # 即使处理中断，也输出截至当前的统计；不将未处理或未完成行算作成功。
    finished = stats["success"] + stats["partial"] + stats["failed"] + stats["skipped"]
    logger.info("汇总：标签总数=%d；成功区间=%d；不完整区间=%d；失败区间=%d；"
                "跳过区间=%d；未开始区间=%d；中断未完成区间=%d；"
                "已开始的非零 onset 区间预期帧数=%d；本次实际保存帧数=%d",
                num_rows - 1, stats["success"], stats["partial"], stats["failed"],
                stats["skipped"], num_rows - 1 - started, started - finished,
                stats["expected"], stats["saved"])
