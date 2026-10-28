"""根据 CASME3 Part A 微表情标注，批量裁剪图像中的人脸区域。

处理流程：
    1. 读取 v1 XLS 标签表的第一个工作表，逐行获取被试、片段及帧区间。
    2. 在每个微表情区间的 onset 帧上使用 MTCNN 检测人脸。
    3. 使用同一检测框裁剪 onset 到 offset 的全部可用帧（包含两个端点）。
    4. 按「被试/片段/帧编号.jpg」的目录结构保存裁剪结果。

输入目录结构：原始数据根目录/被试编号/片段编号的小写形式/color/帧编号.jpg。
输出目录结构：/home/yicheng/CASME3_ME_all/被试编号/片段编号/帧编号.jpg。
运行前需修改下方输入、标注和输出路径，并准备可用的 CUDA 环境。
"""

import xlrd
from facenet_pytorch import MTCNN
from PIL import Image

import os


def crop_safe(im, box):
    """将检测框限制在图像边界内，再返回裁剪后的图像。

    Args:
        im: 待裁剪的 PIL 图像。
        box: 检测框坐标 (x1, y1, x2, y2)，分别表示左上角和右下角。

    Returns:
        裁剪后的 PIL 图像；边界修正后宽度或高度非正时返回 None。
        调用方应检查返回值；当前主流程尚未处理 None 的情况。
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

# 在 GPU 上初始化人脸检测器；下方通过 detect 获取检测框。
mtcnn = MTCNN(keep_all=False, device="cuda")

workbook = xlrd.open_workbook(annotation_path)
sheet = workbook.sheet_by_index(0)
num_rows = sheet.nrows
face_info = []
sublist = []

# 第 0 行为表头；每条数据行对应一个带 onset/apex/offset 标注的区间。
for row_idx in range(1, num_rows):
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
        continue

    # 构造闭区间 [onset, offset] 内所有预期图像路径。
    img_num = []
    img_path = []
    for i in range(onset, offset + 1):
        img_num = str(i) + '.jpg'
        img_path.append(os.path.join(image_dir, img_num))
    img1_num = str(onset) + '.jpg'

    flag = False

    # onset 帧作为整段的检测参考帧；此处要求文件存在且可以正常打开。
    img = Image.open(img_path[0])
    img_format = img.format
    img_mode = img.mode

    # detect 返回人脸框和检测概率，不直接返回裁剪图像；此处不使用概率。
    face1, _ = mtcnn.detect(img)
    next_imgs = []
    if face1 is not None:
        # 取返回的第一个检测框，并在整个区间复用，保持裁剪区域一致。
        box = face1[0]
        for next_img_path in img_path:
            # 跳过不存在的帧；当前实现不记录缺失清单。
            if not os.path.exists(next_img_path):
                continue
            next_img = Image.open(next_img_path)
            cropped_img2 = crop_safe(next_img, box)
            # 统一为参考帧的颜色模式，并保留其 format 属性。
            # 最终保存的编码格式仍由下方 .jpg 文件扩展名决定。
            cropped_img2 = cropped_img2.convert(img_mode)
            cropped_img2.format = img_format
            next_imgs.append(cropped_img2)
    else:
        # 参考帧没有检测到人脸时，跳过整个标注区间。
        continue

    # 输出目录保留标签表中的片段编号写法；同名文件保存时会被覆盖。
    if not os.path.exists(os.path.join('/home/yicheng/CASME3_ME_all', subject_id, session_id)):
        os.makedirs(os.path.join('/home/yicheng/CASME3_ME_all', subject_id, session_id))

    # 从 onset 连续编号；若中途缺帧，后续输出编号将与原始编号不一致。
    num = onset
    for cropped_next_img in next_imgs:
        save_path2 = os.path.join(
            '/home/yicheng/CASME3_ME_all', subject_id, session_id, str(num) + '.jpg'
        )
        cropped_next_img.save(save_path2)
        num += 1
