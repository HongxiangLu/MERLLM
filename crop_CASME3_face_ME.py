# path = os.path.join(path, 'samm_25', 'feature_25')

import re
import xlrd
from facenet_pytorch import MTCNN
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

from torchvision import transforms as T
import numpy as np
import os
import random
import openpyxl


def crop_safe(im, box):
    # box: (x1,y1,x2,y2)
    w, h = im.size
    x1, y1, x2, y2 = map(int, box)  # 用整数
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return im.crop((x1, y1, x2, y2))

path='/home/yicheng/CASME3/part_A/data/CASME3/part_A/data/part_A'
annotation_path = os.path.join('/home/yicheng/CASME3/part_A/annotation', 'cas(me)3_part_A_ME_label_JpgIndex_v1.xls')
# annotation_path = os.path.join('/home/yicheng/CASME3/part_A/annotation', 'cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx')

mtcnn = MTCNN(keep_all=False, device="cuda")

# 读取xls文件
workbook = xlrd.open_workbook(annotation_path)
# workbook = openpyxl.load_workbook(annotation_path)
sheet = workbook.sheet_by_index(0)  # 获取第一个 sheet
# 获取总行数
num_rows = sheet.nrows
face_info = []
sublist = []
for row_idx in range(1, num_rows):
    row = sheet.row_values(row_idx)

# workbook = openpyxl.load_workbook(annotation_path, data_only=True)
# # 获取第一个 sheet（注意 openpyxl 返回的是工作表对象，不是 index 访问）
# sheet = workbook.worksheets[0]
# # 获取总行数
# num_rows = sheet.max_row
# face_info = []
# sublist = []
# # openpyxl 的行索引从 1 开始
# for row_idx in range(2, num_rows + 1):  # 跳过第一行标题
#     row = [cell.value for cell in sheet[row_idx]]

    # 获取图像路径 (第0列和第1列)
    subject_id = str(row[0])
    # if 'CASME3/'+subject_id not in sublist:
    #     sublist.append('CASME3/'+subject_id)
# print(sublist)
    session_id = str(row[1])
    image_dir = os.path.join(path, subject_id, session_id.lower(), 'color')

    # 获取要选取的两张图片 (第2列和第3列)
    # if int(row[2]) == 0 or int(row[3]) == 0:
    #     continue
    onset = int(row[2])
    apex = int(row[3])
    offset = int(row[4])
    if onset == 0:
        continue
    if onset == apex:
        apex = (onset + offset) // 2

    img_num = []
    img_path = []
    for i in range(onset, offset+1):
        img_num = str(i) + '.jpg'
        img_path.append(os.path.join(image_dir, img_num))
    img1_num = str(onset) + '.jpg'
    # img2_num = str(apex) + '.jpg'
    # img1_path = os.path.join(image_dir, img1_num)
    # img2_path = os.path.join(image_dir, img2_num)

    flag = False

    img = Image.open(img_path[0])
    # next_img = Image.open(img2_path)

    img_format = img.format  # 保留原始格式
    img_mode = img.mode  # 保留原始模式 (RGB, RGBA, L)
    # 2. 检测人脸，返回裁剪后的PIL Image
    face1, _ = mtcnn.detect(img)
    # 3. 如果检测到人脸，进行裁剪
    next_imgs = []
    if face1 is not None:
        # 取第一个人脸的 bounding box
        box = face1[0]
        # cropped_img1 = img.crop((box[0], box[1], box[2], box[3]))
        cropped_img1 = crop_safe(img, box)
        # 4. 保持原始格式
        cropped_img1 = cropped_img1.convert(img_mode)
        cropped_img1.format = img_format

        for next_img_path in img_path:
            if not os.path.exists(next_img_path):
                continue
            next_img = Image.open(next_img_path)
            # cropped_img2 = next_img.crop((box[0], box[1], box[2], box[3]))
            cropped_img2 = crop_safe(next_img, box)
            # 4. 保持原始格式
            cropped_img2 = cropped_img2.convert(img_mode)
            cropped_img2.format = img_format
            next_imgs.append(cropped_img2)
    else:
        continue

    # save_path1 = os.path.join('/home/yicheng/CASME3_ME_all', subject_id, session_id, img1_num)
    if not os.path.exists(os.path.join('/home/yicheng/CASME3_ME_all', subject_id, session_id)):
        os.makedirs(os.path.join('/home/yicheng/CASME3_ME_all', subject_id, session_id))

    # cropped_img1.save(save_path1)

    num = onset

    for cropped_next_img in (next_imgs):
        save_path2 = os.path.join('/home/yicheng/CASME3_ME_all', subject_id, session_id, str(num)+'.jpg')
        cropped_next_img.save(save_path2)
        num += 1
