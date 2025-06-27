import os
import os.path as osp
from typing import Tuple, List

import torch
import numpy as np
import cv2

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from utils.textblock import mit_merge_textlines, sort_regions, examine_textblk
from utils.textblock_mask import canny_flood
from utils.split_text_region import manga_split, split_textblock
from utils.imgproc_utils import xywh2xyxypoly
from utils.proj_imgtrans import ProjImgTrans

# 定义模型存储目录
MODEL_DIR = 'data/models'
# 存储检查点文件路径的列表
CKPT_LIST = []

# 更新检查点文件列表
def update_ckpt_list():
    # 检查模型目录是否存在
    if not osp.exists(MODEL_DIR):
        return
    global CKPT_LIST
    # 清空列表
    CKPT_LIST.clear()
    # 遍历模型目录下的所有文件
    for p in os.listdir(MODEL_DIR):
        # 筛选以 'ysgyolo' 或 'ultralyticsyolo' 开头的文件
        if p.startswith('ysgyolo') or p.startswith('ultralyticsyolo'):
            # 将文件路径添加到列表中，并将反斜杠替换为正斜杠
            CKPT_LIST.append(osp.join(MODEL_DIR, p).replace('\\', '/'))

# 类别映射字典，将原始类别名称映射到自定义的类别名称
CLS_MAP = {
    'balloon': 'vertical_textline',
    'changfangtiao': 'horizontal_textline',
    'qipao': 'textblock',
    'fangkuai': 'angled_vertical_textline',
    'kuangwai': 'angled_horizontal_textline',
    'other': 'other'
}

# 定义文字相关的类别
TEXT_CLASSES = ['balloon', 'changfangtiao', 'qipao', 'fangkuai', 'kuangwai']

# 调用函数更新检查点文件列表
update_ckpt_list()


# 注册名为 'ysgyolo' 的文本检测器
@register_textdetectors('ysgyolo')
class YSGYoloDetector(TextDetectorBase):
    # 检测器的参数配置
    params = {
        'model path': {
            'type': 'selector',
            'options': CKPT_LIST,
            'value': 'data/models/ysgyolo_v11_x.pt',
            'editable': True,
            'flush_btn': True,
            'path_selector': True,
            'path_filter': '*.pt *.ckpt *.pth *.safetensors',
            'size': 'median'
        },
        'merge text lines': {
            'type': 'selector',
            'options': ['是', '否'],
            'value': '否',
            'label': '合并文本行'
        },
        'confidence threshold': {
            'type': 'selector',
            'options': [f"{i/10:.1f}" for i in range(0, 11)],
            'value': '0.4',
            'label': '置信度阈值'
        },
        'IoU threshold': {
            'type': 'selector',
            'options': [f"{i/10:.1f}" for i in range(0, 11)],
            'value': '0.3',
            'label': 'IoU阈值'
        },
        'font size multiplier': {
            'type': 'selector',
            'options': [f"{i/10:.1f}" for i in range(1, 51)],
            'value': '1.0',
            'label': '字体大小乘数'
        },
        'font size max': {
            'type': 'selector',
            'options': ['-1（不限制）'] + [str(i) for i in range(1, 101)],
            'value': '-1（不限制）',
            'label': '最大字体大小'
        },
        'font size min': {
            'type': 'selector',
            'options': ['-1（不限制）'] + [str(i) for i in range(1, 101)],
            'value': '-1（不限制）',
            'label': '最小字体大小'
        },
        'detect size': {
            'type': 'selector',
            'options': [str(i) for i in range(256, 4097, 256)],
            'value': '1024',
            'label': '检测尺寸'
        },
        'device': DEVICE_SELECTOR(),
        'label': {
            'value': {  
                'vertical_textline': True, 
                'horizontal_textline': True, 
                'angled_vertical_textline': True, 
                'angled_horizontal_textline': True,
                'textblock': True
            }, 
            'type': 'check_group'
        },
        'source text is vertical': {
            'type': 'selector',
            'options': ['是', '否'],
            'value': '是',
            'label': '源文本是否垂直'
        },
        'mask dilate size': {
            'type': 'selector',
            'options': [str(i) for i in range(0, 11)],
            'value': '2',
            'label': '掩码膨胀大小'
        }
    }

    # 加载模型时需要的键
    _load_model_keys = {'model'}

    # 初始化函数
    def __init__(self, **params) -> None:
        # 调用父类的初始化函数
        super().__init__(**params)
        # 更新检查点文件列表
        update_ckpt_list()
    
    # 加载模型的函数
    def _load_model(self):
        # 获取模型路径参数的值
        model_path = self.get_param_value('model path')
        # 检查模型文件是否存在
        if not osp.exists(model_path):
            global CKPT_LIST
            df_model_path = model_path
            # 遍历检查点文件列表，寻找存在的文件
            for p in CKPT_LIST:
                if osp.exists(p):
                    df_model_path = p
                    break
            # 记录警告信息，尝试回退到默认模型路径
            self.logger.warning(f'{model_path} does not exist, try fall back to default value {df_model_path}')
            model_path = df_model_path

        # 根据模型文件名判断使用的模型类型
        if 'rtdetr' in os.path.basename(model_path):
            from ultralytics import RTDETR as MODEL
        else:
            from ultralytics import YOLO as MODEL
        # 如果模型未加载，则加载模型并指定设备
        if not hasattr(self, 'model') or self.model is None:
            self.model = MODEL(model_path).to(device=self.get_param_value('device'))

    # 获取有效的类别标签
    def get_valid_labels(self):
        # 筛选出值为 True 且不是 'textblock' 的类别标签
        valid_labels = [k for k, v in self.params['label']['value'].items() if v and k != 'textblock']
        return valid_labels

    # 判断是否使用的是 YSG 模型
    @property
    def is_ysg(self):
        return osp.basename(self.get_param_value('model path').startswith('ysg'))
    
    # 获取布尔类型参数的值
    def get_boolean_param(self, param_key):
        value = self.get_param_value(param_key)
        if isinstance(value, str):
            return value.lower() in ['true', '是', '启用']
        return bool(value)
    
    # 获取数值类型参数的值
    def get_numeric_param(self, param_key):
        value = self.get_param_value(param_key)
        # 处理 "不限制" 选项
        if isinstance(value, str) and value.startswith('-1'):
            return -1
        try:
            # 尝试转换为浮点数或整数
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except:
            return value

    # 文本检测函数
    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:

        # 使用模型进行预测
        result = self.model.predict(
            source=img, save=False, show=False, verbose=False, 
            conf=self.get_numeric_param('confidence threshold'), 
            iou=self.get_numeric_param('IoU threshold'),
            agnostic_nms=True
        )[0]
        # 存储有效的类别 ID
        valid_ids = []
        # 获取有效的类别标签集合
        valid_labels = set(self.get_valid_labels())
        # 文本块的类别索引
        textblock_idx = -1
        # 遍历模型预测结果的类别名称
        for idx, name in result.names.items():
            # 如果类别不在文字相关类别中，将其映射为 'other'
            if name not in TEXT_CLASSES:
                name = 'other'
            if CLS_MAP[name] in valid_labels:
                valid_ids.append(idx)
            if name == 'qipao':
                textblock_idx = idx
        need_textblock = self.params['label']['value']['textblock'] == True

        # 初始化掩码矩阵
        mask = np.zeros_like(img[..., 0])
        # 如果没有有效的类别 ID 且不需要检测文本块，则返回空列表和掩码矩阵
        if len(valid_ids) == 0 and not need_textblock:
            return [], mask

        # 获取图像的高度和宽度
        im_h, im_w = img.shape[:2]
        # 存储检测到的文本区域的顶点列表
        pts_list = []
        # 存储检测到的文本块列表
        blk_list = []

        # 获取模型预测的边界框信息
        dets = result.boxes
        if dets is not None and len(dets.cls) > 0:
            # 获取设备信息
            device = dets.cls.device
            # 初始化有效掩码
            valid_mask = torch.zeros((dets.cls.shape[0]), device=device, dtype=torch.bool)
            # 筛选出有效类别的边界框
            for idx in valid_ids:
                valid_mask = torch.bitwise_or(valid_mask, dets.cls == idx)
            if torch.any(valid_mask):
                # 获取有效类别的边界框坐标
                xyxy_list = dets.xyxy[valid_mask]
                # 将坐标转换为 CPU 上的整数类型
                xyxy_list = xyxy_list.to(device='cpu', dtype=torch.float32).round().to(torch.int32)
                # 裁剪坐标，确保在图像范围内
                xyxy_list[:, [0, 2]] = torch.clip(xyxy_list[:, [0, 2]], 0, im_w - 1)
                xyxy_list[:, [1, 3]] = torch.clip(xyxy_list[:, [1, 3]], 0, im_h - 1)
                # 将坐标转换为 numpy 数组
                xyxy_list = xyxy_list.numpy()
                # 在掩码矩阵上绘制矩形
                for xyxy in xyxy_list:
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                # 将边界框坐标转换为多边形顶点坐标
                xyxy_list[:, [2, 3]] -= xyxy_list[:, [0, 1]]
                pts_list += xywh2xyxypoly(xyxy_list).reshape(-1, 4, 2).tolist()
            
            # 如果需要检测文本块
            if need_textblock:
                # 筛选出文本块的边界框
                valid_mask = dets.cls == textblock_idx
                # 判断源文本是否为垂直文本
                is_vertical = self.get_boolean_param('source text is vertical')
                if torch.any(valid_mask):
                    # 获取文本块的边界框坐标
                    xyxy_list = dets.xyxy[valid_mask]
                    # 将坐标转换为 CPU 上的整数类型
                    xyxy_list = xyxy_list.to(device='cpu', dtype=torch.float32).round().to(torch.int32)
                    # 裁剪坐标，确保在图像范围内
                    xyxy_list[:, [0, 2]] = torch.clip(xyxy_list[:, [0, 2]], 0, im_w - 1)
                    xyxy_list[:, [1, 3]] = torch.clip(xyxy_list[:, [1, 3]], 0, im_h - 1)
                    # 将坐标转换为 numpy 数组
                    xyxy_list = xyxy_list.numpy()
                    for xyxy in xyxy_list:
                        x1, y1, x2, y2 = xyxy
                        # 裁剪图像
                        crop = img[y1: y2, x1: x2]
                        # 使用 Canny 边缘检测和洪水填充算法生成掩码
                        bmask  = canny_flood(crop)[0]
                        if is_vertical:
                            # 对垂直文本进行分割
                            span_list = manga_split(bmask)
                            # 调整分割后的文本行坐标
                            lines = [[line.left + x1, line.top + y1, line.width, line.height] for line in span_list]
                            lines = np.array(lines)[::-1]
                            # 计算字体大小
                            font_sz = np.mean(lines[:, 2])
                        else:
                            # 对水平文本进行分割
                            span_list = split_textblock(bmask)[0]
                            # 调整分割后的文本行坐标
                            lines = [[line.left + x1, line.top + y1, line.width, line.height] for line in span_list]
                            lines = np.array(lines)
                            # 计算字体大小
                            font_sz = np.mean(lines[:, 3])
                        # 在掩码矩阵上绘制矩形
                        for line in lines:
                            x1, y1, x2, y2 = line
                            x2 += x1
                            y2 += y1
                            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                        # 将边界框坐标转换为多边形顶点坐标
                        lines = xywh2xyxypoly(lines).reshape(-1, 4, 2).tolist()
                        # 创建文本块对象
                        blk = TextBlock(xyxy=xyxy, lines=np.array(lines), src_is_vertical=is_vertical, vertical=is_vertical)
                        blk.font_size = font_sz
                        blk._detected_font_size = font_sz
                        if is_vertical:
                            blk.alignment = 1
                        else:
                            blk.recalulate_alignment()

                        blk_list.append(blk)
                        
                        # cv2.imwrite('mask.jpg', mask)
                        # for ii in range(len(blk.lines)):
                        #     rst = blk.get_transformed_region(img, ii, 48)
                        #     cv2.imwrite('local_tst.jpg', rst)
                        #     pass

        # 获取模型预测的有向边界框信息
        dets = result.obb
        if dets is not None and len(dets.cls) > 0:
            # 获取设备信息
            device = dets.cls.device
            # 初始化有效掩码
            valid_mask = torch.zeros((dets.cls.shape[0]), device=device, dtype=torch.bool)
            # 筛选出有效类别的有向边界框
            for idx in valid_ids:
                valid_mask = torch.bitwise_or(valid_mask, dets.cls == idx)
            if torch.any(valid_mask):
                # 获取有效类别的有向边界框坐标
                xyxy_list = dets.xyxyxyxy[valid_mask]
                # 将坐标转换为 CPU 上的整数类型
                xyxy_list = xyxy_list.to(device='cpu', dtype=torch.float32).round().to(torch.int32)
                # 裁剪坐标，确保在图像范围内
                xyxy_list[..., 0] = torch.clip(xyxy_list[..., 0], 0, im_w - 1)
                xyxy_list[..., 1] = torch.clip(xyxy_list[..., 1], 0, im_h - 1)
                # 将坐标转换为 numpy 数组
                xyxy_list = xyxy_list.numpy()
                # 在掩码矩阵上绘制多边形
                for pts in xyxy_list:
                    cv2.fillPoly(mask, [pts], 255)
                pts_list += xyxy_list.tolist()

        # 如果需要合并文本行
        if self.get_boolean_param('merge text lines'):
            # 合并文本行并添加到文本块列表
            blk_list += mit_merge_textlines(pts_list, width=im_w, height=im_h)
        else:
            for pts in pts_list:
                # 创建文本块对象
                blk = TextBlock(lines=[pts])
                # 调整文本块的边界框
                blk.adjust_bbox()
                # 检查文本块的有效性
                examine_textblk(blk, im_w, im_h)
                blk_list.append(blk)
        # 对文本块列表进行排序
        blk_list = sort_regions(blk_list)


        # 获取字体大小乘数
        fnt_rsz = self.get_numeric_param('font size multiplier')
        # 获取最大字体大小
        fnt_max = self.get_numeric_param('font size max')
        # 获取最小字体大小
        fnt_min = self.get_numeric_param('font size min')
        # 调整文本块的字体大小
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

        # 获取掩码膨胀大小
        ksize = self.get_numeric_param('mask dilate size')
        if ksize > 0:
            # 创建结构元素
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),(ksize, ksize))
            # 对掩码矩阵进行膨胀操作
            mask = cv2.dilate(mask, element)
            
        return mask, blk_list

    # 更新参数的函数
    def updateParam(self, param_key: str, param_content):
        # 调用父类的更新参数函数
        super().updateParam(param_key, param_content)
        
        # 如果更新的是模型路径参数，则删除已加载的模型
        if param_key == 'model path':
            if hasattr(self, 'model'):
                del self.model

    # 刷新参数的函数
    def flush(self, param_key: str):
        # 如果刷新的是模型路径参数，则更新检查点文件列表
        if param_key == 'model path':
            update_ckpt_list()
            global CKPT_LIST
            return CKPT_LIST