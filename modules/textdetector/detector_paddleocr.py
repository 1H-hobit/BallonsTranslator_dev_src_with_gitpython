from utils.config import pcfg
import numpy as np
import cv2
from typing import Tuple, List, Optional

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEFAULT_DEVICE, DEVICE_SELECTOR, ProjImgTrans
from paddleocr import TextDetection
from utils.textblock import mit_merge_textlines, sort_regions, examine_textblk

PADDLE_OCR_MODEL_DIR = 'data/models/PP-OCRv5_server_det_infer'

def load_paddleocr_model(model_dir):
    model = TextDetection(model_dir=model_dir, device="gpu:0")
    return model

def generate_mask_from_user_rect(img: np.ndarray, rect: np.ndarray, mask_expand_pixels: int) -> np.ndarray:
    """根据用户绘制的矩形区域生成掩码"""
    # 解包矩形坐标 [x1, y1, x2, y2]
    min_x, min_y, max_x, max_y = rect
    
    # 创建掩码
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    h, w = img.shape[:2]
    
    # 应用掩码扩展值
    x_min = max(0, min_x - mask_expand_pixels)
    y_min = max(0, min_y - mask_expand_pixels)
    x_max = min(w, max_x + mask_expand_pixels)
    y_max = min(h, max_y + mask_expand_pixels)
    
    # 确保坐标有效
    if x_min >= x_max or y_min >= y_max:
        return mask
    
    # 创建矩形
    expanded_rect = np.array([
        [[x_min, y_min]],
        [[x_max, y_min]],
        [[x_max, y_max]],
        [[x_min, y_max]]
    ], dtype=np.int32)
    
    # 填充白色区域
    cv2.fillPoly(mask, [expanded_rect], 255)
    return mask

def generate_mask_from_original_bbox(img: np.ndarray, polygon_info_list: List[dict], mask_expand_pixels: int) -> np.ndarray:
    """根据原始边界框生成掩码，使用掩码扩展值"""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    h, w = img.shape[:2]
    for item in polygon_info_list:
        # 获取原始边界框
        min_x = item['min_x']
        min_y = item['min_y']
        max_x = item['max_x']
        max_y = item['max_y']
        
        # 应用掩码扩展值
        x_min = max(0, min_x - mask_expand_pixels)
        y_min = max(0, min_y - mask_expand_pixels)
        x_max = min(w, max_x + mask_expand_pixels)
        y_max = min(h, max_y + mask_expand_pixels)
        
        # 确保坐标有效
        if x_min >= x_max or y_min >= y_max:
            continue
        
        # 创建矩形
        expanded_rect = np.array([
            [[x_min, y_min]],
            [[x_max, y_min]],
            [[x_max, y_max]],
            [[x_min, y_max]]
        ], dtype=np.int32)
        
        # 填充白色区域
        cv2.fillPoly(mask, [expanded_rect], 255)
    return mask


@register_textdetectors('paddleocr_detector')
class PaddleOCRTextDetector(TextDetectorBase):

    params = {
        'description': 'PaddleOCR Text Detector',
        'bbox expand pixels': {
            'type': 'selector',
            'options': [str(i) for i in range(0, 21)],
            'value': '8',
            'label': '文本框扩展像素'
        },
        # 添加掩码扩展像素参数
        'mask expand pixels': {
            'type': 'selector',
            'options': [str(i) for i in range(0, 21)],
            'value': '6',
            'label': '掩码扩展像素'
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
        'merge text lines': {
            'type': 'selector',
            'options': ['是', '否'],
            'value': '否',
            'label': '合并文本行'
        },
        'auto detect text direction': {
            'type': 'selector',
            'options': ['是', '否'],
            'value': '是',
            'label': '自动检测文本方向'
        },
        'mask dilate size': {
            'type': 'selector',
            'options': [str(i) for i in range(0, 11)],
            'value': '2',
            'label': '掩码膨胀大小'
        },
        'dt_scores_threshold': {
            'type': 'selector',
            'options': [f"{i/10:.1f}" for i in range(0, 11)],
            'value': '0.8',
            'label': '置信度阈值'
        },
        'device': DEVICE_SELECTOR(),
    }
    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None

    def _load_model(self):
        self.model = load_paddleocr_model(PADDLE_OCR_MODEL_DIR)

    def get_boolean_param(self, param_key):
        value = self.get_param_value(param_key)
        if isinstance(value, str):
            return value.lower() in ['true', '是', '启用', 'yes']
        return bool(value)
    
    def get_numeric_param(self, param_key):
        value = self.get_param_value(param_key)
        if isinstance(value, str) and value.startswith('-1'):
            return -1
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except:
            return value

    def _detect(self, img: np.ndarray, proj: ProjImgTrans) -> Tuple[np.ndarray, List[TextBlock]]:
        result = self.model.predict(img)
        
        if isinstance(result, list) and len(result) > 0:
            result_dict = result[0]
            dt_polys = result_dict.get('dt_polys', [])
            dt_scores = result_dict.get('dt_scores', [])
        else:
            dt_polys = []
            dt_scores = []
        
        blk_list = []
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        polygon_info_list = []
        auto_detect_direction = self.get_boolean_param('auto detect text direction')
        threshold = self.get_numeric_param('dt_scores_threshold')
        im_h, im_w = img.shape[:2]

        # 第一阶段：收集所有检测框信息并填充掩码
        for i, box in enumerate(dt_polys):
            score = dt_scores[i]
            if float(score) < threshold:
                continue

            if len(box) == 0:
                continue
                
            if isinstance(box, np.ndarray):
                box_arr = box.astype(np.int32)
            else:
                box_arr = np.array(box, dtype=np.int32)
            
            if box_arr.ndim == 2:
                if box_arr.shape[1] == 2:
                    box_arr = box_arr.reshape((-1, 1, 2))
                elif box_arr.shape[0] == 2:
                    box_arr = box_arr.T.reshape((-1, 1, 2))
            
            if box_arr.ndim != 3 or box_arr.shape[2] != 2:
                continue

            # === 自动检测文本方向 ===
            x_coords = box_arr[:, 0, 0]
            y_coords = box_arr[:, 0, 1]
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            width = max_x - min_x
            height = max_y - min_y
            
            # 自动检测文本方向
            if auto_detect_direction:
                # 计算宽高比
                aspect_ratio = width / (height + 1e-5)
                
                # 判断文本方向
                if aspect_ratio > 1.5:  # 宽大于高，水平文本
                    actual_vertical = False
                elif aspect_ratio < 0.67:  # 高大于宽，垂直文本
                    actual_vertical = True
                else:  # 接近正方形，使用对角线方向
                    diag_angle = np.degrees(np.arctan2(height, width))
                    actual_vertical = diag_angle > 45
            else:
                # 如果不自动检测，使用默认设置（水平）
                actual_vertical = False
            
            # === 确保文本区域有最小尺寸 ===
            min_dimension = 3  # 最小尺寸
            if actual_vertical:  # 垂直文本
                if width < min_dimension:
                    # 垂直文本需要最小宽度
                    new_width = max(min_dimension, int(height * 0.15))
                    center_x = (min_x + max_x) // 2
                    min_x = center_x - new_width // 2
                    max_x = center_x + new_width // 2
                    width = max_x - min_x
                    
                    # 更新多边形点
                    for j in range(len(box_arr)):
                        if box_arr[j, 0, 0] < center_x:
                            box_arr[j, 0, 0] = min_x
                        else:
                            box_arr[j, 0, 0] = max_x
            else:  # 水平文本
                if height < min_dimension:
                    # 水平文本需要最小高度
                    new_height = max(min_dimension, int(width * 0.15))
                    center_y = (min_y + max_y) // 2
                    min_y = center_y - new_height // 2
                    max_y = center_y + new_height // 2
                    height = max_y - min_y
                    
                    # 更新多边形点
                    for j in range(len(box_arr)):
                        if box_arr[j, 0, 1] < center_y:
                            box_arr[j, 0, 1] = min_y
                        else:
                            box_arr[j, 0, 1] = max_y

            # 填充mask的掩码区域为白色
            # cv2.fillPoly(mask, [box_arr], 255)
            
            # 重新计算实际尺寸
            x_coords = box_arr[:, 0, 0]
            y_coords = box_arr[:, 0, 1]
            width = np.max(x_coords) - np.min(x_coords)
            height = np.max(y_coords) - np.min(y_coords)
            
            # 字体大小计算
            if actual_vertical:
                estimated_font_size = width
            else:
                estimated_font_size = height

            # 存储多边形信息
            polygon_info_list.append({
                'points': box_arr.reshape(-1, 2).tolist(),
                'font_size': estimated_font_size,
                'score': score,
                'width': width,
                'height': height,
                'min_x': min_x,
                'min_y': min_y,
                'max_x': max_x,
                'max_y': max_y,
                'actual_vertical': actual_vertical
            })

        # 第二阶段：根据合并选项创建文本块
        merge_text_lines = self.get_boolean_param('merge text lines')
        expand_pixels = self.get_numeric_param('bbox expand pixels') or 8

        if merge_text_lines:
            # 只合并文本行，不单独创建文本块
            poly_list = [item['points'] for item in polygon_info_list]
            merged_blks = mit_merge_textlines(poly_list, width=im_w, height=im_h)
            
            # 为合并后的文本块创建TextBlock对象
            for blk in merged_blks:
                font_sizes = []
                actual_vertical = None
                
                # 收集所有文本行的方向信息
                for line in blk.lines:
                    pts = np.array(line)
                    x_coords = pts[:, 0]
                    y_coords = pts[:, 1]
                    width = np.max(x_coords) - np.min(x_coords)
                    height = np.max(y_coords) - np.min(y_coords)
                    
                    # 自动检测文本行方向
                    aspect_ratio = width / (height + 1e-5)
                    if aspect_ratio > 1.5:  # 水平文本
                        line_vertical = False
                    elif aspect_ratio < 0.67:  # 垂直文本
                        line_vertical = True
                    else:  # 正方形，使用对角线方向
                        diag_angle = np.degrees(np.arctan2(height, width))
                        line_vertical = diag_angle > 45
                    
                    if actual_vertical is None:
                        actual_vertical = line_vertical
                    
                    # 计算字体大小
                    if line_vertical:
                        font_size = width
                    else:
                        font_size = height
                    font_sizes.append(font_size)
                
                # 计算中位字体大小
                if font_sizes:
                    median_font_size = np.median(font_sizes)
                
                # 扩展文本块边界框
                x1, y1, x2, y2 = blk.xyxy
                expanded_xyxy = [
                    max(0, x1 - expand_pixels),
                    max(0, y1 - expand_pixels),
                    min(im_w, x2 + expand_pixels),
                    min(im_h, y2 + expand_pixels)
                ]
                
                # 创建新的多边形边界
                x1, y1, x2, y2 = expanded_xyxy
                expanded_lines = [[
                    [x1, y1], [x2, y1],
                    [x2, y2], [x1, y2]
                ]]
                
                # === 创建完整的TextBlock对象 ===
                text_block = TextBlock(
                    xyxy=expanded_xyxy,
                    lines=expanded_lines,
                    score=np.mean([item['score'] for item in polygon_info_list]),  # 平均置信度
                    src_is_vertical=actual_vertical,
                    vertical=actual_vertical
                )
                text_block._detected_font_size = median_font_size
                blk_list.append(text_block)
                
                # 确保文本块有有效尺寸
                x1, y1, x2, y2 = text_block.xyxy
                if actual_vertical:  # 垂直文本
                    if x2 - x1 < 3:  # 宽度太小
                        min_width = max(12, int(median_font_size * 0.8))
                        center_x = (x1 + x2) // 2
                        text_block.xyxy = [center_x - min_width//2, y1, center_x + min_width//2, y2]
                else:  # 水平文本
                    if y2 - y1 < 3:  # 高度太小
                        min_height = max(12, int(median_font_size * 0.8))
                        center_y = (y1 + y2) // 2
                        text_block.xyxy = [x1, center_y - min_height//2, x2, center_y + min_height//2]
            # blk_list = merged_blks
        else:
            # 不合并文本行，为每个检测框创建文本块
            for item in polygon_info_list:
                # 使用第一阶段计算的实际边界值
                min_x = item['min_x']
                min_y = item['min_y']
                max_x = item['max_x']
                max_y = item['max_y']
                actual_vertical = item['actual_vertical']
                
                # 计算扩展后的边界框
                x1 = max(0, min_x - expand_pixels)
                y1 = max(0, min_y - expand_pixels)
                x2 = min(im_w, max_x + expand_pixels)
                y2 = min(im_h, max_y + expand_pixels)
                
                # 创建扩展后的多边形线
                expanded_points = [
                    [x1, y1],  # 左上角
                    [x2, y1],  # 右上角
                    [x2, y2],  # 右下角
                    [x1, y2]   # 左下角
                ]

                # 创建文本块
                blk = TextBlock(
                    xyxy=[x1, y1, x2, y2],
                    lines=[expanded_points],  # 使用扩展后的多边形线
                    score=item['score'],
                    src_is_vertical=actual_vertical,
                    vertical=actual_vertical
                )
                blk._detected_font_size = item['font_size']
                
                # 确保文本块有有效尺寸
                x1, y1, x2, y2 = blk.xyxy
                if actual_vertical:  # 垂直文本
                    if x2 - x1 < 3:  # 宽度太小
                        min_width = max(12, int(item['font_size'] * 0.8))
                        center_x = (x1 + x2) // 2
                        blk.xyxy = [center_x - min_width//2, y1, center_x + min_width//2, y2]
                        
                        # 更新多边形线以匹配新边界框
                        blk.lines = [[
                            [center_x - min_width//2, y1],
                            [center_x + min_width//2, y1],
                            [center_x + min_width//2, y2],
                            [center_x - min_width//2, y2]
                        ]]
                else:  # 水平文本
                    if y2 - y1 < 3:  # 高度太小
                        min_height = max(12, int(item['font_size'] * 0.8))
                        center_y = (y1 + y2) // 2
                        blk.xyxy = [x1, center_y - min_height//2, x2, center_y + min_height//2]
                        
                        # 更新多边形线以匹配新边界框
                        blk.lines = [[
                            [x1, center_y - min_height//2],
                            [x2, center_y - min_height//2],
                            [x2, center_y + min_height//2],
                            [x1, center_y + min_height//2]
                        ]]
                
                blk_list.append(blk)

        # 重新排序文本块
        blk_list = sort_regions(blk_list)

        # 获取掩码扩展值
        mask_expand_pixels = self.get_numeric_param('mask expand pixels') or 6
        # 使用最终文本块生成掩码
        mask = generate_mask_from_original_bbox(img, polygon_info_list, mask_expand_pixels)       

        # 字体大小调整
        fnt_rsz = self.get_numeric_param('font size multiplier')
        fnt_max = self.get_numeric_param('font size max')
        fnt_min = self.get_numeric_param('font size min')
        
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            
            # 应用字体大小限制
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            
            # # 确保字体大小合理
            # if sz < 5:
            #     sz = 12
            # elif sz > 100:
            #     sz = 48
            
            blk.font_size = sz
            blk._detected_font_size = sz
            
            # 设置对齐方式
            if blk.vertical:
                blk.alignment = 1  # 垂直文本右对齐
            else:
                blk.recalulate_alignment()  # 水平文本自动计算对齐

        # 掩码膨胀处理
        ksize = self.get_numeric_param('mask dilate size')
        if ksize > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
            mask = cv2.dilate(mask, element)

        # 确保在返回前保存多边形信息
        proj.polygon_info_list = polygon_info_list  # 确保这行代码在返回前执行

        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        self._load_model()