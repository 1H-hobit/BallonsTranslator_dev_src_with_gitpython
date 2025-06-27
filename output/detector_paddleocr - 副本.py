import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEFAULT_DEVICE, DEVICE_SELECTOR, ProjImgTrans
from paddleocr import TextDetection
from utils.textblock import mit_merge_textlines, sort_regions, examine_textblk

PADDLE_OCR_MODEL_DIR = 'data/models/PP-OCRv5_server_det_infer'

def load_paddleocr_model(model_dir):
    model = TextDetection(model_dir=model_dir, device="gpu:0")
    return model

@register_textdetectors('paddleocr_detector')
class PaddleOCRTextDetector(TextDetectorBase):

    params = {
        'description': 'PaddleOCR Text Detector',
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
            return value.lower() in ['true', '是', '启用']
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
            # 存储所有检测到的多边形信息（点、字体大小、分数）
            polygon_info_list = []
            is_vertical = self.get_boolean_param('source text is vertical')
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

                # 修改多边形坐标
                box_arr_modified = box_arr.copy()
                x_coords = box_arr_modified[:, 0, 0]
                y_coords = box_arr_modified[:, 0, 1]
                
                # 对左上角点的调整
                left_top_idx = np.lexsort((y_coords, x_coords))[0]
                box_arr_modified[left_top_idx, 0, 0] += 2
                box_arr_modified[left_top_idx, 0, 1] += 2
                
                # 对左下角点的调整
                left_bottom_idx = np.lexsort((-y_coords, x_coords))[0]
                box_arr_modified[left_bottom_idx, 0, 0] += 2
                box_arr_modified[left_bottom_idx, 0, 1] -= 2
                
                # 对右上角点的调整
                right_top_idx = np.lexsort((y_coords, -x_coords))[0]
                box_arr_modified[right_top_idx, 0, 0] -= 2
                box_arr_modified[right_top_idx, 0, 1] += 2
                
                # 对右下角点的调整
                right_bottom_idx = np.lexsort((-y_coords, -x_coords))[0]
                box_arr_modified[right_bottom_idx, 0, 0] -= 2
                box_arr_modified[right_bottom_idx, 0, 1] -= 2

                cv2.fillPoly(mask, [box_arr_modified], 255)
                estimated_font_size = (max(y_coords) - min(y_coords)) * 0.8

                # 存储多边形信息
                polygon_info_list.append({
                    'points': box_arr_modified.reshape(-1, 2).tolist(),
                    'font_size': estimated_font_size,
                    'score': score
                })

            # 第二阶段：根据合并选项创建文本块
            merge_text_lines = self.get_boolean_param('merge text lines')

            if merge_text_lines:
                # 只合并文本行，不单独创建文本块
                poly_list = [item['points'] for item in polygon_info_list]
                merged_blks = mit_merge_textlines(poly_list, width=im_w, height=im_h)
                
                # 为合并后的文本块添加字体大小信息
                for blk in merged_blks:
                    # 计算合并后文本块的高度作为字体大小估计
                    h = blk.xyxy[3] - blk.xyxy[1]
                    blk._detected_font_size = h * 0.8
                    blk.src_is_vertical = is_vertical
                    blk.vertical = is_vertical
                blk_list = merged_blks
            else:
                # 不合并文本行，为每个检测框创建文本块
                for item in polygon_info_list:
                    points = item['points']
                    # 计算边界框
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    xyxy = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    
                    blk = TextBlock(
                        xyxy=xyxy, 
                        lines=[points],
                        score=item['score'],
                        src_is_vertical=is_vertical,
                        vertical=is_vertical
                    )
                    blk._detected_font_size = item['font_size']
                    blk_list.append(blk)

            # 重新排序文本块
            blk_list = sort_regions(blk_list)

            # 字体大小调整
            fnt_rsz = self.get_numeric_param('font size multiplier')
            fnt_max = self.get_numeric_param('font size max')
            fnt_min = self.get_numeric_param('font size min')
            
            for blk in blk_list:
                sz = blk._detected_font_size * fnt_rsz
                if fnt_max > 0:
                    sz = min(fnt_max, sz)
                if fnt_min > 0:
                    sz = max(fnt_min, sz)
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

            return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        self._load_model()