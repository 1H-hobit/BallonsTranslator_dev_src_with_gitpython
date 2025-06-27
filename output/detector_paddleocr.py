import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEFAULT_DEVICE, DEVICE_SELECTOR, ProjImgTrans
from paddleocr import TextDetection

PADDLE_OCR_MODEL_DIR = 'data/models/PP-OCRv5_server_det_infer'  # 请替换为实际的 PaddleOCR 检测模型路径
def load_paddleocr_model(model_dir):
# 修复设备标识符：将 'gpu' 映射为 'cuda'
    model = TextDetection(model_dir=model_dir, device="gpu:0")  # 添加use_gpu=False
    return model

@register_textdetectors('paddleocr_detector')
class PaddleOCRTextDetector(TextDetectorBase):

    params = {
        'description': 'PaddleOCR Text Detector',
        'font size multiplier': 1,
        'font size max': -1,
        'font size min': -1,
        'mask dilate size': 0,
        'dt_scores_threshold': 0.8,
    }
    _load_model_keys = {'model'}



    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None


    def _load_model(self):
        self.model = load_paddleocr_model(PADDLE_OCR_MODEL_DIR)

    def _detect(self, img: np.ndarray, proj: ProjImgTrans) -> Tuple[np.ndarray, List[TextBlock]]:
        # 调用 PaddleOCR 的文本检测方法
        result = self.model.predict(img)
        #print("_detect result:\n", result)
        
        # 处理返回结果格式：结果是一个包含字典的列表
        if isinstance(result, list) and len(result) > 0:
            # 取列表中的第一个元素（假设只有一张图片）
            result_dict = result[0]
            dt_polys = result_dict.get('dt_polys', [])
            dt_scores = result_dict.get('dt_scores', [])
        else:
            # 处理其他格式或空结果
            dt_polys = []
            dt_scores = []
        
        blk_list = []
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 处理每个检测到的文本框
        for i, box in enumerate(dt_polys):
            score = dt_scores[i]
            threshold = self.get_param_value('dt_scores_threshold')
            print("threshold:\n", threshold)
            print("score:\n", float(score))
            if float(score) < threshold:
                print("跳过")
                continue
 
            # 确保多边形格式正确 (n,1,2)
            if len(box) == 0:
                continue
                
            # 将多边形转换为整数类型
            if isinstance(box, np.ndarray):
                box_arr = box.astype(np.int32)
            else:
                box_arr = np.array(box, dtype=np.int32)
            
            # 重塑为OpenCV所需格式
            if box_arr.ndim == 2:
                if box_arr.shape[1] == 2:  # (n,2) -> (n,1,2)
                    box_arr = box_arr.reshape((-1, 1, 2))
                elif box_arr.shape[0] == 2:  # (2,n) -> 转置并重塑
                    box_arr = box_arr.T.reshape((-1, 1, 2))
            
            # 检查重塑后的形状
            if box_arr.ndim != 3 or box_arr.shape[2] != 2:
                print(f"警告：无法处理的多边形形状: {box_arr.shape}")
                continue

            # 填充文本框区域到掩码图
            #cv2.fillPoly(mask, [box_arr], 255)

            # 找出所有点的 x 坐标和 y 坐标
            x_coords = box_arr[:, 0, 0]
            y_coords = box_arr[:, 0, 1]

            # 确定右上角点（x最大且y最小的点）
            right_top_idx = np.lexsort((y_coords, -x_coords))[0]
            # 确定右下角点（x最大且y最大的点）
            right_bottom_idx = np.lexsort((-y_coords, -x_coords))[0]
            # 确定左上角点（x最小且y最小的点）
            left_top_idx = np.lexsort((y_coords, x_coords))[0]
            # 确定左下角点（x最小且y最大的点）
            left_bottom_idx = np.lexsort((-y_coords, x_coords))[0]

            # 复制原多边形，避免修改原图
            box_arr_modified = box_arr.copy()

            # 对左上角点的 x y 坐标
            box_arr_modified[left_top_idx, 0, 0] += 2    #  x 坐标减2
            box_arr_modified[left_top_idx, 0, 1] += 2    #  y 坐标加2
            # 对左下角点的 x y 坐标
            box_arr_modified[left_bottom_idx, 0, 0] += 2   #  x 坐标减2
            box_arr_modified[left_bottom_idx, 0, 1] -= 2  #  y 坐标加2

            # 对右上角点的 x y 坐标
            box_arr_modified[right_top_idx, 0, 0] -= 2    #  x 坐标减2
            box_arr_modified[right_top_idx, 0, 1] += 2     #  y 坐标加2
            # 对右下角点的 x y 坐标
            box_arr_modified[right_bottom_idx, 0, 0] -= 2   #  x 坐标减2
            box_arr_modified[right_bottom_idx, 0, 1] -= 2   #  y 坐标加2

            # 填充修改后的多边形到掩码图
            cv2.fillPoly(mask, [box_arr_modified], 255)

            # 计算文本框的边界框 (xmin, ymin, xmax, ymax)
            xyxy = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            # 计算文本框的高度作为字体大小的估计
            estimated_font_size = (max(y_coords) - min(y_coords)) * 0.8
            

            # --------------设定固定扩展像素
            expand_pixels = 2  # 可根据实际情况调整
            # 计算多边形中心点
            center_x = np.mean(box_arr_modified[:, 0, 0])
            center_y = np.mean(box_arr_modified[:, 0, 1])
            # 对每个顶点进行扩展
            expanded_box = box_arr_modified.copy()
            for i in range(len(expanded_box)):
                x, y = expanded_box[i, 0, 0], expanded_box[i, 0, 1]
                # 计算从中心点到当前顶点的方向向量
                dx = x - center_x
                dy = y - center_y
                # 如果dx和dy都为0（中心点与顶点重合），则跳过扩展
                if dx == 0 and dy == 0:
                    continue
                # 计算方向向量的长度
                length = np.sqrt(dx**2 + dy**2)
                # 扩展顶点：沿方向向量向外移动expand_pixels个像素
                expanded_x = x + (dx / length) * expand_pixels
                expanded_y = y + (dy / length) * expand_pixels
                # 更新扩展后的坐标
                expanded_box[i, 0, 0] = int(expanded_x)
                expanded_box[i, 0, 1] = int(expanded_y)
            # 使用扩展后的多边形更新lines
            lines = [expanded_box.reshape(-1, 2).tolist()]
            # 创建 TextBlock 对象
            #lines = [box_arr_modified.reshape(-1, 2).tolist()]

            blk = TextBlock(xyxy=xyxy, lines=lines, score=score)
            blk._detected_font_size = estimated_font_size
            blk_list.append(blk)

        # 应用字体大小调整参数
        fnt_rsz = self.get_param_value('font size multiplier')
        fnt_max = self.get_param_value('font size max')
        fnt_min = self.get_param_value('font size min')
        
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

        # 对掩码图进行膨胀操作，扩大文本区域
        ksize = self.get_param_value('mask dilate size')
        if ksize > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
            mask = cv2.dilate(mask, element)

        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        self._load_model()  # 重新加载模型以应用新设备