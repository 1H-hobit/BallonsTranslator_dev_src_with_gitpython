from typing import Optional, Union, Tuple
import numpy as np
from PIL import Image

import os
import torch
import cv2
import logging
import traceback
from iopaint.download import cli_download_model, scan_models
from iopaint.runtime import setup_model_dir
import tempfile
import os
from PIL import Image
from iopaint.batch_processing import batch_inpaint
from pathlib import Path


def _canonize_mask_array(mask):
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    if mask.ndim == 2 and mask.dtype == 'uint8':
        mask = mask[..., np.newaxis]
    assert mask.ndim == 3 and mask.shape[2] == 1 and mask.dtype == 'uint8'
    return np.ascontiguousarray(mask)

def preprocess_inputs(
    image: Union[np.ndarray, Image.Image],
    mask: Optional[Union[np.ndarray, Image.Image]] = None
) -> Tuple[np.ndarray, np.ndarray, str, Optional[str]]:
    """预处理输入图像和掩码，返回数组和临时文件路径"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = np.ascontiguousarray(image)
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == 'uint8'

        if mask is None:
            # 如果没有提供掩码，默认全图处理
            mask = np.zeros_like(image[..., :1], dtype='uint8')
        else:
            mask = _canonize_mask_array(mask)

        # 确保掩码与图像尺寸一致
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 生成临时文件路径

        # 获取当前目录并创建output文件夹

        temp_dir = Path(os.getcwd())
        output_dir = temp_dir / "output"
        output_dir.mkdir(exist_ok=True)
        img_path = os.path.join(output_dir, "batch_inpaint_image.jpg")
        mask_path = os.path.join(output_dir, "batch_inpaint_mask.png") 

        # 保存处理后的图像
        Image.fromarray(image).save(img_path)
        Image.fromarray(mask.squeeze()).save(mask_path)
        
        return image, mask, img_path, mask_path
    except Exception as e:
        logger.error(f"Error in input preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# 设置日志
logger = logging.getLogger(__name__)


def inpaint(
    image: Union[np.ndarray, Image.Image],
    mask: Optional[Union[np.ndarray, Image.Image]] = None,
    device: str = "cuda",
    model_dir: str = "data/models/LDM",
) -> Optional[np.ndarray]:
    try:
        logger.info(f"Starting inpainting with device: {device}, model_dir: {model_dir}")
        
        # 确定实际使用的设备
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logger.info(f"Actual device used: {device}")
        
        # 预处理输入
        img_arr, mask_arr, img_path, mask_path = preprocess_inputs(image, mask)
        
        # 执行修复
        logger.info("Running inpainting...")
        model_dir = Path(model_dir)
        setup_model_dir(model_dir)
        model = "ldm"
        scanned_models = scan_models()
        if model not in [it.name for it in scanned_models]:
            logger.info(f"{model} not found in {model_dir}, try to downloading")
            cli_download_model(model)
            
        # 获取当前目录并创建output文件夹
        current_dir = Path(os.getcwd())
        output_dir = current_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # 将路径转换为Path对象
        img_path = Path(img_path)
        mask_path = Path(mask_path) if mask_path else None
        output = output_dir
        config_js = output / "config.json"

        result_np = batch_inpaint(model, device, img_path, mask_path, output, config=config_js, concat=False)

        #save_p = output / "inpaint_result.png"
        
        if result_np is None or result_np.size == 0:  # 检查数组是否为空或无数据
            return None
        else:
            logger.info(f"图片修复完成。结果形状: {result_np.shape}")
            return result_np
    
    except Exception as e:
        logger.error(f"图片修复过程中出现严重错误: {str(e)}")
        logger.error(traceback.format_exc())
        return None