from typing import List  # 确保使用标准库
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock
import numpy as np
from modelscope import AutoModel, AutoTokenizer
from transformers import TextIteratorStreamer
import torch
from threading import Thread
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode  # 图像插值模式
from PIL import Image
import cv2

ocr_model = None
ocr_tokenizer = None

@register_OCR('InternVL3-8B')
class CustomOCR(OCRBase):
    lang_map = {
                "Chinese & English": "ch",
    }

    params = {
        # 定义自定义参数
        "max_new_tokens": {
            "value": 4000,
            "description": "max_new_tokens"
        },
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.max_new_tokens = self.params["max_new_tokens"]["value"]
        self.model = None  # 初始化模型变量


    def _load_model(self):
        MODEL_NAME = r"D:\chainlit\models\InternVL3-8B"
        global ocr_model
        global ocr_tokenizer

        ocr_model = AutoModel.from_pretrained(
            MODEL_NAME,
            load_in_4bit=True,
            #load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            device_map="auto",
            trust_remote_code=True).eval()

        ocr_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

    # ======================== 卸载OCR模型和处理器，释放内存 ========================
    def _unload_ocr_model(self):
        global ocr_model
        global ocr_tokenizer
        
        try:
            # 删除模型和分词器的所有引用（不再尝试移动设备）
            del ocr_model
            del ocr_tokenizer
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
            print("✅ OCR模型已成功卸载")
            
        except Exception as e:
            print(f"❌ 卸载OCR模型失败: {str(e)}")
        finally:
            # 确保全局变量置空
            ocr_model = None
            ocr_tokenizer = None


    # ai_ocr图像预处理
    def build_transform(self, input_size):
        # ImageNet数据集的标准化参数
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        """构建图像预处理流水线"""
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        return T.Compose([
            # 确保图像为RGB格式
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            # 调整大小并使用双三次插值
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),  # 转换为张量
            T.Normalize(mean=MEAN, std=STD)  # 标准化
        ])
    
    def find_closest_aspect_ratio(self,aspect_ratio, target_ratios, width, height, image_size):
        """寻找最接近的目标宽高比"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height  # 原始图像面积
        # 遍历所有候选比例
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            # 选择差异最小的比例
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:  # 相同差异时选择面积更大的
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """动态图像预处理：将图像分割为多个子图"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height  # 原始宽高比
        
        # 生成所有可能的宽高比组合
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # 找到最佳比例并计算目标尺寸
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]  # 总块数
        
        # 调整大小并分割图像
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            # 计算每个子图的坐标
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            processed_images.append(resized_img.crop(box))  # 裁剪子图
            
        # 可选添加缩略图
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def ai_ocr_preprocess(self, image_file, input_size=448, max_num=12):
            """加载并预处理图像"""
            transform = self.build_transform(input_size)
            images = self.dynamic_preprocess(image_file, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(img) for img in images]  # 应用预处理
            return torch.stack(pixel_values)  # 堆叠为张量

    def ocr_img(self, img: np.ndarray) -> str:
        # 对整个图像进行 OCR 处理的逻辑
        # 使用自定义模型进行预测
        # 将NumPy数组转回PIL Image对象

        pil_img = Image.fromarray(img)
        pil_img_rgb = pil_img.convert('RGB')

        # 示例：保存图像
        #pil_img.save("output.jpg")

        try:
            # 准备流式输出
            streamer = TextIteratorStreamer(ocr_tokenizer, timeout=60)
            generation_config = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "streamer": streamer
            }

            pixel_values = self.ai_ocr_preprocess(pil_img_rgb).to(torch.bfloat16).cuda()
            
            # 启动生成线程
            Thread(target = ocr_model.chat, kwargs={
                "tokenizer": ocr_tokenizer,
                "pixel_values": pixel_values,  # 使用处理后的张量
                "question": ("<image>\n" + "请识别图片中的文字, 文字输出时特别注意不要添加额外文字，如（图中所有文字：）之类，直接输出原文文字。"),
                "generation_config": generation_config,
                "return_history": False  # 不再需要返回历史
            }).start()

            # 流式响应
            response = ''
            # Loop through the streamer to get the new text as it is generated
            for token in streamer:
                if token == ocr_model.conv_template.sep:
                    continue
                #print(token, end="\n", flush=True)  # Print each new chunk of generated text on the same line
                if "<|im_end|>" in token:
                    token = token.replace("<|im_end|>", "")
                    if token:  # 如果删除后还有内容，继续发送剩余部分
                        response += token
                    continue
                response += token
            if response == "content" or response == "role":
                response = "图片没有文字"
            print(f"最终response:\n{response}")
            result = response
        except Exception as e:
            print(f"出现错误")
        return result
    


    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        self._load_model()
        # 对文本块列表进行 OCR 处理的逻辑
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                cropped_img = img[y1:y2, x1:x2]
                blk.text = self.ocr_img(cropped_img)
            else:
                self.logger.warning('invalid textbbox to target img')
                blk.text = ['']
        self._unload_ocr_model()  # 添加在方法末尾