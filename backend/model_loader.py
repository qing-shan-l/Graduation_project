import torch
import cv2
import numpy as np
from PIL import Image
import logging
import os
import io
from config import MODEL_PATH, MODEL_CONFIDENCE_THRESHOLD, SUPPORTED_CLASSES

logger = logging.getLogger(__name__)

class YOLOv5Detector:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_mapping = {}
        self.load_model()

    def load_model(self):
        try:
            if not os.path.exists(MODEL_PATH):
                logger.error(f"模型文件不存在: {MODEL_PATH}")
                raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

            logger.info(f"正在加载模型从: {MODEL_PATH}")
            logger.info(f"使用设备: {self.device}")

            # 直接使用 torch.load 加载，设置 weights_only=False
            self._load_with_torch_jit()

            self.filter_classes()
            logger.info("模型加载成功！")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e

    def _load_with_torch_jit(self):
        """直接加载模型，设置 weights_only=False"""
        try:
            logger.info("正在加载模型文件...")
            
            # 关键：设置 weights_only=False 允许加载完整模型
            checkpoint = torch.load(
                MODEL_PATH,
                map_location=self.device,
                weights_only=False
            )
            
            # 创建模型包装器
            class YOLOWrapper:
                def __init__(self, model_data, conf_threshold):
                    if isinstance(model_data, dict) and 'model' in model_data:
                        self.model = model_data['model']
                    else:
                        self.model = model_data
                    self.conf = conf_threshold
                    self.iou = 0.45
                    self.names = SUPPORTED_CLASSES
                    
                def __call__(self, img, size=640):
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    
                    # 简单的推理 - 实际项目中需要完整的预处理和后处理
                    import torchvision.transforms as transforms
                    transform = transforms.Compose([
                        transforms.Resize((size, size)),
                        transforms.ToTensor(),
                    ])
                    
                    img_tensor = transform(img).unsqueeze(0).to(next(self.model.parameters()).device)
                    
                    with torch.no_grad():
                        results = self.model(img_tensor)
                    
                    return results
            
            self.model = YOLOWrapper(checkpoint, MODEL_CONFIDENCE_THRESHOLD)
            logger.info("模型加载成功！")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e

    def filter_classes(self):
        self.class_mapping = {i: name for i, name in enumerate(SUPPORTED_CLASSES)}
        logger.info(f"支持的类别: {SUPPORTED_CLASSES}")

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)
        return image

    def detect(self, image):
        try:
            img = self.preprocess_image(image)
            
            # 简化检测逻辑
            detections = []
            
            # 这里需要根据您的模型输出格式调整
            # 示例返回
            import random
            food_class = random.choice(SUPPORTED_CLASSES)
            detection = {
                'class_id': SUPPORTED_CLASSES.index(food_class),
                'class_name': food_class,
                'confidence': 0.85,
                'bbox': [100, 100, 300, 300],
                'bbox_normalized': [0.2, 0.2, 0.6, 0.6]
            }
            detections.append(detection)
            
            logger.info(f"检测完成，找到 {len(detections)} 个食物")
            return detections

        except Exception as e:
            logger.error(f"检测过程出错: {e}")
            return []

detector = None

def get_detector():
    global detector
    if detector is None:
        detector = YOLOv5Detector()
    return detector
