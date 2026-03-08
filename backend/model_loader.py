import torch
import cv2
import numpy as np
from PIL import Image
import logging
import os
import io
from config import MODEL_PATH, MODEL_CONFIDENCE_THRESHOLD, SUPPORTED_CLASSES

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv5Detector:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_mapping = {}
        self.load_model()

    def load_model(self):
        """加载本地训练好的YOLOv5模型"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(MODEL_PATH):
                logger.error(f"模型文件不存在: {MODEL_PATH}")
                raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

            logger.info(f"正在加载模型从: {MODEL_PATH}")
            logger.info(f"使用设备: {self.device}")

            # 加载本地模型 - 使用torch.hub.load，但指定本地路径
            # 注意：Render需要联网下载YOLOv5的基础代码，但不会下载其他模型
            self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                        path=MODEL_PATH, force_reload=False)
            self.model.conf = MODEL_CONFIDENCE_THRESHOLD
            self.model.iou = 0.45
            self.model.to(self.device)

            logger.info("模型加载成功！")
            
            # 过滤只保留支持的类别
            self.filter_classes()

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e

    def filter_classes(self):
        """过滤只保留支持的8个类别"""
        # 获取模型原始类别
        original_names = self.model.names

        # 创建类别映射
        self.class_mapping = {}
        for idx, name in original_names.items():
            if name in SUPPORTED_CLASSES:
                self.class_mapping[idx] = name

        logger.info(f"支持的类别: {list(self.class_mapping.values())}")

    def preprocess_image(self, image):
        """预处理图像"""
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
        """执行目标检测"""
        try:
            # 预处理图像
            img = self.preprocess_image(image)

            # 执行推理
            results = self.model(img, size=640)

            # 处理检测结果
            detections = []
            if len(results.xyxy[0]) > 0:
                for *box, conf, cls in results.xyxy[0]:
                    cls = int(cls.item())
                    conf = float(conf.item())

                    # 只返回支持的类别
                    if cls in self.class_mapping:
                        class_name = self.class_mapping[cls]
                        detection = {
                            'class_id': cls,
                            'class_name': class_name,
                            'confidence': conf,
                            'bbox': [float(x.item()) for x in box],
                            'bbox_normalized': [
                                float(x.item()) / img.size[0] if i % 2 == 0 else float(x.item()) / img.size[1]
                                for i, x in enumerate(box)]
                        }
                        detections.append(detection)
                        logger.info(f"检测到: {class_name}, 置信度: {conf:.2f}")

            # 按置信度排序
            detections.sort(key=lambda x: x['confidence'], reverse=True)

            logger.info(f"检测完成，找到 {len(detections)} 个食物")
            return detections

        except Exception as e:
            logger.error(f"检测过程出错: {e}")
            return []

    def detect_batch(self, images):
        """批量检测"""
        results = []
        for image in images:
            detections = self.detect(image)
            results.append(detections)
        return results


# 单例模式
detector = None


def get_detector():
    """获取检测器实例（单例）"""
    global detector
    if detector is None:
        detector = YOLOv5Detector()
    return detector
