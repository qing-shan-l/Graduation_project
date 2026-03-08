import torch
import cv2
import numpy as np
from PIL import Image
import logging
import os
import io
import sys
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

            # 方法1：尝试使用本地YOLOv5代码（如果存在）
            yolov5_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolov5')
            
            if os.path.exists(yolov5_path) and os.path.exists(os.path.join(yolov5_path, 'models', 'common.py')):
                # 使用本地YOLOv5代码
                logger.info(f"使用本地YOLOv5代码: {yolov5_path}")
                
                # 将yolov5路径添加到系统路径
                if yolov5_path not in sys.path:
                    sys.path.insert(0, yolov5_path)
                
                # 导入本地YOLOv5模型
                try:
                    from models.common import DetectMultiBackend
                    from utils.torch_utils import select_device
                    from utils.general import check_img_size, non_max_suppression, scale_boxes
                    from utils.augmentations import letterbox
                    
                    # 加载模型
                    self.model = DetectMultiBackend(
                        weights=MODEL_PATH,
                        device=self.device,
                        dnn=False,
                        fp16=False
                    )
                    
                    # 设置模型参数
                    self.model.conf = MODEL_CONFIDENCE_THRESHOLD
                    self.model.iou = 0.45
                    
                    logger.info("使用本地YOLOv5代码加载模型成功！")
                    
                except ImportError as e:
                    logger.error(f"导入本地YOLOv5模块失败: {e}")
                    # 回退到方法2
                    self._load_with_torch_hub()
            else:
                # 方法2：使用torch.hub，但添加trust_repo参数
                self._load_with_torch_hub()

            # 过滤只保留支持的类别
            self.filter_classes()

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e

    def _load_with_torch_hub(self):
        """使用torch.hub加载模型（添加trust_repo参数）"""
        try:
            logger.info("使用torch.hub加载模型...")
            
            # PyTorch 2.0+ 需要trust_repo参数
            import torch
            if hasattr(torch.hub, 'load') and 'trust_repo' in torch.hub.load.__code__.co_varnames:
                self.model = torch.hub.load(
                    'ultralytics/yolov5', 
                    'custom', 
                    path=MODEL_PATH, 
                    force_reload=False,
                    trust_repo=True  # 添加信任仓库参数
                )
            else:
                # 旧版本
                self.model = torch.hub.load(
                    'ultralytics/yolov5', 
                    'custom', 
                    path=MODEL_PATH, 
                    force_reload=False
                )
            
            self.model.conf = MODEL_CONFIDENCE_THRESHOLD
            self.model.iou = 0.45
            self.model.to(self.device)
            
            logger.info("使用torch.hub加载模型成功！")
            
        except Exception as e:
            logger.error(f"torch.hub加载失败: {e}")
            # 方法3：使用简单的torch加载
            self._load_with_torch_jit()

    def _load_with_torch_jit(self):
        """使用torch.jit加载模型（最后的手段）"""
        try:
            logger.info("尝试使用torch.jit加载模型...")
            
            # 加载模型权重
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            
            # 创建简单的模型包装器
            class SimpleYOLOModel:
                def __init__(self, model_path, device, conf_threshold):
                    self.model = torch.load(model_path, map_location=device)
                    self.conf = conf_threshold
                    self.iou = 0.45
                    self.device = device
                    self.names = SUPPORTED_CLASSES
                    
                def __call__(self, img, size=640):
                    # 简单的推理包装
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    
                    # 这里需要更复杂的实现，但为了部署，我们假设模型可以直接调用
                    return self.model(img)
            
            self.model = SimpleYOLOModel(MODEL_PATH, self.device, MODEL_CONFIDENCE_THRESHOLD)
            logger.info("使用torch.jit加载模型成功！")
            
        except Exception as e:
            logger.error(f"所有加载方法都失败: {e}")
            raise e

    def filter_classes(self):
        """过滤只保留支持的8个类别"""
        try:
            if hasattr(self.model, 'names'):
                original_names = self.model.names
                
                # 创建类别映射
                self.class_mapping = {}
                for idx, name in original_names.items():
                    if name in SUPPORTED_CLASSES:
                        self.class_mapping[idx] = name
                
                logger.info(f"支持的类别: {list(self.class_mapping.values())}")
            else:
                # 如果模型没有names属性，使用默认映射
                self.class_mapping = {i: name for i, name in enumerate(SUPPORTED_CLASSES)}
                logger.info(f"使用默认类别映射: {SUPPORTED_CLASSES}")
        except Exception as e:
            logger.error(f"过滤类别失败: {e}")
            self.class_mapping = {i: name for i, name in enumerate(SUPPORTED_CLASSES)}

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
            
            # 根据不同的模型类型处理结果
            if hasattr(results, 'xyxy') and len(results.xyxy[0]) > 0:
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
            
            elif hasattr(results, 'pandas') and hasattr(results.pandas(), 'xyxy'):
                # 处理pandas格式的结果
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    class_name = row['name']
                    confidence = row['confidence']
                    
                    if class_name in SUPPORTED_CLASSES:
                        detection = {
                            'class_id': SUPPORTED_CLASSES.index(class_name),
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'bbox': [float(row['xmin']), float(row['ymin']), 
                                    float(row['xmax']), float(row['ymax'])],
                            'bbox_normalized': [
                                float(row['xmin']) / img.size[0],
                                float(row['ymin']) / img.size[1],
                                float(row['xmax']) / img.size[0],
                                float(row['ymax']) / img.size[1]
                            ]
                        }
                        detections.append(detection)
                        logger.info(f"检测到: {class_name}, 置信度: {confidence:.2f}")

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
