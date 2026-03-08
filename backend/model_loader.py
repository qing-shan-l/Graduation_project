import torch
import cv2
import numpy as np
from PIL import Image
import logging
import os
import io
import sys
import pickle
from pathlib import Path, PosixPath, WindowsPath
from config import MODEL_PATH, MODEL_CONFIDENCE_THRESHOLD, SUPPORTED_CLASSES

logger = logging.getLogger(__name__)


class PathFixer(pickle.Unpickler):
    """自定义Unpickler，处理WindowsPath到PosixPath的转换"""
    def find_class(self, module, name):
        if module == 'pathlib' and name == 'WindowsPath':
            return PosixPath
        return super().find_class(module, name)


class YOLOv5Detector:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_mapping = {}
        self.load_model()

    def load_model(self):
        """加载YOLOv5模型，处理跨平台路径兼容性"""
        try:
            if not os.path.exists(MODEL_PATH):
                logger.error(f"模型文件不存在: {MODEL_PATH}")
                raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

            logger.info(f"正在加载模型从: {MODEL_PATH}")
            logger.info(f"使用设备: {self.device}")

            # 方法1: 使用torch.hub.load（最可靠）
            try:
                logger.info("尝试使用torch.hub.load加载模型...")
                
                # 设置环境变量
                os.environ['YOLO_VERBOSE'] = 'False'
                
                self.model = torch.hub.load(
                    'ultralytics/yolov5:master',
                    'custom',
                    path=MODEL_PATH,
                    force_reload=False,
                    trust_repo=True,
                    skip_validation=True
                )
                
                self.model.conf = MODEL_CONFIDENCE_THRESHOLD
                self.model.iou = 0.45
                self.model.to(self.device)
                
                logger.info("torch.hub.load加载成功！")
                
            except Exception as e:
                logger.error(f"torch.hub.load失败: {e}")
                
                # 方法2: 使用自定义unpickler处理路径兼容性
                logger.info("尝试使用自定义unpickler加载模型...")
                
                # 使用自定义unpickler加载
                with open(MODEL_PATH, 'rb') as f:
                    checkpoint = PathFixer(f).load()
                
                # 设置YOLOv5路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                yolov5_path = os.path.join(project_root, 'yolov5')
                
                if yolov5_path not in sys.path:
                    sys.path.insert(0, yolov5_path)
                
                # 导入YOLOv5模块
                from models.common import DetectMultiBackend, AutoShape
                from utils.torch_utils import select_device
                
                # 创建模型包装器
                class ModelWrapper:
                    def __init__(self, checkpoint, device, conf):
                        self.model = checkpoint
                        self.conf = conf
                        self.iou = 0.45
                        self.device = device
                        self.names = SUPPORTED_CLASSES
                        
                    def __call__(self, img, size=640):
                        # 简单的推理包装
                        import torchvision.transforms as transforms
                        transform = transforms.Compose([
                            transforms.Resize((size, size)),
                            transforms.ToTensor(),
                        ])
                        
                        img_tensor = transform(img).unsqueeze(0)
                        
                        with torch.no_grad():
                            if hasattr(self.model, 'eval'):
                                self.model.eval()
                            results = self.model(img_tensor)
                        
                        return results
                
                self.model = ModelWrapper(checkpoint, self.device, MODEL_CONFIDENCE_THRESHOLD)
                logger.info("自定义unpickler加载成功！")

            # 过滤类别
            self.filter_classes()
            logger.info("模型加载完成！")

        except Exception as e:
            logger.error(f"模型加载最终失败: {e}")
            raise e

    def filter_classes(self):
        """过滤类别"""
        try:
            if hasattr(self.model, 'names'):
                original_names = self.model.names
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                original_names = self.model.model.names
            else:
                logger.warning("无法获取类别名称，使用默认映射")
                self.class_mapping = {i: name for i, name in enumerate(SUPPORTED_CLASSES)}
                return
            
            self.class_mapping = {}
            for idx, name in original_names.items():
                if name in SUPPORTED_CLASSES:
                    self.class_mapping[idx] = name
            
            logger.info(f"支持的类别: {list(self.class_mapping.values())}")
            
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
        elif isinstance(image, str) or isinstance(image, Path):
            image = Image.open(str(image))
        return image

    def detect(self, image):
        """执行检测"""
        try:
            img = self.preprocess_image(image)
            results = self.model(img, size=640)

            detections = []
            
            if hasattr(results, 'pandas') and hasattr(results.pandas(), 'xyxy'):
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    class_name = row['name']
                    confidence = row['confidence']
                    
                    if class_name in SUPPORTED_CLASSES:
                        detection = {
                            'class_id': SUPPORTED_CLASSES.index(class_name),
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'bbox': [
                                float(row['xmin']), float(row['ymin']), 
                                float(row['xmax']), float(row['ymax'])
                            ],
                            'bbox_normalized': [
                                float(row['xmin']) / img.size[0],
                                float(row['ymin']) / img.size[1],
                                float(row['xmax']) / img.size[0],
                                float(row['ymax']) / img.size[1]
                            ]
                        }
                        detections.append(detection)
                        
            elif hasattr(results, 'xyxy') and len(results.xyxy[0]) > 0:
                for *box, conf, cls in results.xyxy[0]:
                    cls = int(cls.item())
                    conf = float(conf.item())

                    if cls in self.class_mapping:
                        class_name = self.class_mapping[cls]
                        detection = {
                            'class_id': cls,
                            'class_name': class_name,
                            'confidence': conf,
                            'bbox': [float(x.item()) for x in box],
                            'bbox_normalized': [
                                float(x.item()) / img.size[0] if i % 2 == 0 else float(x.item()) / img.size[1]
                                for i, x in enumerate(box)
                            ]
                        }
                        detections.append(detection)

            detections.sort(key=lambda x: x['confidence'], reverse=True)
            logger.info(f"检测完成，找到 {len(detections)} 个食物")
            return detections

        except Exception as e:
            logger.error(f"检测过程出错: {e}")
            return []


# 单例模式
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = YOLOv5Detector()
    return _detector
