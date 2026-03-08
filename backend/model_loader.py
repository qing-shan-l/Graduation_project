import torch
import cv2
import numpy as np
from PIL import Image
import logging
import os
import io
import sys
from pathlib import Path
from config import MODEL_PATH, MODEL_CONFIDENCE_THRESHOLD, SUPPORTED_CLASSES

logger = logging.getLogger(__name__)


class YOLOv5Detector:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_mapping = {}
        self.load_model()

    def load_model(self):
        """加载YOLOv5模型"""
        try:
            # 检查模型文件
            if not os.path.exists(MODEL_PATH):
                logger.error(f"模型文件不存在: {MODEL_PATH}")
                raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

            logger.info(f"正在加载模型从: {MODEL_PATH}")
            logger.info(f"使用设备: {self.device}")

            # 获取项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            
            # 设置YOLOv5路径
            yolov5_path = os.path.join(project_root, 'yolov5')
            logger.info(f"YOLOv5路径: {yolov5_path}")
            
            # 将YOLOv5路径添加到sys.path
            if os.path.exists(yolov5_path):
                if yolov5_path not in sys.path:
                    sys.path.insert(0, yolov5_path)
                    logger.info(f"已添加YOLOv5路径到sys.path")
            else:
                logger.error(f"YOLOv5目录不存在: {yolov5_path}")
                raise FileNotFoundError(f"YOLOv5目录不存在")
            
            # 方法1: 使用torch.hub.load（最可靠）
            logger.info("尝试使用torch.hub.load加载模型...")
            try:
                # 先设置环境变量，避免下载
                os.environ['TORCH_HUB'] = yolov5_path
                
                self.model = torch.hub.load(
                    'ultralytics/yolov5:master',
                    'custom',
                    path=MODEL_PATH,
                    force_reload=False,
                    trust_repo=True,
                    skip_validation=True  # 跳过验证
                )
                self.model.conf = MODEL_CONFIDENCE_THRESHOLD
                self.model.iou = 0.45
                self.model.to(self.device)
                
                logger.info("torch.hub.load加载成功！")
                
            except Exception as e:
                logger.error(f"torch.hub.load失败: {e}")
                
                # 方法2: 尝试直接导入本地模块
                logger.info("尝试使用本地模块加载...")
                try:
                    # 导入YOLOv5模块
                    import models
                    from models.common import DetectMultiBackend, AutoShape
                    from utils.general import check_img_size, non_max_suppression, scale_boxes
                    from utils.torch_utils import select_device
                    
                    # 加载模型
                    self.model = DetectMultiBackend(
                        weights=MODEL_PATH,
                        device=select_device(self.device),
                        dnn=False,
                        data=None,
                        fp16=False
                    )
                    
                    # 设置参数
                    self.model.conf = MODEL_CONFIDENCE_THRESHOLD
                    self.model.iou = 0.45
                    
                    # 包装为AutoShape
                    self.model = AutoShape(self.model)
                    
                    logger.info("本地模块加载成功！")
                    
                except ImportError as e2:
                    logger.error(f"本地模块加载失败: {e2}")
                    
                    # 方法3: 最后的备选方案
                    logger.info("尝试使用备选加载方式...")
                    try:
                        # 直接添加子目录
                        models_path = os.path.join(yolov5_path, 'models')
                        utils_path = os.path.join(yolov5_path, 'utils')
                        
                        if models_path not in sys.path:
                            sys.path.insert(0, models_path)
                        if utils_path not in sys.path:
                            sys.path.insert(0, utils_path)
                        
                        from common import DetectMultiBackend, AutoShape
                        from general import non_max_suppression, scale_boxes
                        
                        self.model = DetectMultiBackend(
                            weights=MODEL_PATH,
                            device=self.device,
                            dnn=False,
                            data=None,
                            fp16=False
                        )
                        
                        self.model.conf = MODEL_CONFIDENCE_THRESHOLD
                        self.model.iou = 0.45
                        self.model = AutoShape(self.model)
                        
                        logger.info("备选加载成功！")
                        
                    except Exception as e3:
                        logger.error(f"所有加载方法都失败: {e3}")
                        raise e3

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
