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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv5Detector:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_mapping = {}
        self.load_model()

    def _setup_yolov5_path(self):
        """设置YOLOv5路径并导入必要模块"""
        try:
            # 获取项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)  # 上一级是项目根目录
            
            # YOLOv5路径
            yolov5_path = os.path.join(project_root, 'yolov5')
            
            logger.info(f"当前目录: {current_dir}")
            logger.info(f"项目根目录: {project_root}")
            logger.info(f"YOLOv5路径: {yolov5_path}")
            
            # 检查yolov5目录是否存在
            if not os.path.exists(yolov5_path):
                logger.error(f"YOLOv5目录不存在: {yolov5_path}")
                # 列出项目根目录内容以便调试
                try:
                    contents = os.listdir(project_root)
                    logger.info(f"项目根目录内容: {contents}")
                except:
                    pass
                raise FileNotFoundError(f"YOLOv5目录不存在: {yolov5_path}")
            
            # 检查models子目录是否存在
            models_path = os.path.join(yolov5_path, 'models')
            if not os.path.exists(models_path):
                logger.error(f"models目录不存在: {models_path}")
                # 列出yolov5目录内容
                try:
                    contents = os.listdir(yolov5_path)
                    logger.info(f"yolov5目录内容: {contents}")
                except:
                    pass
                raise FileNotFoundError(f"models目录不存在: {models_path}")
            
            # 将YOLOv5路径添加到系统路径（必须放在最前面）
            if yolov5_path not in sys.path:
                sys.path.insert(0, yolov5_path)
                logger.info(f"已添加YOLOv5路径到sys.path: {yolov5_path}")
            
            # 打印当前sys.path以便调试
            logger.debug(f"当前sys.path: {sys.path}")
            
            # 尝试导入models模块
            try:
                import models
                logger.info(f"成功导入models模块，位置: {models.__file__}")
                
                # 验证关键子模块
                from models.common import DetectMultiBackend, AutoShape
                from utils.general import check_img_size, non_max_suppression, scale_boxes
                from utils.torch_utils import select_device
                from utils.augmentations import letterbox
                
                logger.info("成功导入YOLOv5所有必要模块")
                
                # 返回需要的类和函数
                return {
                    'DetectMultiBackend': DetectMultiBackend,
                    'AutoShape': AutoShape,
                    'check_img_size': check_img_size,
                    'non_max_suppression': non_max_suppression,
                    'scale_boxes': scale_boxes,
                    'select_device': select_device,
                    'letterbox': letterbox
                }
                
            except ImportError as e:
                logger.error(f"导入YOLOv5模块失败: {e}")
                # 尝试直接导入特定模块作为备选
                try:
                    # 直接添加models路径
                    if models_path not in sys.path:
                        sys.path.insert(0, models_path)
                    
                    from common import DetectMultiBackend, AutoShape
                    
                    # 添加utils路径
                    utils_path = os.path.join(yolov5_path, 'utils')
                    if utils_path not in sys.path:
                        sys.path.insert(0, utils_path)
                    
                    from general import check_img_size, non_max_suppression, scale_boxes
                    from torch_utils import select_device
                    from augmentations import letterbox
                    
                    logger.info("通过直接路径导入YOLOv5模块成功")
                    
                    return {
                        'DetectMultiBackend': DetectMultiBackend,
                        'AutoShape': AutoShape,
                        'check_img_size': check_img_size,
                        'non_max_suppression': non_max_suppression,
                        'scale_boxes': scale_boxes,
                        'select_device': select_device,
                        'letterbox': letterbox
                    }
                except ImportError as e2:
                    logger.error(f"备选导入也失败: {e2}")
                    raise e
                
        except Exception as e:
            logger.error(f"设置YOLOv5路径失败: {e}")
            raise e

    def load_model(self):
        """加载本地训练好的YOLOv5模型"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(MODEL_PATH):
                logger.error(f"模型文件不存在: {MODEL_PATH}")
                raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

            logger.info(f"正在加载模型从: {MODEL_PATH}")
            logger.info(f"使用设备: {self.device}")
            
            # 设置YOLOv5路径并获取模块
            yolov5_modules = self._setup_yolov5_path()
            
            # 使用YOLOv5的DetectMultiBackend加载模型
            DetectMultiBackend = yolov5_modules['DetectMultiBackend']
            select_device = yolov5_modules['select_device']
            
            # 加载模型
            self.model = DetectMultiBackend(
                weights=MODEL_PATH,
                device=select_device(self.device),
                dnn=False,
                data=None,
                fp16=False
            )
            
            # 设置模型参数
            self.model.conf = MODEL_CONFIDENCE_THRESHOLD
            self.model.iou = 0.45
            self.model.classes = None  # 所有类别
            self.model.max_det = 1000  # 最大检测数
            
            # 包装为AutoShape以便直接处理图像
            AutoShape = yolov5_modules['AutoShape']
            self.model = AutoShape(self.model)
            
            logger.info("模型加载成功！")
            
            # 过滤只保留支持的类别
            self.filter_classes()

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e

    def filter_classes(self):
        """过滤只保留支持的8个类别"""
        try:
            # 获取模型类别名称
            if hasattr(self.model, 'names'):
                original_names = self.model.names
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                original_names = self.model.model.names
            else:
                # 如果无法获取，使用默认映射
                logger.warning("无法从模型获取类别名称，使用默认映射")
                self.class_mapping = {i: name for i, name in enumerate(SUPPORTED_CLASSES)}
                logger.info(f"使用默认类别映射: {SUPPORTED_CLASSES}")
                return
            
            # 创建类别映射
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
            # 如果是OpenCV图像（BGR），转换为RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(image, bytes):
            # 如果是字节数据
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            # 如果是文件路径
            image = Image.open(image)
        elif isinstance(image, Path):
            image = Image.open(str(image))

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
            
            # YOLOv5 AutoShape返回的结果处理
            if hasattr(results, 'pandas') and hasattr(results.pandas(), 'xyxy'):
                # pandas格式的结果
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    class_name = row['name']
                    confidence = row['confidence']
                    
                    # 检查是否是支持的类别
                    if class_name in SUPPORTED_CLASSES:
                        detection = {
                            'class_id': SUPPORTED_CLASSES.index(class_name),
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'bbox': [
                                float(row['xmin']), 
                                float(row['ymin']), 
                                float(row['xmax']), 
                                float(row['ymax'])
                            ],
                            'bbox_normalized': [
                                float(row['xmin']) / img.size[0],
                                float(row['ymin']) / img.size[1],
                                float(row['xmax']) / img.size[0],
                                float(row['ymax']) / img.size[1]
                            ]
                        }
                        detections.append(detection)
                        logger.info(f"检测到: {class_name}, 置信度: {confidence:.2f}")
                        
            elif hasattr(results, 'xyxy') and len(results.xyxy[0]) > 0:
                # tensor格式的结果
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
                                for i, x in enumerate(box)
                            ]
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
_detector = None


def get_detector():
    """获取检测器实例（单例）"""
    global _detector
    if _detector is None:
        _detector = YOLOv5Detector()
    return _detector
