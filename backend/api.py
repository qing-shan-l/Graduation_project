import sys
import os

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import io
import json
import logging
from datetime import datetime

# 现在可以正常导入本地模块
from model_loader import get_detector
from nutrition_advisor import get_advisor
from config import HOST, PORT, SUPPORTED_CLASSES, CLASS_CHINESE_NAMES
from helpers import compress_image, validate_image, generate_filename, image_to_base64

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="餐食识别与饮食建议API",
    description="基于YOLOv5的餐食识别与营养建议系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
detector = None
advisor = None


# 启动时加载模型和营养顾问
@app.on_event("startup")
async def startup_event():
    logger.info("正在启动餐食识别服务...")
    try:
        # 初始化检测器
        global detector, advisor
        detector = get_detector()
        advisor = get_advisor()
        logger.info("服务启动成功！")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise e


# 健康检查
@app.get("/")
async def root():
    return {
        "service": "餐食识别与饮食建议系统",
        "version": "1.0.0",
        "status": "running",
        "supported_classes": SUPPORTED_CLASSES,
        "supported_classes_zh": CLASS_CHINESE_NAMES,
        "timestamp": datetime.now().isoformat()
    }


# 获取支持的食物列表
@app.get("/foods")
async def get_foods():
    """获取支持的食物列表"""
    foods = []
    for food_class in SUPPORTED_CLASSES:
        foods.append({
            "class_name": food_class,
            "name_zh": CLASS_CHINESE_NAMES.get(food_class, food_class)
        })
    return {
        "count": len(foods),
        "foods": foods
    }


# 单张图片识别
@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        return_image: bool = Form(False),
        compress: bool = Form(True)
):
    """
    餐食识别接口

    - **file**: 上传的图片文件（支持jpg/png）
    - **return_image**: 是否返回标注后的图片
    - **compress**: 是否压缩图片
    """
    try:
        # 读取文件
        contents = await file.read()

        # 验证图片
        is_valid, message = validate_image(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)

        # 压缩图片（可选）
        if compress:
            contents = compress_image(contents)

        # 执行检测
        global detector
        if detector is None:
            detector = get_detector()

        detections = detector.detect(contents)

        # 记录检测结果
        logger.info(f"检测到 {len(detections)} 个食物: {detections}")

        response = {
            "success": True,
            "filename": file.filename,
            "detections": detections,
            "count": len(detections)
        }

        # 如果需要返回标注图片
        if return_image and detections:
            # TODO: 添加标注功能
            pass

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


# 获取营养建议
@app.post("/analyze")
async def analyze_meal(
        file: UploadFile = File(...),
        user_profile: Optional[str] = Form(None)
):
    """
    餐食分析与营养建议接口

    - **file**: 上传的图片文件
    - **user_profile**: 用户个人信息（JSON格式）
    """
    try:
        # 读取文件
        contents = await file.read()

        # 验证图片
        is_valid, message = validate_image(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)

        # 压缩图片
        contents = compress_image(contents)

        # 执行检测
        global detector
        if detector is None:
            detector = get_detector()

        detections = detector.detect(contents)
        logger.info(f"分析检测到 {len(detections)} 个食物")

        # 解析用户信息
        user_data = None
        if user_profile:
            try:
                user_data = json.loads(user_profile)
            except:
                pass

        # 生成营养建议
        global advisor
        if advisor is None:
            advisor = get_advisor()

        nutrition_advice = advisor.generate_advice(detections, user_data)

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "detections": detections,
            "nutrition_advice": nutrition_advice
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


# 批量识别
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """批量识别多张图片"""
    results = []

    for file in files:
        try:
            contents = await file.read()
            is_valid, _ = validate_image(contents)

            if is_valid:
                contents = compress_image(contents)
                global detector
                if detector is None:
                    detector = get_detector()
                detections = detector.detect(contents)

                results.append({
                    "filename": file.filename,
                    "success": True,
                    "detections": detections,
                    "count": len(detections)
                })
            else:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "无效的图片格式"
                })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return JSONResponse(content={
        "total": len(files),
        "success_count": sum(1 for r in results if r["success"]),
        "results": results
    })


# 获取营养信息
@app.get("/nutrition/{food_class}")
async def get_food_nutrition(food_class: str):
    """获取特定食物的营养信息"""
    if food_class not in SUPPORTED_CLASSES:
        raise HTTPException(status_code=404, detail="不支持的食物类别")

    global advisor
    if advisor is None:
        advisor = get_advisor()

    nutrition = advisor.get_nutrition_info(food_class)

    if nutrition:
        return nutrition
    else:
        raise HTTPException(status_code=404, detail="未找到营养信息")


# 获取营养建议（无图片）
@app.post("/advice")
async def get_nutrition_advice(
        foods: List[str],
        quantities: Optional[List[float]] = None
):
    """
    根据食物列表获取营养建议

    - **foods**: 食物类别列表
    - **quantities**: 对应食物的份数（默认1份）
    """
    if not foods:
        raise HTTPException(status_code=400, detail="请提供食物列表")

    # 验证食物类别
    invalid_foods = [f for f in foods if f not in SUPPORTED_CLASSES]
    if invalid_foods:
        raise HTTPException(status_code=400, detail=f"不支持的食物: {invalid_foods}")

    # 处理份数
    if quantities is None:
        quantities = [1.0] * len(foods)
    elif len(quantities) != len(foods):
        raise HTTPException(status_code=400, detail="食物与份数数量不匹配")

    # 构建检测结果格式
    detections = []
    for food, quantity in zip(foods, quantities):
        detections.append({
            "class_name": food,
            "confidence": 1.0,
            "quantity": quantity
        })

    # 生成建议
    global advisor
    if advisor is None:
        advisor = get_advisor()

    advice = advisor.generate_advice(detections)

    return advice


# 获取服务器IP地址接口
@app.get("/server-info")
async def get_server_info():
    """获取服务器信息，包括IP地址"""
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    # 尝试获取更合适的本地IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        pass

    return {
        "hostname": hostname,
        "ip": local_ip,
        "port": PORT,
        "url": f"http://{local_ip}:{PORT}"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, reload=True)