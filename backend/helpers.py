import os
import uuid
from datetime import datetime
from PIL import Image
import io
import base64


def generate_filename(extension='jpg'):
    """生成唯一文件名"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}.{extension}"


def compress_image(image_bytes, max_size=(640, 640), quality=85):
    """压缩图像"""
    img = Image.open(io.BytesIO(image_bytes))

    # 调整尺寸
    img.thumbnail(max_size, Image.Resampling.LANCZOS)

    # 保存到字节流
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=quality)
    return output.getvalue()


def image_to_base64(image_bytes):
    """图像转base64"""
    return base64.b64encode(image_bytes).decode('utf-8')


def validate_image(file_bytes):
    """验证图像文件"""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        # 检查格式
        if img.format not in ['JPEG', 'PNG', 'JPG']:
            return False, "不支持的图片格式，请上传JPG或PNG格式"
        # 检查大小（不超过10MB）
        if len(file_bytes) > 10 * 1024 * 1024:
            return False, "图片过大，请上传小于10MB的图片"
        return True, "验证通过"
    except Exception as e:
        return False, f"图片验证失败: {str(e)}"