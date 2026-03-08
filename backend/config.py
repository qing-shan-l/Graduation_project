import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'best.pt')  # 修改路径
NUTRITION_DB_PATH = os.path.join(BASE_DIR, 'nutrition_db.json')

# 模型配置
MODEL_CONFIDENCE_THRESHOLD = 0.25
MODEL_IMG_SIZE = 640

# 服务器配置 - 修改为支持Render
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))  # Render会提供PORT环境变量


# 支持的食品类别（8类）
SUPPORTED_CLASSES = [
    'dumplings',      # 饺子/包子
    'french_fries',   # 薯条
    'fried_rice',     # 炒饭
    'hamburger',      # 汉堡
    'ice_cream',      # 冰淇淋
    'peking_duck',    # 北京烤鸭
    'steak',          # 牛排
    'sushi'           # 寿司
]

# 中文映射
CLASS_CHINESE_NAMES = {
    'dumplings': '饺子/包子',
    'french_fries': '薯条',
    'fried_rice': '炒饭',
    'hamburger': '汉堡',
    'ice_cream': '冰淇淋',
    'peking_duck': '北京烤鸭',
    'steak': '牛排',
    'sushi': '寿司'
}