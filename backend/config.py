import os

# 基础路径配置 - 适配Render环境
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径 - 指向项目根目录下的model文件夹
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'model', 'best.pt')

# 营养数据库路径
NUTRITION_DB_PATH = os.path.join(BASE_DIR, 'nutrition_db.json')

# 模型配置
MODEL_CONFIDENCE_THRESHOLD = 0.25
MODEL_IMG_SIZE = 640

# 服务器配置 - Render会自动提供PORT环境变量
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))

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
