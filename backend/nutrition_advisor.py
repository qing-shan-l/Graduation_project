import json
import logging
from typing import Dict, List, Optional
from config import NUTRITION_DB_PATH, CLASS_CHINESE_NAMES, SUPPORTED_CLASSES

logger = logging.getLogger(__name__)


class NutritionAdvisor:
    def __init__(self):
        self.nutrition_db = self.load_nutrition_db()
        self.chinese_names = CLASS_CHINESE_NAMES
        # 添加食物名称映射
        self.food_name_mapping = {
            'dumplings': '饺子/包子',
            'french_fries': '薯条',
            'fried_rice': '炒饭',
            'hamburger': '汉堡',
            'ice_cream': '冰淇淋',
            'peking_duck': '北京烤鸭',
            'steak': '牛排',
            'sushi': '寿司'
        }

    def load_nutrition_db(self) -> Dict:
        """加载营养数据库"""
        try:
            with open(NUTRITION_DB_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("营养数据库文件不存在，使用默认数据")
            return self.get_default_nutrition_db()

    def get_default_nutrition_db(self) -> Dict:
        """获取默认营养数据"""
        return {
            "dumplings": {
                "name_zh": "饺子/包子",
                "calories_per_100g": 220,
                "protein_per_100g": 8.5,
                "fat_per_100g": 10.2,
                "carbs_per_100g": 24.3,
                "fiber_per_100g": 2.1,
                "serving_size_g": 150,
                "description": "传统中式主食，馅料多样，建议适量食用"
            },
            "french_fries": {
                "name_zh": "薯条",
                "calories_per_100g": 312,
                "protein_per_100g": 3.4,
                "fat_per_100g": 15.5,
                "carbs_per_100g": 41.0,
                "fiber_per_100g": 3.8,
                "serving_size_g": 100,
                "description": "高热量油炸食品，建议控制食用量"
            },
            "fried_rice": {
                "name_zh": "炒饭",
                "calories_per_100g": 168,
                "protein_per_100g": 5.1,
                "fat_per_100g": 5.8,
                "carbs_per_100g": 24.2,
                "fiber_per_100g": 1.2,
                "serving_size_g": 200,
                "description": "主食类，建议搭配蔬菜食用"
            },
            "hamburger": {
                "name_zh": "汉堡",
                "calories_per_100g": 250,
                "protein_per_100g": 12.0,
                "fat_per_100g": 10.0,
                "carbs_per_100g": 28.0,
                "fiber_per_100g": 2.0,
                "serving_size_g": 200,
                "description": "快餐食品，建议选择全麦面包版本"
            },
            "ice_cream": {
                "name_zh": "冰淇淋",
                "calories_per_100g": 207,
                "protein_per_100g": 3.5,
                "fat_per_100g": 11.0,
                "carbs_per_100g": 23.6,
                "fiber_per_100g": 0.7,
                "serving_size_g": 100,
                "description": "高糖甜品，建议适量食用"
            },
            "peking_duck": {
                "name_zh": "北京烤鸭",
                "calories_per_100g": 337,
                "protein_per_100g": 18.5,
                "fat_per_100g": 28.0,
                "carbs_per_100g": 3.5,
                "fiber_per_100g": 0.5,
                "serving_size_g": 150,
                "description": "传统名菜，皮脆肉嫩，建议搭配薄饼和蔬菜"
            },
            "steak": {
                "name_zh": "牛排",
                "calories_per_100g": 271,
                "protein_per_100g": 26.0,
                "fat_per_100g": 19.0,
                "carbs_per_100g": 0.0,
                "fiber_per_100g": 0.0,
                "serving_size_g": 200,
                "description": "优质蛋白来源，建议选择瘦肉部分"
            },
            "sushi": {
                "name_zh": "寿司",
                "calories_per_100g": 150,
                "protein_per_100g": 6.0,
                "fat_per_100g": 2.0,
                "carbs_per_100g": 28.0,
                "fiber_per_100g": 1.5,
                "serving_size_g": 150,
                "description": "日式传统美食，富含omega-3脂肪酸"
            }
        }

    def get_nutrition_info(self, food_class: str, quantity: float = 1.0) -> Dict:
        """获取营养信息"""
        if food_class not in self.nutrition_db:
            return None

        food_data = self.nutrition_db[food_class]
        serving_size = food_data.get('serving_size_g', 100)

        # 计算实际摄入量（基于份数）
        actual_serving_g = serving_size * quantity

        # 计算营养成分（四舍五入保留整数）
        nutrition = {
            'food_class': food_class,
            'food_name_zh': food_data['name_zh'],
            'quantity': quantity,
            'serving_size_g': serving_size,
            'actual_serving_g': round(actual_serving_g, 1),
            'calories': round(food_data['calories_per_100g'] * actual_serving_g / 100),
            'protein': round(food_data['protein_per_100g'] * actual_serving_g / 100, 1),
            'fat': round(food_data['fat_per_100g'] * actual_serving_g / 100, 1),
            'carbs': round(food_data['carbs_per_100g'] * actual_serving_g / 100, 1),
            'fiber': round(food_data.get('fiber_per_100g', 0) * actual_serving_g / 100, 1),
            'description': food_data.get('description', '')
        }

        return nutrition

    def generate_advice(self, detections: List[Dict], user_profile: Optional[Dict] = None) -> Dict:
        """生成营养建议"""
        if not detections or len(detections) == 0:
            return {
                'total_nutrition': {
                    'calories': 0,
                    'protein': 0,
                    'fat': 0,
                    'carbs': 0,
                    'fiber': 0
                },
                'foods': [],
                'advice': '未检测到餐食，请重新上传图片',
                'suggestions': ['请确保图片清晰，食物在图片中央', '尝试调整拍摄角度和光线']
            }

        # 获取每个检测到的食物的营养信息
        foods_info = []
        total_nutrition = {
            'calories': 0,
            'protein': 0,
            'fat': 0,
            'carbs': 0,
            'fiber': 0
        }

        for det in detections:
            food_class = det['class_name']
            confidence = det.get('confidence', 0.8)

            # 转换食物名称为中文
            food_name_zh = self.food_name_mapping.get(food_class, food_class)

            # 假设每个检测到的食物为1份
            nutrition = self.get_nutrition_info(food_class, quantity=1.0)
            if nutrition:
                foods_info.append({
                    'food_class': food_class,
                    'food_name_zh': food_name_zh,
                    'confidence': round(confidence, 2),
                    'nutrition': nutrition
                })

                # 累加总营养
                for key in total_nutrition.keys():
                    total_nutrition[key] += nutrition[key]
            else:
                # 如果在营养库中找不到，使用默认值
                default_nutrition = {
                    'calories': 200,
                    'protein': 10.0,
                    'fat': 8.0,
                    'carbs': 20.0,
                    'fiber': 2.0
                }
                foods_info.append({
                    'food_class': food_class,
                    'food_name_zh': food_name_zh,
                    'confidence': round(confidence, 2),
                    'nutrition': {
                        'food_name_zh': food_name_zh,
                        'calories': 200,
                        'protein': 10.0,
                        'fat': 8.0,
                        'carbs': 20.0,
                        'fiber': 2.0,
                        'actual_serving_g': 150,
                        'description': '常规食物'
                    }
                })
                for key in total_nutrition.keys():
                    total_nutrition[key] += default_nutrition[key]

        # 四舍五入总营养值
        for key in total_nutrition:
            if key == 'calories':
                total_nutrition[key] = round(total_nutrition[key])
            else:
                total_nutrition[key] = round(total_nutrition[key], 1)

        # 生成饮食建议
        advice = self.generate_diet_advice(total_nutrition, user_profile)

        # 生成具体建议
        suggestions = self.generate_suggestions(foods_info, total_nutrition)

        return {
            'total_nutrition': total_nutrition,
            'foods': foods_info,
            'advice': advice,
            'suggestions': suggestions
        }

    def generate_diet_advice(self, total_nutrition: Dict, user_profile: Optional[Dict] = None) -> str:
        """生成饮食建议"""
        calories = total_nutrition['calories']

        # 基础建议
        if calories < 300:
            base_advice = "这是一份轻食餐，热量较低。"
        elif calories < 600:
            base_advice = "这是一份中等热量的餐食。"
        else:
            base_advice = "这是一份高热量的餐食。"

        # 营养均衡建议
        protein = total_nutrition['protein']
        fat = total_nutrition['fat']
        carbs = total_nutrition['carbs']

        protein_advice = ""
        if protein > 30:
            protein_advice = "蛋白质含量丰富，有助于肌肉合成。"
        elif protein < 15:
            protein_advice = "蛋白质含量偏低，建议搭配蛋奶或豆制品。"

        fat_advice = ""
        if fat > 30:
            fat_advice = "脂肪含量较高，建议控制油脂摄入。"

        carbs_advice = ""
        if carbs > 80:
            carbs_advice = "碳水化合物含量较高，建议适量运动消耗。"

        # 根据用户个人情况调整
        goal_advice = ""
        if user_profile:
            if user_profile.get('goal') == 'weight_loss' and calories > 500:
                goal_advice = "考虑到您的减重目标，建议减少份量。"
            elif user_profile.get('goal') == 'muscle_gain' and protein < 30:
                goal_advice = "为增肌效果更好，建议增加蛋白质摄入。"

        return f"{base_advice} {protein_advice} {fat_advice} {carbs_advice} {goal_advice}".strip()

    def generate_suggestions(self, foods_info: List[Dict], total_nutrition: Dict) -> List[str]:
        """生成具体建议列表"""
        suggestions = []

        if not foods_info:
            return suggestions

        # 基于食物组成的建议
        food_names = [f['food_name_zh'] for f in foods_info]

        if len(foods_info) == 1:
            suggestions.append(f"您当前餐食主要是{food_names[0]}。")
        else:
            suggestions.append(f"您当前餐食包含：{', '.join(food_names)}。")

        # 营养均衡建议
        if total_nutrition['fiber'] < 5:
            suggestions.append("膳食纤维摄入不足，建议搭配蔬菜或水果。")

        if total_nutrition['protein'] < 20:
            suggestions.append("蛋白质摄入较少，可以增加蛋奶或瘦肉。")

        if total_nutrition['fat'] > 40:
            suggestions.append("脂肪摄入较高，建议选择更清淡的烹饪方式。")

        if total_nutrition['carbs'] > 100:
            suggestions.append("碳水化合物摄入较高，可以适当减少主食份量。")

        # 添加通用健康建议
        suggestions.append("建议细嚼慢咽，每餐用时不少于20分钟。")
        suggestions.append("餐后半小时适量运动，有助消化。")

        return suggestions


# 单例模式
advisor = None


def get_advisor():
    """获取营养顾问实例（单例）"""
    global advisor
    if advisor is None:
        advisor = NutritionAdvisor()
    return advisor