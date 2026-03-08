import os
import sys

print("="*60)
print("环境调试信息")
print("="*60)

# 当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前目录: {current_dir}")

# 项目根目录
project_root = current_dir
print(f"项目根目录: {project_root}")

# 检查yolov5目录
yolov5_path = os.path.join(project_root, 'yolov5')
print(f"\n检查YOLOv5路径: {yolov5_path}")
if os.path.exists(yolov5_path):
    print(f"✓ YOLOv5目录存在")
    print(f"yolov5目录内容: {os.listdir(yolov5_path)}")
    
    models_path = os.path.join(yolov5_path, 'models')
    if os.path.exists(models_path):
        print(f"✓ models目录存在")
        print(f"models目录内容: {os.listdir(models_path)}")
        
        # 检查关键文件
        required_files = ['common.py', 'yolo.py', '__init__.py']
        for file in required_files:
            file_path = os.path.join(models_path, file)
            if os.path.exists(file_path):
                print(f"  ✓ {file} 存在")
            else:
                print(f"  ✗ {file} 不存在")
    else:
        print(f"✗ models目录不存在")
else:
    print(f"✗ YOLOv5目录不存在")

# 检查model目录
model_path = os.path.join(project_root, 'model')
print(f"\n检查模型路径: {model_path}")
if os.path.exists(model_path):
    print(f"✓ model目录存在")
    print(f"model目录内容: {os.listdir(model_path)}")
    
    best_pt_path = os.path.join(model_path, 'best.pt')
    if os.path.exists(best_pt_path):
        print(f"✓ best.pt 存在")
    else:
        print(f"✗ best.pt 不存在")
else:
    print(f"✗ model目录不存在")

print("\nPython路径:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

print("="*60)
