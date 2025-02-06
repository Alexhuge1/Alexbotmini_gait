import os
import shutil

def remove_pycache(root_dir):
    # 遍历指定目录及其子目录
    for root, dirs, files in os.walk(root_dir):
        # 检查当前目录中是否有 __pycache__ 目录
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                # 删除 __pycache__ 目录及其内容
                shutil.rmtree(pycache_path)
                print(f"Deleted {pycache_path}")
            except Exception as e:
                print(f"Error deleting {pycache_path}: {e}")

if __name__ == "__main__":
    # 指定要搜索的根目录，这里使用当前目录
    root_directory = '.'
    remove_pycache(root_directory)

