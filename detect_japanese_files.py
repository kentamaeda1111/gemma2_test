import os
import re

# 検査対象のファイルパスを指定
TARGET_FILES = [
    "src/models/training/train_tensordock.py",
    # 他のファイルパスを追加可能
]

def contains_japanese(text):
    """文字列に日本語が含まれているかチェック"""
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+')
    return japanese_pattern.findall(text)

def scan_file_for_japanese(file_path):
    """ファイルを読み込んで日本語の箇所を抽出"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            japanese_texts = contains_japanese(content)
            if japanese_texts:
                # 日本語を含む行を抽出
                japanese_lines = []
                for line_num, line in enumerate(content.split('\n'), 1):
                    if any(jp_text in line for jp_text in japanese_texts):
                        japanese_lines.append((line_num, line.strip()))
                return japanese_lines
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def find_japanese_files(files=None):
    """指定されたファイルまたはディレクトリから日本語を検索"""
    japanese_files = {}
    
    if files:
        # 特定のファイルのみを検査
        for file_path in files:
            if os.path.exists(file_path):
                japanese_lines = scan_file_for_japanese(file_path)
                if japanese_lines:
                    japanese_files[file_path] = japanese_lines
            else:
                print(f"Warning: File not found - {file_path}")
    else:
        # ディレクトリ全体を検査
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(('.py', '.md')):
                    file_path = os.path.join(root, file)
                    japanese_lines = scan_file_for_japanese(file_path)
                    if japanese_lines:
                        japanese_files[file_path] = japanese_lines
    
    return japanese_files

def main():
    # TARGET_FILESが指定されている場合はそれらのファイルのみを検査
    japanese_files = find_japanese_files(TARGET_FILES if TARGET_FILES else None)
    
    print("=== Files containing Japanese text ===")
    for file_path, lines in japanese_files.items():
        print(f"\n{file_path}:")
        for line_num, line in lines:
            print(f"  Line {line_num}: {line}")
    print(f"\nTotal files found: {len(japanese_files)}")

if __name__ == "__main__":
    main() 