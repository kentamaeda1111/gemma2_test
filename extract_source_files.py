import json

def extract_unique_source_files(input_file, output_file):
    # JSONファイルを読み込む
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # source_fileの値を重複なく取得
    source_files = set()
    for item in data:
        if 'source_file' in item:
            source_files.add(item['source_file'])
    
    # ソートしてファイルに書き込む
    sorted_files = sorted(list(source_files))
    with open(output_file, 'w', encoding='utf-8') as f:
        for file in sorted_files:
            f.write(f"{file}\n")

    print(f"Extracted {len(sorted_files)} unique source files to {output_file}")

# 実行
extract_unique_source_files('2gouki.json', 'source_files.txt') 