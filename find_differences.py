def find_missing_files(source_files, check_files, output_file):
    # 両方のファイルを読み込む
    with open(source_files, 'r', encoding='utf-8') as f:
        source_set = set(line.strip() for line in f)
    
    with open(check_files, 'r', encoding='utf-8') as f:
        check_set = set(line.strip() for line in f)
    
    # check.txtにあってsource_files.txtにない項目を抽出
    missing_files = check_set - source_set
    
    # 結果をソートしてファイルに書き込む
    sorted_missing = sorted(list(missing_files))
    with open(output_file, 'w', encoding='utf-8') as f:
        for file in sorted_missing:
            f.write(f"{file}\n")
    
    print(f"Found {len(sorted_missing)} files in check.txt that are missing from source_files.txt")
    print(f"Results written to {output_file}")

# 実行
find_missing_files('source_files.txt', 'check.txt', 'missing_files.txt') 