import os
import glob

def list_raw_gemma_files():
    # raw_gemmaディレクトリのパス
    raw_gemma_dir = "data/dialogue/raw_gemma"
    
    # 出力ファイルのパス
    output_file = "data/dialogue/raw_gemma_files.txt"
    
    # raw_gemmaディレクトリ内のすべてのjsonファイルを取得
    json_files = glob.glob(os.path.join(raw_gemma_dir, "*.json"))
    
    # ファイル名のみを抽出してソート
    filenames = sorted([os.path.basename(f) for f in json_files])
    
    # ファイル名をテキストファイルに書き出し
    with open(output_file, "w", encoding="utf-8") as f:
        for filename in filenames:
            f.write(f"{filename}\n")
            
    print(f"Found {len(filenames)} files and wrote them to {output_file}")

if __name__ == "__main__":
    list_raw_gemma_files() 