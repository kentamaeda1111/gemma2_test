import json
import os
import glob
from anthropic import Anthropic
from typing import List, Dict
import time

class DialogueQualityEvaluator:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        
    def evaluate_dialogue(self, dialogue_history: List[Dict]) -> Dict:
        """
        Evaluate the quality of Gemma's responses in terms of Socratic dialogue
        Returns both tone and logic scores
        """
        # Skip the first Gemma question as specified
        relevant_exchanges = []
        for i in range(1, len(dialogue_history)-1, 2):
            if i+1 < len(dialogue_history):
                claude_msg = dialogue_history[i]["content"]
                gemma_msg = dialogue_history[i+1]["content"]
                relevant_exchanges.append({
                    "claude": claude_msg,
                    "gemma": gemma_msg
                })
        
        if not relevant_exchanges:
            return {"tone_score": 0.0, "logic_score": 0.0, "comment": "No relevant exchanges found"}
            
        prompt = """以下の対話を評価してください。Claudeの発言に対するGemmaの返答について、以下の2つの観点から評価してください：

1. ソクラテス的な口調（0-4点）
- 「かね？」「だろうか？」「ではないかね？」等の文末表現
- 「友よ」「君」等の対話者への呼びかけ方
- ソクラテスらしい語り口

2. ”問い”を返す形式（0-4点）
- 基本的には対話者の返答に

3. 日本語以外の言語や記号が出ていないか（0-4点）

4. 論理性と自然な対話（0-4点）
- 返答の論理的整合性
- 会話の自然な流れ
- 相手の発言に対する適切な応答

各観点について：
0: 全く不適切
1: 不適切な点が多い
2: 改善の余地あり
3: 概ね良好
4: 非常に優れている

評価は以下のフォーマットで返してください：
口調：[数字]
論理性：[数字]
コメント：[簡潔なコメント]

対話:
"""
        
        for exchange in relevant_exchanges:
            prompt += f"\nClaude: {exchange['claude']}\nGemma: {exchange['gemma']}\n"
            
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,  # 増やして完全な評価を取得
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # レスポンスをパースして評価を抽出
            evaluation_text = response.content[0].text.strip()
            
            # 評価を解析
            tone_score = 0.0
            logic_score = 0.0
            comment = ""
            
            for line in evaluation_text.split('\n'):
                if '口調：' in line:
                    tone_score = float(line.split('：')[1].strip())
                elif '論理性：' in line:
                    logic_score = float(line.split('：')[1].strip())
                elif 'コメント：' in line:
                    comment = line.split('：')[1].strip()
            
            return {
                "tone_score": tone_score,
                "logic_score": logic_score,
                "comment": comment
            }
            
        except Exception as e:
            print(f"Error evaluating dialogue: {e}")
            return {"tone_score": 0.0, "logic_score": 0.0, "comment": f"Error: {str(e)}"}

    def process_dialogue_files(self, input_dir: str, output_dir: str):
        """
        Process all dialogue files in the input directory and save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        dialogue_files = glob.glob(os.path.join(input_dir, "dialogue_*.json"))
        results = {}
        
        for file_path in dialogue_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    dialogue_data = json.load(f)
                
                # Get dialogue ID from filename
                file_name = os.path.basename(file_path)
                dialogue_id = file_name.split('_')[1]
                
                # Evaluate dialogue quality
                evaluation = self.evaluate_dialogue(dialogue_data['history'])
                
                # Store results
                results[dialogue_id] = {
                    "dialogue_id": dialogue_id,
                    "timestamp": dialogue_data['timestamp'],
                    "model_version": dialogue_data['model_version'],
                    "checkpoint": dialogue_data['checkpoint'],
                    "tone_score": evaluation["tone_score"],
                    "logic_score": evaluation["logic_score"],
                    "comment": evaluation["comment"]
                }
                
                print(f"Processed {file_name}: Tone={evaluation['tone_score']}, Logic={evaluation['logic_score']}")
                
                # Add delay to avoid rate limiting
                time.sleep(1)
            
    except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        # Save results
        output_path = os.path.join(output_dir, "dialogue_quality_scores.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    api_key = os.getenv("CLAUDE_API_KEY_QUALITY2")
    if not api_key:
        raise ValueError("CLAUDE_API_KEY_QUALITY2 environment variable not set")
    
    evaluator = DialogueQualityEvaluator(api_key)
    
    input_dir = "data/dialogue/raw_gemma"
    output_dir = "data/dialogue/quality_scores"
    
    evaluator.process_dialogue_files(input_dir, output_dir)

if __name__ == "__main__":
    main()
