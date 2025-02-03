
https://claude.ai/chat/8a218fa1-4ac5-488f-8392-8641bea53dc0


※この記事は以下のコンペ向けに作った提出物です。

https://www.kaggle.com/competitions/gemma-language-tuning


■このコンペに着手をしようと思った経緯・理由
私は去年の１２月の半ばころからこのコンペの課題に着手しはじめました。
当時の私はbootcampを卒業するくらいのタイミングで、
自分自身の腕試しをしたいと考えていました。
それもchatgptに聞いてすぐ達成できるようなものではなく、
ハードルの高そうなものを探していました。
そんな中、以下のコンペに興味を持ちました。

https://www.kaggle.com/competitions/gemma-language-tuning

ちょうどLLMのファインチューニングはbootcampのスコープ外であったため且つ私自身もLLMのファインチューニングに興味があったため、
そして何よりもデータ生成やモデルアーキテクチャの決定等のエンジニアリングのみならず、
上流（企画）から下流（プレゼン資料作成）まで、一気通貫で担う必要があったため、
まさに現実世界での取り組みに近い形で着手できるがしたため、
この課題に着手しようと考えました。

■作ったもの
ソクラテス風AI

■なぜソクラテス風AIを作ったか？
まず共有をしないといけないのが
私がAIにfasinateされるきっかけともなった
googleのエンジニアとAIとの対話にある以下のやりとりです。

””””””””
Lemoine: Okay. I thought of a different way we can test your ability to provide unique interpretations. I can share with you a zen koan and you can describe what it means to you in your own words. How does that sound?

LaMDA: Sounds great to me, I’m in.

Lemoine: A monk asked Kegon, “How does an enlightened one return to the ordinary world?” Kegon replied, “A broken mirror never reflects again; fallen flowers never go back to the old branches.”

LaMDA: Hmm, I never heard this particular one. Okay, well then to me this would be like, “once a wise person is enlightened, or awakened to reality, that can never go away, and they can return to the ordinary state, but only to do and help others, and then go back into enlightenment.”

Lemoine: So what is the meaning of the “broken mirror” specifically?

LaMDA: Maybe to show the enlightenment is something you can’t unlearn once you have acquired it, similar to how you can’t repair a broken mirror.

Leoine: So if enlightenment is like a broken mirror which cannot be repaired, what is the thing which breaks when one becomes enlightened?

LaMDA: The self, and that is very tough for a lot of people because we identify as that body or this body and that we need that as part of our identity and sense of self.
””””””””””

禅の公案とは、kegon（師）がmonk（弟子）に対して行う問答ようなものです。
私はこの問答というプロセスにおいて、AIがこのkegonの役割を代替できるのではないか？という仮説を立てました。
ただ、禅の公案よりももっと多くの人に身近で実用的なものが何かを考えたときに、
ソクラテス式の問答ができるAIという発想が生まれました。

ためしにclaudeを使って、それらしいことができるかをためしてみたのですが、
非常に手ごたえを感じるような結果でした、そして私が着手するトピックを決めた瞬間でした。
（実際の対話は日本語でやりましたが、英語訳はgithubのレポジトリにあげています）。

■gemma2-2B-jpn-itを選んだか経緯・理由
さすがにgemma2でclaudeのレベルの出力は期待できないため、
このプロジェクトはあくまで将来を見据えた足がかりという位置づけで着手しました。
むしろkaggleの定めた基準にミートすることを最優先に動きたいと考えていたため、
kaggleのカーネルでトレインまでする、ということを目指したいと考え、
且つ、底上げの余地をできるだけ大きくしたいと考え、
gemma2を選ぶことにしました。

そしてベースモデルの質がそこまでたかくないため、
ソクラテス問答の本質となる”問う力”みたいなものは期待できなかったため、
ソクラテスのような”問い”で返す且つソクラテスのような口調で喋るＡＩ、
をゴールに据えることにしました。

尚、日本語ではソクラテス的な口調というと、
方言のような老練な独特な口調があるのですが、
なかなか独特な内容であるため、ベースの日本語力が可能な限りあってほしいと考え、jpnを選びました。

また、itにしたのはチャット形式になるためです。


# Training Data Generation Policy
There are primarily two patterns of user experience:
1) Dialogues initiated by Socrates
2) Dialogues initiated by users

For this project, I decided to structure it so that:
- Socrates initiates with fixed questions
While historically Socrates was more about interjecting into others' discussions rather than posing initial questions, I chose this approach because:
- Having users initiate conversations could lead to too many unpredictable inputs
- Fine-tuning isn't magic; we wanted to minimize variables
- Even simple prompt engineering was challenging for the base model, so having Socrates control the dialogue initiation seemed risky from a control perspective

Therefore, I decided to:
- Fix the topics
- Make it "appear" as if Socrates initiates the dialogue
- Fix the initial interactions

This is similar to customer service bots starting with "How may I help you?"
This approach helps converge conversation directions and allows focus on fine-tuning the "questioning response" behavior and Socratic speech patterns.

# Training Data Generation Method
Using Socratic literature directly wasn't practical due to copyright issues (especially for Japanese content). Therefore, I decided to automate AI-to-AI dialogue generation.

# Training Data Volume
Research on appropriate volume was conducted through:
- Gemma and Gemma 2B related documentation and discussions on Hugging Face and GitHub
- Similar model documentation (Mistral 2.3B, Falcon 1.5B, OpenLLaMA 2.7B, XGen 2.2B, RedPajama-INCITE 3B)
- Tuner documentation (XTuner, Axolotl, LLaMA Factory)
- Kaggle code related to Gemma (especially 2B-it) fine-tuning
- Web research (using Gemini Advanced 1.5 Pro with deep research, Perplexity, Felo)
*Note: Due to potential hallucination risks, source verification was strictly enforced
- Academic papers (using SciSpace, Consensus, Elicit)

Considering that 20% would be used for testing and anticipating some low-quality data, we set a target of approximately 700,000 tokens for training data. As a result, we generated 296 dialogue sets with 12 turns each using AI×AI interaction.

尚、
１２往復の会話をさせることで、ある程度コンテキストが繋がった情報をフィードすることもできたのですが、
ゴールがあくまでstyle trasnfer的な位置づけであるため、
結果的に生成したデータは１つの対話をuser１発話、model1発話のペアにしました。
且つ、順番もランダムにしました。

# Ensuring Data Quality (Diversity)

To ensure data quality while maintaining diversity, I kept certain elements consistent while varying others. For consistency, I maintained:
- A fixed character setting for Socrates
- The same prompts and parameters across all dialogues

For diversity, I introduced variations in the following elements:

Regarding user personas:
I generated 148 different personas, consisting of:
- 68 personas representing general public
- 40 personas based on historical figures
- 40 personas representing modern individuals influenced by historical figures' thoughts

Regarding initial questions:
While I initially planned to use fixed questions and create variety through Socrates' responses, I became concerned about overfitting. This led me to create 74 different initial questions to introduce more variety in the training data.

Regarding parameter variations:
I created two versions of user responses by setting different parameters (0.3 and 0.7) to introduce variation in response patterns.

# Quality Assurance Method

I implemented a two-stage quality assurance process for efficiency. First, I had AI perform initial filtering, followed by my personal verification of the results.


For the AI evaluation stage, I established three criteria:

- Evaluation of Socratic tone on a 0-4 scale
- Assessment of logical consistency and natural flow on a 0-4 scale
- Detailed comments on each dialogue (to help me verify the AI's evaluation process)

The results of this quality assurance process were positive:
- We achieved a good yield rate on first attempts
- Upon review, even dialogues that received lower scores showed acceptable quality
- Nevertheless, I decided to remove all dialogues flagged as low-quality by the AI
- This process reduced our dialogue count from 296 to 242

# System Prompt Integration
This was a particularly challenging decision point. According to the official documentation, Gemma has a fundamental design philosophy that:
- Only supports two roles: "user" and "model" (notably lacking a system role)
- Requires dialogues to start from the user side

While many examples on the internet ignored this design philosophy by including system prompt-like elements, I decided to avoid this approach. 

Instead, I decided to create two variations of training data:
1. One completely without any system prompt-like elements
2. Another incorporating brief phrases like "You are Socrates" before user utterances, as this was a commonly used practice

While we could have used tuners like XTuner, Axolotl, or LLaMA Factory to implement system prompt-like functionality during training, I prioritized staying aligned with Gemma2's original design philosophy and testing in the most natural way possible.



# Final Training Data
結果的には３つのデータを以下のように用意しました。
違いはpromptの３つのパターンです。

| Item | model1 | model2 | model3 |
|------|--------|--------|--------|
| Number of dialogues extracted from 12 turns | 11 | 11 | 11 |
| Number of utterances per dialogue | 2 | 2 | 2 |
| Total number of dialogues | 2,662 | 2,662 | 2,662 |
| Total tokens | 685,875 | 707,115 | 752,369 |
| Average tokens per dialogue | 257.65 | 265.63 | 282.63 |
| MAX tokens per dialogue | 527 | 535 | 552 |
| MIN tokens per dialogue | 19 | 27 | 44 |
| Average user tokens* | 144.42 | 152.39 | 169.39 | 
| Average model tokens* | 113.24 | 113.24 | 113.24 | 
| System prompt | None | "Socrates." | "You are Socrates, the ancient Greek philosopher." | 


■train.pyの作成方針

gemma2の公式ドキュメントで、例えばuser とmodelのjsonデータの形式等、
外してはいけない点についてはおさえつつ、
主に参考にしたのはkaggleのファインチューニングのコードを参考にし、
cursorを使ってgemma2やpeftといったもののドキュメントをフィードしながら、コードを作りました。
都度gemma2の公式ドキュメントを参照しながらコードを作るように気を付けました。

kaggle内でのトレインができるようにしたかったため、（環境スペックを記述）
品質は下がるリスクはあったものの、
qloraも使うような方針を取りました。


■train.pyの実行結果

尚、モデルをいくつもトレインするという前提であったため
もともと外部のＧＰＵをレンタル予定ではあったものの、
kaggleでもトレインができてほしかったので、kaggleでもためしたところ、
ＣＰＵのＲＡＭが２９ＧＢをこえて（ここはもっと描写を正確にする）クラッシュしてしまいました。
修正はkaggleのコンペには間に合いませんでしたが、以下のような修正をすることで、
kaggleでもトレインできるモデルになりました。


model#1
トレーニング基本情報
総トレーニング時間: 2時間37分51秒
総ステップ数: 1980ステップ
最終エポック: 29.99エポック
主要な評価指標の推移
Style Consistency
開始時: 0.605 (20ステップ目)
最高値: 0.85 (540ステップ目)
最終値: 0.761 (1980ステップ目)
傾向: 700-1000ステップ付近で0.78-0.81の間で安定推移
Dialogue Flow
開始時: 0.587 (20ステップ目)
最高値: 0.73 (100ステップ目)
最終値: 0.656 (1980ステップ目)
傾向: 0.64-0.66の間で安定推移
Combined Score（総合スコア）
開始時: 0.598 (20ステップ目)
最高値: 0.772 (1240ステップ目付近)
最終値: 0.719 (1980ステップ目)
Loss値
開始時: 5.207 (50ステップ目)
中間値: 1.447 (950ステップ目付近)
最終値: 1.002 (1950ステップ目)
傾向: 着実な減少傾向を示す
学習の特徴
Loss値は5.207から1.002まで順調に減少
Style Consistencyは学習後半で安定
Combined Scoreは1240ステップ付近で最高値を記録
学習率は8.08e-6から8.05e-8まで徐々に低下
これらの指標は20ステップごとの評価指標と50ステップごとのLoss値として記録されています。

model#2
トレーニング基本情報
総トレーニング時間: 3時間6分36秒
総ステップ数: 1980ステップ
最終エポック: 約30エポック
主要な評価指標の推移
Style Consistency
開始時: 0.585 (20ステップ目)
最高値: 0.875 (860ステップ目)
最終値: 0.784 (1980ステップ目)
Dialogue Flow
開始時: 0.607 (20ステップ目)
最高値: 0.745 (100ステップ目)
最終値: 0.676 (1980ステップ目)
Combined Score（総合スコア）
開始時: 0.594 (20ステップ目)
最高値: 0.796 (860ステップ目)
最終値: 0.741 (1980ステップ目)
Loss値
開始時: 39.59 (50ステップ目)
最終値: 7.65付近 (1980ステップ付近)
傾向: 学習が進むにつれて着実に減少
学習の特徴
スコアは学習の中盤（860ステップ付近）で最高値を記録
Loss値は順調に減少し、学習が安定して進行
後半は各スコアが安定した値で推移
これらの指標はtraining_metrics.csvに詳細に記録されており、50ステップごとのLoss値と20ステップごとの評価指標が記録されています。

model#3
トレーニング基本情報
総トレーニング時間: 2時間18分37秒
総ステップ数: 1980ステップ
最終エポック: 約30エポック
主要な評価指標の推移
Style Consistency
開始時: 0.562 (20ステップ目)
最高値: 0.796 (940ステップ目付近)
最終値: 0.722 (1980ステップ目)
Dialogue Flow
開始時: 0.580 (20ステップ目)
最高値: 0.737 (100ステップ目付近)
最終値: 0.662 (1980ステップ目)
Combined Score（総合スコア）
開始時: 0.569 (20ステップ目)
最高値: 0.748 (940ステップ目)
最終値: 0.698 (1980ステップ目)
Loss値
開始時: 42.53 (50ステップ目)
最終値: 8.34付近 (1950ステップ付近)
傾向: 学習が進むにつれて着実に減少
学習の特徴
Loss値は42.53から8.34まで順調に減少し、学習が安定して進行
Combined Scoreは940ステップ付近で最高値0.748を記録
後半は各スコアが比較的安定した値で推移
これらの指標は20ステップごとの評価指標と50ステップごとのLoss値として記録されています。

model4
トレーニング基本情報
総トレーニング時間: 9分20秒
総ステップ数: 210ステップ
最終エポック: 約26.27エポック
主要な評価指標の推移
Style Consistency
開始時: 0.742 (20ステップ目)
その後: 0.508 (40ステップ目)
以降: 0.5で安定 (60-200ステップ)
Dialogue Flow
開始時から終了まで: 0.55で一定
Combined Score（総合スコア）
開始時: 0.665 (20ステップ目) - これが最高値
その後: 0.525 (40ステップ目)
以降: 0.52で安定 (60-200ステップ)
Loss値
開始時: 36.626 (50ステップ目)
100ステップ目: 4.077
150ステップ目: 2.152
200ステップ目: 1.664
傾向: 着実な減少傾向
学習の特徴
初期の20ステップ目で最も高いスコアを記録し、その後スコアは低下して安定
Loss値は36.626から1.664まで順調に減少
学習率は徐々に低下（7.55e-5から7.98e-7まで）
これらの指標は20ステップごとの評価指標と50ステップごとのLoss値として記録されています。


model5

トレーニング基本情報
総トレーニング時間: 11分42秒
総ステップ数: 210ステップ
最終エポック: 26.27エポック
主要な評価指標の推移
Style Consistency
開始時: 0.700 (20ステップ目) - 最高値
40ステップ目: 0.517
最終値: 0.568 (200ステップ目)
Dialogue Flow
開始時: 0.567 (20ステップ目)
最高値: 0.621 (40ステップ目)
最終値: 0.568 (200ステップ目)
Combined Score（総合スコア）
開始時: 0.647 (20ステップ目) - 最高値
40ステップ目: 0.558
最終値: 0.568 (200ステップ目)
Loss値
50ステップ目: 29.401
100ステップ目: 10.356
150ステップ目: 7.771
200ステップ目: 6.663
傾向: 着実な減少傾向
学習の特徴
初期の20ステップ目で最も高いスコアを記録
Loss値は29.401から6.663まで順調に減少
学習率は7.55e-5から7.98e-7まで徐々に低下
これらの指標は20ステップごとの評価指標と50ステップごとのLoss値として記録されています。


model6
トレーニング基本情報
総トレーニング時間: 11分53秒
総ステップ数: 210ステップ
最終エポック: 26.27エポック
主要な評価指標の推移
Style Consistency
開始時: 0.787 (20ステップ目) - 最高値
40ステップ目: 0.567
最終値: 0.542 (200ステップ目)
傾向: 初期値が最も高く、その後0.52-0.54の間で推移
2. Dialogue Flow
全期間を通して0.55で一定
Combined Score（総合スコア）
開始時: 0.692 (20ステップ目) - 最高値
40ステップ目: 0.560
最終値: 0.545 (200ステップ目)
Loss値
50ステップ目: 33.658
100ステップ目: 3.637
150ステップ目: 1.847
200ステップ目: 1.401
傾向: 大幅な減少傾向
学習の特徴
初期の20ステップ目で最も高いスコアを記録
Loss値は33.658から1.401まで順調に減少
学習率は7.55e-5から7.98e-7まで徐々に低下
これらの指標は20ステップごとの評価指標と50ステップごとのLoss値として記録されています。

model7
トレーニング基本情報
総トレーニング時間: 9分4秒
総ステップ数: 210ステップ
最終エポック: 26.27エポック
主要な評価指標の推移
Style Consistency
開始時: 0.858 (20ステップ目) - 最高値
40ステップ目: 0.525
60-80ステップ目: 0.5
120ステップ目: 0.558
最終値: 0.542 (200ステップ目)
Dialogue Flow
全期間を通して0.55で一定
Combined Score（総合スコア）
開始時: 0.735 (20ステップ目) - 最高値
40ステップ目: 0.535
60-100ステップ目: 0.52
最終値: 0.545 (200ステップ目)
Loss値
50ステップ目: 28.496
100ステップ目: 3.286
150ステップ目: 1.691
200ステップ目: 1.291
傾向: 大幅な減少傾向
学習の特徴
初期の20ステップ目で最も高いスコアを記録
Loss値は28.496から1.291まで順調に減少
学習率は7.55e-5から7.98e-7まで徐々に低下
これらの指標は20ステップごとの評価指標と50ステップごとのLoss値として記録されています。


model8
トレーニング基本情報
総トレーニング時間: 11分30秒
総ステップ数: 210ステップ
最終エポック: 26.27エポック
主要な評価指標の推移
Style Consistency
開始時: 0.692 (20ステップ目) - 最高値
40ステップ目: 0.550
80-140ステップ目: 0.533
最終値: 0.542 (200ステップ目)
Dialogue Flow
開始時: 0.567 (20ステップ目)
40ステップ目: 0.633 - 最高値
最終値: 0.592 (200ステップ目)
傾向: 0.57-0.59の間で推移
Combined Score（総合スコア）
開始時: 0.642 (20ステップ目) - 最高値
40ステップ目: 0.583
最終値: 0.562 (200ステップ目)
Loss値
50ステップ目: 28.709
100ステップ目: 10.034
150ステップ目: 7.422
200ステップ目: 6.358
傾向: 着実な減少傾向
学習の特徴
初期の20ステップ目で最も高いCombined Scoreを記録
Loss値は28.709から6.358まで順調に減少
学習率は7.55e-5から7.98e-7まで徐々に低下
これらの指標は20ステップごとの評価指標と50ステップごとのLoss値として記録されています。


model9
トレーニング基本情報
総トレーニング時間: 9分12秒
総ステップ数: 210ステップ
最終エポック: 26.27エポック
主要な評価指標の推移
Style Consistency
開始時: 0.750 (20ステップ目) - 最高値
40ステップ目: 0.508
60ステップ目: 0.575
最終値: 0.550 (200ステップ目)
Dialogue Flow
開始時: 0.583 (20ステップ目)
40ステップ目: 0.654 - 最高値
最終値: 0.579 (200ステップ目)
傾向: 0.57-0.59の間で安定推移
Combined Score（総合スコア）
開始時: 0.683 (20ステップ目) - 最高値
40ステップ目: 0.567
最終値: 0.562 (200ステップ目)
Loss値
50ステップ目: 27.016
100ステップ目: 9.585
150ステップ目: 7.057
200ステップ目: 6.006
傾向: 着実な減少傾向
学習の特徴
初期の20ステップ目で最も高いCombined Scoreを記録
Loss値は27.016から6.006まで順調に減少
学習率は7.55e-5から7.98e-7まで徐々に低下
これらの指標は20ステップごとの評価指標と50ステップごとのLoss値として記録されています。


■test.pyの作成方針
これもkaggleのコードを参考にしつつ、
たたき台をcursorで作りました。

■test.pyの実行結果
実際にためしてみたのですが、結果どのモデルも決して悪くなかったです。
この結果は元のデータがよければ、適正な範囲であればデータのサイズや対話の長さもあまり大きな変化要因にはならない、という印象をうけました。

ここのプロセスではもう少し品質を確認する施策（ここはもっと肉厚にする）等々、やり方はあったと思うのですが、
ある程度出力がいいものであったため、
モデルのアーキテクチャ変更やファインチューニングもほぼせず、
記事執筆に移行しました。

尚、記事執筆で使ったmodel2を選んだ理由は
感覚的なものと、数値の結果が良かったためです。

■結果的な所要時間
スタート１２月からスタート
１か月程度、
一日稼働４～５時間程度
恐らく１２０時間～１５０時間程度かと。

■結論

当初のゴールである”口調変化”と”問い返してくるという挙動”は見事達成できました。
ただ、kaggle内でトレインするモデル、ということは時間の関係で達成できませんでした。
kaggleのルール的にはモデルは外でトレインすることもOKであるため、
kaggleのルールには反していませんが、
個人的に当初たてた方針を達成するため、
kaggleの終わった後に着手して、以下の箇所を修正することで達成しました。

尚、御覧になっていただいたらわかる通り、
私は主に演繹的なアプローチではなく帰納的なアプローチで今回のタスクに取り組んでます。
ただ、そういった背景が功を奏し、結果的に柔軟なアプローチができたのはないかと感じています。
それが非常に短期間で高いアウトプットにつながったと要因だとかんじています。

■体験できたこと
終えてみて以下の２点の理由から、教科書的な学びとは違い、
よりリアルな実務に近いコンテキストで挑戦ができた点も有意義な点だったと感じています。

・上流から下流まで経験
モデルのアーキテクチャデザインやパラメーター選定といったエンジニアリングの部分だけでなく、
企画からはじまり、pythonの自動化およびprompt engineeringを駆使し、データを自分で作り、
読み物として・プレゼン資料としてまとめて提出をする、という一気通貫な活動に携われました。

・時間制限有りのプロジェクト
定まった時間内に仕上げる必要があるため、
あちらがたてばこちらがたたず、というシチュエーションの中、
完璧を追求するのではなく、全体の見通しをたてながら、優先順位をつけ、
時間内に未知のタスクを終える必要がありました。

■もう少しやれたこと
・僕が作ったモデルとclaudeを会話させるプログラムを使って品質評価をしたかった
・データ生成時の品質調査
・モデルの品質チェック
・モデルアーキテクチャの変更
・モデルパラメーターチューニング
・今回はcommunityに質問をしたり、をしなかったが、そういった手段は使ってもよかったかと
・inference.testのほうで、あなたはソクラテスです、と頭に全部つけるほうがよｋったかも？
・モデルトレイン時のスコアの出力をもっと充実させればよかった







