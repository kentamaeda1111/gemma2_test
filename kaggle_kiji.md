
大変だったことkaggleではしらせれるようにしたこと

評価メトリクスに関して全然うまくいかなかった。結局最後は試してみる、という結論だった。


今後やること　testのpromptや状態保存をもっといじれると思う

結局いろいろ学んだけど、最終的には検証するしかない、ということ。

quality checkの自動化はやってもいいかと


あとこれは実験的な意味合いもある。そもそもどれくらいデータがよければいいものになるのか？という

以下の記事を参考にした。
https://huggingface.co/google/gemma-2-2b/discussions/26
https://huggingface.co/google/gemma-2-2b/discussions/28


”””””””””””””””””””””””””””””””””””””””

※この記事は以下のコンペ向けに作った提出物です。
inference以外のコードはgithubにあげています。

https://www.kaggle.com/competitions/gemma-language-tuning


""""""""""""""""""""""""""""""プラニングフェーズ"""""""""""""""""""""""

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

■着手した課題
私が取り組むことにしたのはgemma-2をソクラテス風の口調にファインチューニングする、というものです。
具体的にはソクラテスが必ず”問い”で返す、という挙動、
そしてソクラテス風の口調（「かね？」のような）にすることです。

■なぜソクラテス
なぜこの方向性にしたかについて説明するためには、
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
広く注目されているもの、というわけではないかもしれませんが、
most likely to suceedというドキュメンタリーの中で紹介されている高校も
ソクラテス式の問答を積極的に取り入れていたり、
スタンフォードハイスクールでも哲学が唯一の必修科目だったり等
確実にそこに価値を感じている層がいるもの、というのが私の認識です。

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

繰り返しになりますが、ベースモデルの質がそこまでたかくないため、
ソクラテス問答の本質となる”問う力”みたいなものは期待できなかったため、
ソクラテスのような”問い”で返す且つソクラテスのような口調で喋るＡＩ、
をゴールに据えることにしています。

尚、日本語ではソクラテス的な口調というと、
方言のような老練な独特な口調があるのですが、
なかなか独特な内容であるため、ベースの日本語力が可能な限りあってほしいと考え、jpnを選びました。

また、itにしたのはチャット形式になるためです。

""""""""""""""""""""""""""""""データ生成フェーズ"""""""""""""""""""""""
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
2. Another incorporating phrase "You are Socrates, the ancient Greek philosopher." before user utterances

While we could have used tuners like XTuner, Axolotl, or LLaMA Factory to implement system prompt-like functionality during training, I prioritized staying aligned with Gemma2's original design philosophy and testing in the most natural way possible.



# Final Training Data
結果的には2つのデータを以下のように用意しました。
違いはpromptの３つのパターンです。

| Item | model1 | model2 |
|------|--------|--------|
| Number of dialogues extracted from 12 turns | 11 | 11 |
| Number of utterances per dialogue | 2 | 2 |
| Total number of dialogues | 2,662 | 2,662 |
| Total tokens | 685,875 | 752,369 |
| Average tokens per dialogue | 257.65 | 282.63 |
| MAX tokens per dialogue | 527 | 552 |
| MIN tokens per dialogue | 19 | 44 |
| Average user tokens* | 144.42 | 169.39 | 
| Average model tokens* | 113.24 | 113.24 | 
| System prompt | None | "You are Socrates, the ancient Greek philosopher." | 

""""""""""""""""""""""""""""""トレイン・テストフェーズ"""""""""""""""""""""""
■train.pyの作成方針

gemma2の公式ドキュメントで、例えばuser とmodelのjsonデータの形式等、
外してはいけない点についてはおさえつつ、
主に参考にしたのはkaggleのファインチューニングのコードを参考にし、
cursorを使ってgemma2やpeftといったもののドキュメントをフィードしながら、コードを作りました。
都度gemma2の公式ドキュメントを参照しながらコードを作るように気を付けました。

kaggle内でのトレインができるようにしたかったため、（環境スペックを記述）
品質は下がるリスクはあったものの、
qloraも使うような方針を取りました。


最初はattention mask等をせずにおこなったのですが、結果はさんさんたるものでした。
attention maskがカギでした。

評価指標は色々とためしましたが、
結局長くやったほうがよかったです。


■train.pyの実行結果

尚、モデルをいくつもトレインするという前提であったため
もともと外部のＧＰＵをレンタル予定ではあったものの、
kaggleでもトレインができてほしかったので、kaggleでもためしたところ、
ＣＰＵのＲＡＭが２９ＧＢをこえて（ここはもっと描写を正確にする）クラッシュしてしまいました。
修正はkaggleのコンペには間に合いませんでしたが、以下のような修正をすることで、
kaggleでもトレインできるモデルになりました。


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







