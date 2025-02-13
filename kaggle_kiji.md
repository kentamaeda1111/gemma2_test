

※この記事は以下のコンペ向けに作った提出物です。
inference以外のコードはgithubにあげています。

https://www.kaggle.com/competitions/gemma-language-tuning


""""""""""""""""""""""""""""""プラニングフェーズ"""""""""""""""""""""""

https://claude.ai/chat/6c38fe35-0a06-4c1e-be3c-6da3693ebadb
■着手したトピックおよびそのトピックを選んだ理由
私が取り組むことにしたのはgemma-2をソクラテス風の口調にファインチューニングする、というものです。
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
ただ、禅の公案は少しなじみがない人が多いと思ったので、
もっと多くの人に身近で実用的なものが何かを考えたときに、
ソクラテス式の問答ができるAIという発想が生まれました。

広く注目されているもの、というわけではないかもしれませんが、
most likely to suceedというドキュメンタリーの中で紹介されている高校も
ソクラテス式の問答を積極的に取り入れていたり、
スタンフォードハイスクールでも哲学が唯一の必修科目だったり等
確実にそこに価値を感じている層がいるもの、というのが私の認識です。

ためしにclaudeを使って、それらしいことができるかをためしてみたのですが、
非常に手ごたえを感じるような結果でした、そして私が着手するトピックを決めた瞬間でした。
以下はその手ごたえを感じた会話の一部を抜粋したものです。

""""""""""""""

AI:
Why do we tend to feel as if consciousness is 'self,' even though we know consciousness is just one part of 'self'?"

Myself:
Rather than thinking that consciousness is "self," I think in most cases either people aren't aware that consciousness is just a part of "self," or even if they become aware of this, there's a dynamic at work where, being greedy, we want to think of it as ours.

AI:
"That's a very insightful answer. You've pointed out two important things:
The lack of awareness that 'consciousness is just a part of self'
The 'dynamic of wanting to think of it as ours' that works even when we become aware
So here's my question:
Why do you think this 'dynamic of wanting to think of it as ours' works particularly strongly with 'consciousness,' more than with other parts like hands and feet?"


Myself:
I think it's because we don't want to be separated from it. We might think separation = death. I don't feel love for my hand, but I feel love for my daughter because I don't want to be separated from her. My hand will never abandon me of its own will. That's because I think it's under my complete control. However, my daughter might leave me. I think the mechanism of wanting to keep her close tries to manipulate through something called love, tries to make her addicted to me. Perhaps living beings instinctively understand the North Wind and Sun theory.

AI:
"That's a very deep insight. You've made an essential point about the relationship between the feeling of 'not wanting to be separated' and 'control.'
Let's think about 'consciousness' from this perspective. You made an interesting comparison:
You don't feel love for your hand because it's under complete control
You feel love for your daughter because she might leave
So here's my question:
Is 'consciousness' under our 'complete control'? For example, when we go to sleep, can we freely control our consciousness?"


Myself:
It's not under our control. I think we just want to believe it's under our control.


AI:
"That's right. Your observation that 'we just want to believe it's under our control' is a very important insight.
Let's organize what we've discussed so far. We seem to be in this paradoxical situation:
1. Consciousness:
- Isn't actually under our control
- But we want to believe it is
- We fear being separated from it
2. This belief and fear:
- Makes it harder to see the fact that consciousness is just a part of self
- Strengthens attachment to consciousness
- Functions as a kind of self-defense

So here's my question:
If we could free ourselves from this belief that 'we can/want to control consciousness,' how would our understanding of 'self' change?"


Myself:
I think it's a sense of being part of nature. A stance of accepting things as they are.

AI:
"That's right. The insight of 'being part of nature' and 'accepting things as they are' seems to be an important point we've reached in our dialogue.
Through our conversation, we've followed this path:
1. First, about the concept of 'self':
- It's a collective term for multiple elements
- Consciousness is just one part
- Self is still self even without consciousness

2. Then about attachment to consciousness:
- The belief in wanting to control something we can't control
- Fear of separation
- Function as self-defense

3. And finally, liberation from these:
- Understanding self as part of nature
- Accepting things as they are

So let me ask an essential question:
What changes do you think this understanding of 'being part of nature' brings to our daily lives?"

Myself:
I think it removes attachments. Since we ultimately have to let go anyway, I think it gives us a stance of being grateful for and enjoying our current state.

AI:
"That's a wonderful insight. Accepting the reality that 'we ultimately have to let go' and taking the attitude of 'being grateful for and enjoying our current state' seems to show an important path to what we might call 'enlightenment.'
""""""""""""""

githubのレポジトリには渡したpromptや、AIの返答の意図も含め、全ての対話をアップしましたので、
興味がある方はそちらをご覧ください。

■gemma2-2B-jpn-itを選んだか経緯・理由
さすがにgemma2でclaudeのレベルの出力は期待できないため、
このプロジェクトはあくまで将来を見据えた足がかりという位置づけであり、gemma2でどの程度のものが作れるのか？という検証を行うためのもの、という位置づけです。
むしろ今回はkaggleのコンペ向けという前提があったため、
品質の高さを追求するというよりは、
kaggle環境でトレインできるようなモデルを作るということを目指したかったため、
まずは一番小さなgemma ２Bモデルでためしてみようと考えました。

尚、日本語ではソクラテス的な口調というと、
方言のような老練な独特な口調があるのですが、
なかなか独特な内容であるため、ベースの日本語力が可能な限りあってほしいと考え、jpnを選びました。

また、itにしたのはチャット形式になるためです。

■gemma2-2b-jpn-itの底力
まず底力をためしました。
ここで気づいたのはpromptが非常に大事だということです。
以下みたいな感じでもやりましたが、全然だめだめでした。
ただ以下のようにしたらうまくいくようになりました。

ただ、それでも問い返してくるような挙動にはなったものの、口調はどうしてもソクラテスになりませんでした。
ということでソクラテスの口調にするためのファインチューニングとなると、ベースの良さにできるだけ影響はあたえず、
表層レイヤーの働きかけを行う必要があります。

（（（（（（（（（（（（（ここは実際にやってみせる））））））））））））））））））））））
※その際質問をした場合はどうか？もチェック

■ユーザー体験の設定
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

■Training Data Generation Method
Using Socratic literature directly wasn't practical due to copyright issues (especially for Japanese content). Therefore, I decided to automate AI-to-AI dialogue generation.

■Training Data Volume
Research on appropriate volume was conducted through:
- Gemma and Gemma 2B related documentation and discussions on Hugging Face and GitHub
- Similar model documentation (Mistral 2.3B, Falcon 1.5B, OpenLLaMA 2.7B, XGen 2.2B, RedPajama-INCITE 3B)
- Tuner documentation (XTuner, Axolotl, LLaMA Factory)
- Kaggle code related to Gemma (especially 2B-it) fine-tuning
- Web research (using Gemini Advanced 1.5 Pro with deep research, Perplexity, Felo)
*Note: Due to potential hallucination risks, source verification was strictly enforced
- Academic papers (using SciSpace, Consensus, Elicit)

かなり書いてあることにばらつきがあったのですが、
念のため多めに生成することにし、700,000tokenほどのデータにしようと考えました。
尚、
１２往復の会話をさせることで、ある程度コンテキストが繋がった情報をフィードすることもできたのですが、
ゴールがあくまでstyle trasnfer的な位置づけであるため、
結果的に生成したデータは１つの対話をuser１発話、model1発話のペアにしました。
且つ、順番もランダムにしました。

■Ensuring Data Quality (Diversity)

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

■Quality Assurance Method

I implemented a two-stage quality assurance process for efficiency. First, I had AI perform initial filtering, followed by my personal verification of the results.

For the AI evaluation stage, I established three criteria:

- Evaluation of Socratic tone on a 0-4 scale
- Assessment of logical consistency and natural flow on a 0-4 scale
- Detailed comments on each dialogue (to help me verify the AI's evaluation process)
（explanable AIの取り組み）

■Evaluation Results
Analysis of 3,256 dialogue pairs (296 conversations × 11 pairs each) revealed:

**Updated Socratic Style Distribution**
style_distribution = {
    "Score 0 (Not Socratic at all)": {
        "count": 0,
        "percentage": "0.0%",
        "characteristics": "No Socratic tone whatsoever"
    },
    "Score 1 (No Socratic Elements)": {
        "count": 0,
        "percentage": "0.0%",
        "characteristics": "Lacks any recognizable Socratic elements"
    },
    "Score 2 (Slightly Socratic)": {
        "count": 61,
        "percentage": "1.9%",
        "characteristics": "Not bad, but some noticeable issues"
    },
    "Score 3 (Quite Socratic)": {
        "count": 2018,
        "percentage": "62.0%",
        "characteristics": "Feels generally Socratic in tone"
    },
    "Score 4 (Truly Socratic)": {
        "count": 1177,
        "percentage": "36.1%",
        "characteristics": "Excellent Socratic style—near flawless"
    }
}

**Updated Logical Consistency Distribution**
logic_distribution = {
    "Score 0 (Utterly Nonsensical)": {
        "count": 0,
        "percentage": "0.0%",
        "characteristics": "Completely incoherent statements"
    },
    "Score 1 (Out of Sync)": {
        "count": 0,
        "percentage": "0.0%",
        "characteristics": "Conversation appears disjointed"
    },
    "Score 2 (Minor Issues)": {
        "count": 10,
        "percentage": "0.3%",
        "characteristics": "Overall okay, but a few logical gaps"
    },
    "Score 3 (Coherent)": {
        "count": 1234,
        "percentage": "37.9%",
        "characteristics": "Natural, sensible flow"
    },
    "Score 4 (Excellent)": {
        "count": 2012,
        "percentage": "61.8%",
        "characteristics": "Highly consistent, smooth dialogue"
    }
}

The results of this quality assurance process were positive:
- We achieved a good yield rate on first attempts
- Upon review, even dialogues that received lower scores showed acceptable quality
- Nevertheless, I decided to remove all dialogues flagged as low-quality by the AI
- This process reduced our dialogue count from 296 to 242

- **Strong Socratic Presence**: With 98.1% of dialogues at scores 3 or 4, most interactions successfully capture a Socratic tone, indicating the fine-tuning approach is largely effective.
- **Robust Logic**: An even higher proportion (99.7%) of dialogues achieve top marks (scores 3 or 4) in logical consistency, reflecting coherent and natural conversational flow.
- **Room for Refinement**: Although 36.1% of dialogues reach the highest Socratic style mark (score 4), there’s still an opportunity to further enhance the dialogue’s authentically “Socratic” qualities.
- **Minimal Weak Points**: Very few instances of score 2 (1.9% for style and 0.3% for logic), and none at scores 0 or 1, imply that outright failures in either category are negligible.
- **Stable Performance Across Lengths**: The consistent quality regardless of dialogue length suggests a strong and adaptable model foundation, poised for targeted improvements in style nuance if needed.


■System Prompt Integration
This was a particularly challenging decision point. According to the official documentation（https://huggingface.co/google/gemma-2-2b/discussions/28ここにかいてあったこと、というかんじのほうがいいかと）, Gemma has a fundamental design philosophy that:
- Only supports two roles: "user" and "model" (notably lacking a system role)
- Requires dialogues to start from the user side

While many examples on the internet ignored this design philosophy by including system prompt-like elements, I decided to avoid this approach. 

調査をしたうえで、promptがなくてもうまくいきそうだと感じたためです。

While we could have used tuners like XTuner, Axolotl, or LLaMA Factory to implement system prompt-like functionality during training, I prioritized staying aligned with Gemma2's original design philosophy and testing in the most natural way possible.


■ Final Training Data
結果的には以下のようなデータを用意しました。

Number of dialogues extracted from 12 turns: 11 
Number of utterances per dialogue: 2 
Total number of dialogues: 2,662 
Total tokens: 685,875 
Average tokens per dialogue: 257.65 
MAX tokens per dialogue: 527 
MIN tokens per dialogue: 19 
Average user tokens: 144.42 
Average model tokens: 113.24 

""""""""""""""""""""""""""""""トレインフェーズ"""""""""""""""""""""""

■attention mask
ここが一番大きなポイントでした。
レイヤーに働きかけるうえで、
ここはあったほうがいいのかないほうがいいのか？
が自信がなかったため、２パターンつくりました。

■データ量
まずは７０００００トークンの半分にしてみました。
で、チェックポイントを多めにつくりました。


■LoLAの設定
表層レイヤーにはたらきかけるため、に以下のような設定にしました。

https://chatgpt.com/c/67aace53-79d0-800c-b216-7759a84cb911
https://gemini.google.com/app/d69be52592e131ce

■評価セットについて、
実は色々試している過程で検討しましたが、
最終的には各checkpoint毎の出力をAIに評価してもらうという方法にきりかえたため、
最終的には未実装です。




■kaggle環境で走らせるための施策
〇４ビット量子化
これはkaggle環境で走らせるために必要だと考えました。

〇評価データセットの設計
これも高いとkaggleのCPUのRAMの上限をこえてしまうのでkaggleでクラッシュしない範囲にとどめました。
eval_dataset = tokenized_dataset.select(indices[split_idx:split_idx+50])

〇ハイパーパラメータ設定
ここについてはあまりコメントなし

〇max memory
eval_dataset = tokenized_dataset.select(indices[split_idx:split_idx+50])
を５０まで下げ、且つAutoModelForCausalLM.from_pretrainedでmax_memory={0: "4GiB", 1: "4GiB", "cpu": "24GB"}を設定することで
実現しました。

""""""""""""""""""""""""""""""テストフェーズ"""""""""""""""""""""""
■方針
attention maskをあるパターンとないパターンで、５つのチェックポイントでそれぞれ出力をテストしました。
テーマは６個にしました。
品質管理のpromptは以下のようなものです。

統計学的な有意差があるか否か、はもう少しサンプルをとったほうがいいかもしれませんが、
いったん最低限のラインでやりました。


帰無仮説の棄却について
https://chatgpt.com/c/67ace493-a720-800c-97b8-ef608604a34c
https://gemini.google.com/app/866575f90cdf5090
https://claude.ai/chat/e4904c65-688a-413c-9cf1-324612da7f64

オートメーション化の際には以下のような感じにしてクラッシュしないようにしました。


    def __del__(self):
        """デストラクタ: モデルのリソースを解放"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()  # GPUメモリをクリア

コードはgithubにあげているのでそちらを確認してください。


■結果

なんといってもトーンの改善は良い。


アプローチは横ばいでフォーマットが改善したのが意外。
フォーマットは英語の使用だったり、不要な記号の使用がベースが多かった。


出力の質のばらつきもnam、yamともに改善している。


ロジックとアプローチが横ばいかややスコアが落ちたのが残念。
トレードオフ的なものがあるのか？


attentionはいらないかと。


ベストはnamは700, yamは980
トーン40
ロジック25
アプローチ25
フォーマット10

namの７００でいっきにフォーマットが落ちてるのは１回だけ！\\nが入ってしまっているから。おしい

ということでnam700を採用！

■推論

では最終的に私がいいのではないかなと思ったバージョンを使って以下で推論をしてみましょう。



■結論
当初のゴールである”口調変化”と”問い返してくるという挙動”、そしてkaggle内でのトレインは見事達成できました。

■今後やってもいいと考えること、改善できそうな点
・もう少しgemma2のそれぞれのcheckpointの出力の評価はサンプル数を増やしてもいいかと
・目的関数へのタッチもやってみたいかも？
・最初のAIｘAIの学習データの品質チェックや、トレイン中のカスタム評価関数、gemma2の出力の品質評価等、
AIにけっこう頼っているので、もっと数値評価をするアプローチも導入してもいいかと。
・xtunerを使ったsystem promptを作ったトレイン等、モデルアーキテクチャの変更
・今回はcommunityに質問をしたり、をしなかったが、そういった手段は使ってもよかったかと

■最後に



■なぜそもそもコンペに着手をしようと思ったのかの経緯・理由
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

■所感
御覧になっていただいたらわかる通り、
私は主に演繹的なアプローチではなく帰納的なアプローチで今回のタスクに取り組んでます。
ただ、そういった背景が功を奏し、結果的に柔軟なアプローチができたのはないかと感じています。
それが非常に短期間で高いアウトプットにつながったと要因だとかんじています。

宮崎はやおのThe Boy and the Heronは邦題は君たちはどう生きるか、であるということも


