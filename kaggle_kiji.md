

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
且つ、底上げの余地をできるだけ大きくしたかったため、
gemma2を選ぶことにしました。

もちろん本質も追及したいとは思ったのですが、
繰り返しになりますが、ベースモデルの質がそこまでたかくないため、
ソクラテス問答の本質となる”問う力”みたいなものは期待できなかったため、
ソクラテスのような”問い”で返す且つソクラテスのような口調で喋るＡＩ、
を暫定的なゴールに据えることにし、
そのうえでいかに質を高められるか、を目指しました。

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
（explanable AIの取り組み）
The results of this quality assurance process were positive:
- We achieved a good yield rate on first attempts
- Upon review, even dialogues that received lower scores showed acceptable quality
- Nevertheless, I decided to remove all dialogues flagged as low-quality by the AI
- This process reduced our dialogue count from 296 to 242

# System Prompt Integration
This was a particularly challenging decision point. According to the official documentation（https://huggingface.co/google/gemma-2-2b/discussions/28ここにかいてあったこと、というかんじのほうがいいかと）, Gemma has a fundamental design philosophy that:
- Only supports two roles: "user" and "model" (notably lacking a system role)
- Requires dialogues to start from the user side

While many examples on the internet ignored this design philosophy by including system prompt-like elements, I decided to avoid this approach. 

Instead, I decided to create two variations of training data:
1. One completely without any system prompt-like elements
2. Another incorporating phrase "You are Socrates, the ancient Greek philosopher." before user utterances

While we could have used tuners like XTuner, Axolotl, or LLaMA Factory to implement system prompt-like functionality during training, I prioritized staying aligned with Gemma2's original design philosophy and testing in the most natural way possible.

# Final Training Data
結果的には2つのデータを以下のように用意しました。
違いはsystem promptのような文言を挿入しているか否か、という点のみです。

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

４ビット量子化の設定や、LoLAを使う、という方針はkaggle環境でgemma-2の2Bをトレインするためには必須だと考えました。

https://chatgpt.com/c/67aace53-79d0-800c-b216-7759a84cb911
https://gemini.google.com/app/d69be52592e131ce

設定の仕方やどの程度が妥当か、はデータ生成時にも参考にした以下のような文献・媒体にあたりました。

- Gemma and Gemma 2B related documentation and discussions on Hugging Face and GitHub
- Similar model documentation (Mistral 2.3B, Falcon 1.5B, OpenLLaMA 2.7B, XGen 2.2B, RedPajama-INCITE 3B)
- Tuner documentation (XTuner, Axolotl, LLaMA Factory)
- Kaggle code related to Gemma (especially 2B-it) fine-tuning
- Web research (using Gemini Advanced 1.5 Pro with deep research, Perplexity, Felo)
*Note: Due to potential hallucination risks, source verification was strictly enforced
- Academic papers (using SciSpace, Consensus, Elicit)

ただ、モデルのパフォーマンスを大きく変える要因になったのは
4ビット量子化やLoLAやTrainingArgumentsの設定ではありませんでした。

どれだけチューニングをしても、出力されるものはだめだめでした。（ここはもっと記述をする）
非常にうまくいった戦略がattention機構への働きかけです。
これをする前と後ではかなり差がありました。

あと評価メトリクスの作成も苦戦しました。
結果的にここはあきらめました。
評価メトリクスではよさそうでも実際つかってみたら、そうでもない、ということが連続して、
最終的にはlearning rateとlossだけにしました。

あともう一つ苦戦した点がkaggle環境ではしらせれるようにすることです。
GPUは大丈夫だったのですが、CPUがどうしてもevaluateのところで２９GBを越え、
クラッシュをしてしまいました。

最終的にはeval_dataset = tokenized_dataset.select(indices[split_idx:split_idx+50])
を５０まで下げ、且つAutoModelForCausalLM.from_pretrainedでmax_memory={0: "4GiB", 1: "4GiB", "cpu": "24GB"}を設定することで
実現しました。


テスト時はデータ生成フェーズでも申し上げた通り
問いを固定する形にしました。ただ、汎用性の高さをチェックするためトレインデータには入ってないものをあえて選びました。
最終的にはsystem promptのようなものを組み込むパターンとそうでないパターンで二つやりましたが、
感覚的にはsystem_promptがないほうがうまくいっていたような気がします。
ここはもっと検証する必要があるでしょう。
ただ、ここから言えることはsystem_promptは必ずしもインパクトが大きくなさそう、という点です。


帰無仮説の棄却について
https://chatgpt.com/c/67ace493-a720-800c-97b8-ef608604a34c
https://gemini.google.com/app/866575f90cdf5090
https://claude.ai/chat/e4904c65-688a-413c-9cf1-324612da7f64


■結論

当初のゴールである”口調変化”と”問い返してくるという挙動”、そしてkaggle内でのトレインは見事達成できました（１０時間くらいかかりますが。。。（笑））。

評価メトリクスに関して全然うまくいかなかった。結局最後は試してみる、という結論だった。最終的には検証するしかない、と。
今後やること　testのpromptや状態保存をもっといじれると思う
quality checkの自動化はやってもいいかと
もっと底上げをする余地があるとすると、目的関数へのタッチかな？


尚、御覧になっていただいたらわかる通り、
私は主に演繹的なアプローチではなく帰納的なアプローチで今回のタスクに取り組んでます。
ただ、そういった背景が功を奏し、結果的に柔軟なアプローチができたのはないかと感じています。
それが非常に短期間で高いアウトプットにつながったと要因だとかんじています。

ただ、一番うまく言ったモデルのパターンでも、
こちらが質問をした場合は、回答はしてくれず、問いで返す、という挙動になる。全然だめ

■もう少しやれたこと
・僕が作ったモデルとclaudeを会話させるプログラムを使って品質評価をしたかった
・データ生成時の品質調査
・モデルの品質チェック
・モデルアーキテクチャの変更（promptを使うパターン）
・今回はcommunityに質問をしたり、をしなかったが、そういった手段は使ってもよかったかと
・inference.testのほうで、あなたはソクラテスです、と頭に全部つけるほうがよｋったかも？

LORAの設定をかえてみる

初期の問いを変えてみる




