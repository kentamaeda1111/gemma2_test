# Overall Strategy
Given the limited time, I needed an efficient approach. We decided to:
1. Create multiple models by varying the training data
2. Select the best performing model
Therefore, data generation became a crucial factor in this project.

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
Using Socratic literature directly wasn't practical due to:
- Copyright issues (especially for Japanese content)
Therefore, I decided to:
- Automate AI-to-AI dialogue generation

# Training Data Volume
Research on appropriate volume was conducted through:
- Gemma and Gemma 2B related documentation and discussions on Hugging Face and GitHub
- Similar model documentation (Mistral 2.3B, Falcon 1.5B, OpenLLaMA 2.7B, XGen 2.2B, RedPajama-INCITE 3B)
- Tuner documentation (XTuner, Axolotl, LLaMA Factory)
- Kaggle code related to Gemma (especially 2B-it) fine-tuning
- Web research (using Gemini Advanced 1.5 Pro with deep research, Perplexity, Felo)
*Note: Due to potential hallucination risks, source verification was strictly enforced
- Academic papers (using SciSpace, Consensus, Elicit)

Based on research, we aimed for:
- Total tokens: 20,000 - 500,000 
- Average dialogue length: 50-300 tokens

Considering that 20% would be used for testing and anticipating some low-quality data, we set a target of approximately 700,000 tokens for training data. As a result, we generated 296 dialogue sets with 12 turns each using AI×AI interaction.

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
Generated 9 variations with:
- Total tokens: 22,753-685,875 (80% of these values)
- Average tokens per dialogue: 76.87-310.24
- System prompt: None + 2 variations
- All user-initiated

Note: Models 4-9 inadvertently included lower-quality data

# Comparison of Training Data Characteristics Across Model Configurations

| Item | model1 | model2 | model3 | model4 | model5 | model6 | model7 | model8 | model9 |
|------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Number of dialogues extracted from 12 turns | 11 | 11 | 11 | 1 | 1 | 1 | 1 | 1 | 1 |
| Number of utterances per dialogue | 2 | 2 | 2 | 2 | 4 | 2 | 2 | 4 | 4 |
| Total number of dialogues | 2,662 | 2,662 | 2,662 | 296 | 296 | 296 | 296 | 296 | 296 |
| Total tokens | 707,115 | 752,369 | 685,875 | 22,753 | 84,431 | 25,121 | 30,153 | 86,799 | 91,831 |
| Average tokens per dialogue | 265.63 | 282.63 | 257.65 | 76.87 | 285.24 | 84.87 | 101.87 | 293.24 | 310.24 |
| MAX tokens per dialogue | 535 | 552 | 527 | 129 | 416 | 137 | 154 | 424 | 441 |
| MIN tokens per dialogue | 27 | 44 | 19 | 36 | 181 | 44 | 61 | 189 | 206 |
| Average user tokens* | 152.39 | 169.39 | 144.42 | 12 | 67.63 | 20 | 37 | 71.63 | 80.13 |
| Average model tokens* | 113.24 | 113.24 | 113.24 | 64.87 | 74.99 | 64.87 | 64.87 | 74.99 | 74.99 |
| System prompt | "Socrates." | "You are Socrates, the ancient Greek philosopher." | None | None | None | "Socrates." | "You are Socrates, the ancient Greek philosopher." | "Socrates." | "You are Socrates, the ancient Greek philosopher." |
