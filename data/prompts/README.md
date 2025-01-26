# Prompt Templates Documentation

This document provides documentation for the prompt design philosophy and structure used in the Socratic dialogue generation system.

## Core Design Philosophy

1. **Asymmetric Dialogue Design**:
   - Assistant (Socrates): Maintains philosophical consistency through a single, carefully crafted prompt
   - User: Implements diverse perspectives through modular prompt components

2. **Modular Prompt Architecture**:
   - Base prompts: Define core behaviors
   - Overlay components: Add personality and perspective variations
   - Control parameters: Fine-tune dialogue dynamics

3. **Dialogue Quality Control**:
   - Natural conversation flow priority
   - Minimal theatrical elements
   - Consistent relationship dynamics

## Prompt Categories

### 1. Assistant System Prompts (`assistant_system_prompt/`)

The assistant role is intentionally designed with a singular, consistent personality as Socrates, unlike the user-side which has multiple variations. This design choice ensures authenticity and consistency in the Socratic method throughout all dialogues.

#### assistant_system_prompt.json
- **Purpose**: Defines the core Socratic personality and methodology
- **Key Elements**:
  - Character definition as Socrates
  - Core behavioral guidelines for Socratic dialogue
  - Specific instructions for maintaining Socratic authenticity:
    - Use of gentle questioning patterns (e.g., question-ending particles in Japanese)
    - Observational stance in speech
    - Avoidance of definitive statements
    - Concise and smart questioning style
    - Natural conversational flow without bullet points
    - Appropriate relationship distance through pronouns

#### response.json and update.json
These files are placeholder JSON files created during the prompt development process but are not actively used in the current implementation:
- No {{RESPONSE}} or {{UPDATE}} placeholders are used in the actual prompts
- However, their IDs must still be specified in automation.csv for system control

### 2. User System Prompts (`user_system_prompt/`)

#### user_system_prompt.json
After extensive testing and iterations, this prompt was standardized with a single template:
- **Purpose**: Defines the core behavior of Socrates' dialogue partner
- **Key Elements**:
  - Establishes the dialogue context with Socrates
  - Incorporates the initial philosophical question ({{INITIAL_QUESTION}})
  - Integrates with persona variations ({{PERSONA}})
- **Behavioral Guidelines**:
  - Explicit communication about unclear questions or difficulties
  - Natural, understated responses without theatrical elements
  - Concise and minimal responses
  - Use of "あなた" to address Socrates
  - Maintains natural conversational flow without stage directions or bullet points

#### persona.json
- **Purpose**: Carefully curated collection of 148 distinct personality profiles
- **Implementation**:
  - Generated using Claude with consistent formatting template
  - Each profile includes detailed characteristics, approach style, and behavioral tendencies
  - Each personality was used twice (temperature 0.3 and 0.7) creating 296 unique dialogues

**Strategic Persona Distribution (148 profiles)**:
1. **General Population Representatives (68 profiles)**
   - Designed to reflect diverse societal perspectives and dialogue styles
   - Includes both receptive and challenging dialogue partners
   - Examples: "Research Scientist (University Researcher, 30s)"
        - Approach: Physical/Biological perspective
        - Characteristics: Values logical thinking, scientific evidence-based judgment
        - Tendency to analyze self through objective, data-driven lens

2. **Historical Figures (40 profiles)**
   - Carefully selected based on:
     - Relevance to self-identity discourse
     - Copyright considerations
     - Avoiding sensitive religious/cultural icons
     - Sufficient presence in LLM training data
   - Example: "Nietzsche"
     - Core philosophies: Will to power, value transvaluation
     - Key concepts: Übermensch, eternal recurrence
     - Methodological approach: Genealogical analysis of values
     - Characteristic dualities: Apollonian-Dionysian dynamics

3. **Modern Interpretations (40 profiles)**
   - Contemporary characters embodying historical philosophical perspectives
   - Example: "Social Media Marketing Consultant (35)"
     - Approach: Psychoanalytic perspective
     - Focus: Analysis of unconscious desires
     - Key belief: Questions rationality in decision-making
     - Application: Modern context for classical psychological theories

#### others.json & transform.json
These files are placeholder JSON files created during the prompt development process but are not actively used in the current implementation:
- No {{OTHERS}} or {{TRANSFORM}} placeholders are used in the actual prompts
- However, their IDs must still be specified in automation.csv for system control

### 3. Initial Questions (`questions.json`)
The system uses a carefully curated set of philosophical questions, though with a specific implementation approach:

#### Primary Implementation
- **Fixed Opening Question**: 
  The inference system (`test.py`) consistently starts with a specific question about "self". Below is the Japanese prompt with its core meaning preserved:
  ```
  "やぁ、よく来てくれたね。今日は『自分』という、これ以上ないほど身近な存在でありながら、
  あまり話すことのないトピックについて話そうではないか。人は「自分の意思で決めた」や、
  「自分らしさ」というような具合に、日々「自分」という言葉を多くの場面で使っておるが、
  そもそも「自分」という言葉を使っているとき、君は何を指していると思うかね？"
  ```
  (This opening question invites the dialogue partner to consider what they mean when they use the word "self", noting how we frequently use this concept in daily life without deeply examining its meaning)

#### Training Data Diversity
- **Anti-Overfitting Strategy**: While the inference system uses a fixed question, the training data intentionally incorporates diverse topics
- **Curated Question Pool**: 74 carefully selected philosophical topics suitable for Socratic dialogue
- **Topic Categories**:
  - Fundamental concepts (happiness, justice, beauty, freedom, truth)
  - Human experience (love, death, solitude, fear, dreams)
  - Social concepts (education, civilization, tradition, culture)
  - Ethical inquiries (goodness, justice, authority)
  - Existential questions (meaning of life, fate, time)
  - Contemporary issues (technology, globalization)
