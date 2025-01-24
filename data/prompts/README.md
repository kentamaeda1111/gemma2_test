# Prompt Templates Documentation

This document provides English documentation for the Japanese prompt templates used in the dialogue generation system.

## Overview

The prompt system is designed to create Socratic dialogues with varying characteristics:
- Different student personas and knowledge levels
- Varying degrees of emotional engagement
- Progressive development of philosophical understanding

## Prompt Categories

### 1. Assistant System Prompts (`assistant_system_prompt/`)

#### assistant_system_prompt.json
- **Purpose**: Core behavioral guidelines for the Socrates role
- **Key Elements**:
  - Socratic questioning techniques
  - Response adaptation based on student understanding
  - Balance between guidance and discovery

#### response.json
- **Purpose**: Specific response patterns for different dialogue stages
- **Key Elements**:
  - Question formulation patterns
  - Acknowledgment phrases
  - Deep inquiry triggers

#### update.json
- **Purpose**: Guidelines for conversation progression
- **Key Elements**:
  - Timing for deeper questions
  - Recognition of student progress
  - Conversation flow management

### 2. User System Prompts (`user_system_prompt/`)

#### user_system_prompt.json
- **Purpose**: Base templates for student responses
- **Variations**:
  - Emotional engagement levels (neutral to engaged)
  - Response length control
  - AI awareness settings

#### persona.json
- **Purpose**: Different student character profiles
- **Types**:
  - Philosophy novice
  - Analytical thinker
  - Intuitive responder
  - etc.

#### others.json & transform.json
- **Purpose**: Additional response modifications
- **Features**:
  - Conversation style adjustments
  - Progressive understanding indicators
  - Personality trait expressions

### 3. Initial Questions (`questions.json`)
- Collection of philosophical questions used to initiate dialogues
- Categories include:
  - Existence and reality
  - Knowledge and truth
  - Ethics and morality
  - etc.

## Implementation Notes

1. **Temperature Settings**:
   - Lower settings (0.3-0.5): More consistent, analytical responses
   - Higher settings (0.6-0.7): More varied, creative responses

2. **Prompt Combinations**:
   - Different combinations create unique dialogue patterns
   - Tested for natural conversation flow
   - Optimized for Japanese language nuances

3. **Quality Control**:
   - Prompts designed to maintain philosophical depth
   - Balance between accessibility and sophistication
   - Consistent with Socratic method principles 