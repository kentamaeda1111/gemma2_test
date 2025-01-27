# Japanese-Speaking Socratic Gemma

A Kaggle competition submission demonstrating fine-tuning of Gemma-2b for Japanese Socratic dialogue generation. View the [competition notebook](https://www.kaggle.com/code/kentamaeda/japanese-speaking-socratic-gemma) for detailed results.

## Project Impact

- Successfully trained Gemma-2b to conduct philosophical dialogues in Japanese while maintaining Socratic methodology
- Developed an innovative dual-AI system for automated training data generation
- Created a scalable pipeline for high-quality Japanese dialogue generation
- Implemented efficient model training using LoRA and 4-bit quantization

## Technical Innovations

### Automated Data Generation
- Engineered a dual-AI system using Claude API for realistic student-teacher interactions
- Implemented systematic quality assessment for dialogue filtering
- Developed configurable conversation parameters for diverse dialogue patterns

### Model Optimization
- Achieved efficient fine-tuning through LoRA implementation
- Reduced memory footprint using 4-bit quantization (QLoRA)
- Optimized inference for deployment on consumer-grade hardware
- Maintained model quality while reducing resource requirements

### Quality Control
- Developed automated metrics for dialogue quality assessment
- Implemented systematic filtering for training data selection
- Created comprehensive evaluation pipeline for model outputs

## Quick Start

```bash
# Clone and setup
git clone https://github.com/kentamaeda1111/gemma2_test.git
cd gemma2_test
pip install -r requirements.txt

# Configure API keys
cp .env.template .env  # Edit with your keys:
# - CLAUDE_API_KEY_1/2 (dialogue generation)
# - HUGGINGFACE_API_KEY (model access)

# Run pipeline
python -m src.data.generation.automation
python -m src.models.training.train
python -m src.models.inference.test
```

## Technical Requirements

### For Training
- GPU: NVIDIA GPU with 24GB+ VRAM (A5000/A6000/A100)
- RAM: 32GB minimum
- Storage: 50GB+ free space
- Training Time: ~2.5 hours on A100

### For Inference
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 16GB minimum
- Storage: 20GB free space

## Project Structure

```
src/
├── data/
│   ├── generation/     # Dialogue generation system
│   └── processing/     # Quality assessment tools
├── models/
│   ├── training/       # Fine-tuning implementation
│   └── inference/      # Testing interface
└── utils/             # Shared utilities
```

## Documentation

Detailed documentation available in component READMEs:
- [Generation System](src/data/generation/README.md)
- [Data Processing](src/data/processing/README.md)
- [Model Training](src/models/training/README.md)
- [Inference System](src/models/inference/README.md)

## License

MIT License - Free for reference and educational use.