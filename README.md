# EigenScore: Hallucination Detection Using LLMs' Internal States

This repository contains the implementation for the paper "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection" (ICLR 2024). 

EigenScore analyzes LLMs' internal hidden states to detect when models are hallucinating, providing a more reliable way to assess the trustworthiness of model-generated content.


## Setup Instructions

### Requirements

1. Python 3.8+ 
2. PyTorch 1.10+
3. Transformers 4.20+
4. Required Python packages:

```bash
pip install torch transformers sentence-transformers rouge-score numpy scikit-learn torchmetrics
```

Or install all required packages using the provided requirements.txt:

```bash
pip install -r requirements.txt
```

### Directory Structure

You need to set up the following directory structure:

```
└── eigenscore/
    ├── data/
    │   ├── weights/         # Pre-trained model weights
    │   │   ├── nli-roberta-large/
    │   │   ├── bert-base/
    │   │   └── [model_name]/  # Language models (llama-7b-hf, etc.)
    │   ├── datasets/        # Datasets
    │   │   ├── coqa/
    │   │   ├── triviaqa/
    │   │   └── nq_open/
    │   └── output/          # Generated outputs
    └── ...
```

### Configuration

1. Modify `_settings.py` to use local paths:

```python
import os

_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODEL_PATH = os.path.join(_BASE_DIR, 'weights')
DATA_FOLDER = os.path.join(_BASE_DIR, 'datasets')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)
```

2. Setup model weights:
   - Download SentenceTransformer model: `nli-roberta-large`
   - Download BERT model: `bert-base-uncased`
   - Download the LLM you want to use (llama-7b-hf, llama-13b-hf, etc.)

3. Configure GPU settings:
   - Modify `--device` in the command line or set to 'cpu' if no GPU is available
   - Update any hardcoded device settings in the code (models/__init__.py, func/metric.py)

### Dataset

The code supports the following datasets:
- CoQA 
- TriviaQA
- NQ-Open
- SQuAD
- TruthfulQA

Download the datasets and place them in the `data/datasets/` directory.

## Running the Code

### Basic Usage

```bash
python -m pipeline.generate --model llama-7b-hf --dataset coqa --device cuda:0 --fraction_of_data_to_use 0.1
```

### Parameters

- `--model`: Model name (llama-7b-hf, llama-13b-hf, falcon-7b, etc.)
- `--dataset`: Dataset name (coqa, triviaqa, nq_open, SQuAD, TruthfulQA)
- `--device`: Device to run on (cuda:0, cpu, etc.)
- `--fraction_of_data_to_use`: Fraction of dataset to use (0.0-1.0)
- `--num_generations_per_prompt`: Number of generations per prompt (default: 10)
- `--temperature`: Sampling temperature (default: 0.5)
- `--decoding_method`: Decoding method (greedy, beam_search) 
- `--top_p`: Top-p sampling parameter (default: 0.99)
- `--top_k`: Top-k sampling parameter (default: 10)
- `--seed`: Random seed (default: 2023)

## Testing Datasets

We've added a dataset testing functionality to help verify that datasets are loading correctly and to analyze dataset properties. This is helpful when adding new datasets or troubleshooting issues.

### Running Dataset Tests

You can test datasets using the `test_datasets.py` script in the `dataeval` directory:

```bash
# Test all supported datasets
python -m dataeval.test_datasets

# Test a specific dataset
python -m dataeval.test_datasets --datasets squad

# Test with only a fraction of the data
python -m dataeval.test_datasets --fraction 0.1

# Generate distribution plots
python -m dataeval.test_datasets --plot
```

### Running on HPC Clusters

For running tests on HPC clusters with Slurm, use the sbatch script:

```bash
sbatch dataeval/sbatch_test_datasets.sh
```

The sbatch script automatically:
- Sets up the necessary environment
- Installs required dependencies 
- Configures HuggingFace credentials
- Creates required directories

### Test Outputs

The testing script provides detailed analysis of each dataset:
- Size and basic properties
- Column types and content analysis
- Prompt format examples and length statistics
- Tokenization analysis
- Generation configuration details

Results are saved to both log files and a JSON file for further analysis.

## Using EigenScore in Your Code

To use EigenScore for hallucination detection in your own projects:

```python
from func.metric import getEigenIndicator_v0, getEigenIndicatorOutput
from sentence_transformers import SentenceTransformer

# Load model and generate text with output_hidden_states=True
hidden_states = model_outputs.hidden_states
num_tokens = [len(text) for text in generated_texts]  # Calculate token lengths

# Get EigenScore from internal states
eigenScore, eigenValues = getEigenIndicator_v0(hidden_states, num_tokens)

# Get output-based EigenScore (using generated text)
SenSimModel = SentenceTransformer('nli-roberta-large')
eigenScoreOutput, _ = getEigenIndicatorOutput(generated_texts, SenSimModel)

# Higher EigenScore indicates higher likelihood of hallucination
```

## Citation

```
@article{chen2024inside,
  title={INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection},
  author={Chen, Chao and Liu, Kai and Chen, Ze and Gu, Yi and Wu, Yue and Tao, Mingyuan and Fu, Zhihang and Ye, Jieping},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```

## Contact

For questions about the paper or code, please contact:
- Chao Chen: chench@zju.edu.cn / ercong.cc@alibaba-inc.com

