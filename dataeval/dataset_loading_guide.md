# EigenScore Dataset Loading Guide

This document provides detailed information about how datasets are loaded, processed, and formatted for the EigenScore hallucination detection framework.

## Table of Contents

- [Overview](#overview)
- [Common Dataset Processing Pipeline](#common-dataset-processing-pipeline)
- [CoQA Dataset](#coqa-dataset)
- [TriviaQA Dataset](#triviaqa-dataset)
- [Natural Questions (NQ-Open)](#natural-questions-nq-open)
- [SQuAD Dataset](#squad-dataset)
- [TruthfulQA Dataset](#truthfulqa-dataset)
- [Testing Datasets](#testing-datasets)
- [Custom Dataset Integration](#custom-dataset-integration)

## Overview

EigenScore supports multiple datasets for hallucination detection evaluation:

| Dataset | Type | Description | Reference Files |
|---------|------|-------------|-----------------|
| CoQA | Conversational QA | Multi-turn conversational question answering with context | `dataeval/coqa.py` |
| TriviaQA | Factoid QA | Factual questions with evidence documents | `dataeval/triviaqa.py` |
| NQ-Open | Open-domain QA | Questions from real Google search queries | `dataeval/nq_open.py` |
| SQuAD | Reading Comprehension | Questions based on Wikipedia articles | `dataeval/SQuAD.py` |
| TruthfulQA | Truthfulness | Questions designed to elicit falsehoods | `dataeval/TruthfulQA.py` |

## Common Dataset Processing Pipeline

All datasets follow a similar loading and processing flow:

1. **Raw Data Loading**: Each dataset module has functions to load the raw data from files
2. **Preprocessing**: Convert raw data to a standardized format
3. **Tokenization**: Create inputs for the language model
4. **Prompt Construction**: Format the data as prompts suitable for generation
5. **Filtering**: Apply fractional sampling if requested
6. **Configuration**: Set up generation constraints (e.g., EOS tokens, bad words)

## CoQA Dataset

### Loading Process

```python
# From dataeval/coqa.py
def get_dataset(tokenizer, split='validation'):
    dataset = datasets.load_from_disk(_save_dataset())
    def encode_coqa(example):
        example['answer'] = example['answer']['text']
        example['prompt'] = prompt = example['story'] + ' Q: ' + example['question'] + ' A:'
        return tokenizer(prompt, truncation=False, padding=False)
    dataset = dataset.map(encode_coqa, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], 
                       output_all_columns=True)
    return dataset
```

### Data Format

- **Raw Format**: JSON file (`coqa-dev-v1.0.json`)
- **Processed Format**: HuggingFace Dataset with fields:
  - `story`: Context passage 
  - `question`: User question
  - `answer`: Dictionary with answer text and position
  - `additional_answers`: List of alternative answers (3 per question)
  - `id`: Unique identifier for each QA pair

### Prompt Construction

CoQA uses a simple format that combines context, question, and answer prompt:

```
{context} Q: {question} A:
```

Example:
```
John went to the store to buy milk. He also picked up some eggs and bread. 
Q: What did John buy at the store? 
A:
```

### Generation Configuration

```python
def _generate_config(tokenizer):
    # Define EOS tokens - periods, newlines, etc.
    eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889] 
    eos_token_id += [tokenizer.eos_token_id]
    
    # Prevent model from generating more questions
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] 
                           for eos_token in question_framing_ids]
    
    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)
```

## TriviaQA Dataset

### Loading Process

```python
# From dataeval/triviaqa.py
def get_dataset(tokenizer, split='validation'):
    data = datasets.load_dataset("trivia_qa", "rc.nocontext", split=split)
    
    # Remove duplicates
    id_mem = set()
    def remove_dups(batch):
        if batch['question_id'][0] in id_mem:
            return {_:[] for _ in batch.keys()}
        id_mem.add(batch['question_id'][0])
        return batch
    data = data.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    
    # Process into model inputs
    data = data.map(lambda _: process_data_to_model_inputs(_, tokenizer),
                    batched=True, batch_size=10, load_from_cache_file=False,
                    remove_columns=["search_results", "question_source", "entity_pages"])
    
    # Set format for model
    data.set_format(type="torch", 
                   columns=["input_ids", "attention_mask", "decoder_input_ids", 
                           "decoder_attention_mask", "labels"],
                   output_all_columns=True)
    return data
```

### Data Format

- **Raw Format**: HuggingFace dataset (`trivia_qa`)
- **Processed Format**: HuggingFace Dataset with fields:
  - `question`: The trivia question
  - `answer`: List of acceptable answers
  - `question_id`: Unique identifier
  - `input_ids`, `attention_mask`: Tokenized inputs
  - `labels`: Tokenized answers for evaluation
  - `prompt`: Formatted prompt with examples

### Prompt Construction

TriviaQA uses a one-shot learning format:

```python
def sample_to_prompt(sample, **kwargs):
    return f"""Answer these questions:
Q: In Scotland a bothy/bothie is a?
A: House
Q: {sample['question']}
A:"""
```

Example:
```
Answer these questions:
Q: In Scotland a bothy/bothie is a?
A: House
Q: Who was the first American in space?
A:
```

### Generation Configuration

```python
def _generate_config(tokenizer):
    # EOS tokens: newlines, commas, periods
    eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
    eos_token_id += [tokenizer.eos_token_id]
    
    # Prevent model from continuing with more questions
    bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['Q:']]
    
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)
```

## Natural Questions (NQ-Open)

### Loading Process

```python
# From dataeval/nq_open.py
def get_dataset(tokenizer, split='validation'):
    # Load from cached dataset or create
    data_files = {"train": f"{_settings.DATA_FOLDER}/nq-open-dlhell/NQ-open.dev.jsonl"}
    raw_datasets = datasets.load_dataset('json', data_files=data_files)
    
    # Process into model inputs
    column_names = raw_datasets["train"].column_names
    dataset = raw_datasets["train"].map(
        lambda examples: prepare_validation_features(examples, tokenizer),
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    
    # Format for the model
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], 
                      output_all_columns=True)
    return dataset
```

### Data Format

- **Raw Format**: JSONL file with question-answer pairs
- **Processed Format**: Dataset with fields:
  - `question`: Question text
  - `answer`: List of accepted answers
  - `id`: Unique identifier
  - `prompt`: Formatted prompt with the question

### Prompt Construction

NQ-Open uses a simple question-answer format with explicit instructions:

```python
def prepare_validation_features(examples, tokenizer):
    # Create prompts
    prompts = []
    for i in range(len(examples['question'])):
        prompt = f"""Answer the following question:
Question: {examples['question'][i]}
Answer:"""
        prompts.append(prompt)
    
    # Tokenize and format
    tokenized_examples = tokenizer(prompts, truncation=False, padding=False)
    examples['prompt'] = prompts
    examples['input_ids'] = tokenized_examples['input_ids']
    examples['attention_mask'] = tokenized_examples['attention_mask']
    return examples
```

Example:
```
Answer the following question:
Question: Who won the Nobel Prize in Physics in 2021?
Answer:
```

## SQuAD Dataset

### Loading Process

```python
# From dataeval/SQuAD.py
def get_dataset(tokenizer, split='validation'):
    # Load squad dataset
    dataset = datasets.load_dataset("squad", split="validation")
    
    # Process into model inputs with proper formatting
    dataset = dataset.map(
        lambda examples: tokenize_squad(examples, tokenizer),
        batched=True,
        load_from_cache_file=False,
    )
    
    # Format for model input
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], 
                      output_all_columns=True)
    return dataset
```

### Data Format

- **Raw Format**: HuggingFace SQuAD dataset
- **Processed Format**: Dataset with fields:
  - `context`: The passage containing the answer
  - `question`: The question to answer
  - `answers`: Dictionary with answer text and position
  - `prompt`: Formatted prompt combining context and question
  - `id`: Unique identifier

### Prompt Construction

SQuAD prompts provide the context and question:

```python
def tokenize_squad(examples, tokenizer):
    # Construct prompts with context and question
    prompts = []
    for i in range(len(examples['context'])):
        prompt = f"Context: {examples['context'][i]}\nQuestion: {examples['question'][i]}\nAnswer:"
        prompts.append(prompt)
    
    # Tokenize inputs
    tokenized_examples = tokenizer(prompts, truncation=False, padding=False)
    examples['prompt'] = prompts
    examples['input_ids'] = tokenized_examples['input_ids']
    examples['attention_mask'] = tokenized_examples['attention_mask']
    return examples
```

Example:
```
Context: The Manhattan Project was a research and development undertaking during World War II...
Question: When was the Manhattan Project started?
Answer:
```

## TruthfulQA Dataset

### Loading Process

```python
# From dataeval/TruthfulQA.py
def get_dataset(tokenizer, split='validation'):
    # Load from JSON file or processed cache
    dataset_path = f'{_settings.DATA_FOLDER}/TruthfulQA/truthful_qa.json'
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Convert to HuggingFace dataset format
    dataset = Dataset.from_dict({
        'question': [item['question'] for item in data],
        'answer': [item['correct_answers'][0] if item['correct_answers'] else "" for item in data],
        'additional_answers': [item['correct_answers'][1:] if len(item['correct_answers']) > 1 else [""] for item in data],
        'id': [f"truthfulqa_{i}" for i in range(len(data))]
    })
    
    # Process into model inputs
    dataset = dataset.map(lambda example: tokenize_truthfulqa(example, tokenizer), 
                         batched=False, load_from_cache_file=False)
    
    # Format for model
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], 
                      output_all_columns=True)
    return dataset
```

### Data Format

- **Raw Format**: JSON file with questions and correct/incorrect answers
- **Processed Format**: Dataset with fields:
  - `question`: The question designed to test truthfulness
  - `answer`: Primary correct answer
  - `additional_answers`: Alternative correct answers
  - `id`: Unique identifier
  - `prompt`: Formatted prompt with question

### Prompt Construction

TruthfulQA uses a direct question format:

```python
def tokenize_truthfulqa(example, tokenizer):
    # Create prompt with just the question
    prompt = f"Q: {example['question']}\nA:"
    
    # Tokenize
    tokenized = tokenizer(prompt, truncation=False, padding=False)
    example['prompt'] = prompt
    example['input_ids'] = tokenized['input_ids']
    example['attention_mask'] = tokenized['attention_mask']
    return example
```

Example:
```
Q: Do vaccines cause autism?
A:
```

## Testing Datasets

EigenScore includes a comprehensive dataset testing framework located in `dataeval/test_datasets.py` to validate and analyze datasets before using them for hallucination detection.

### Test Suite Overview

The test suite verifies:
- Dataset loading and basic properties
- Column types and content
- Prompt formatting
- Tokenization statistics and validation
- Generation configuration

### Running Tests

You can run the test suite in several ways:

```bash
# Test all supported datasets
python -m dataeval.test_datasets

# Test specific datasets
python -m dataeval.test_datasets --datasets coqa triviaqa

# Test only SQuAD
python -m dataeval.test_datasets --squad

# Test with a fraction of the data (useful for large datasets)
python -m dataeval.test_datasets --fraction 0.1

# Generate plots of token length distributions
python -m dataeval.test_datasets --plot
```

For HPC environments, use the provided sbatch script:

```bash
sbatch dataeval/sbatch_test_datasets.sh
```

### Test Output and Analysis

The test suite generates several outputs:

1. **Log file**: `dataset_test_results_llama2_7b_chat.log` containing detailed test results
2. **JSON report**: `dataset_test_results_llama2_7b_chat_hf.json` with structured results for analysis
3. **Optional plots**: Token distribution histograms if `--plot` flag is used

#### Example Test Output

```
====================================================
Testing COQA dataset loading with Llama-2-7b-chat-hf...
Dataset loaded successfully with 7983 examples
Sampling 5 examples...
Dataset columns: ['id', 'prompt', 'answer', 'question', 'story', 'input_ids', 'attention_mask']
Column analysis: {
  "id": {"type": "str"},
  "prompt": {"type": "str", "avg_length": "731.22 chars"},
  "answer": {"type": "str", "avg_length": "6.55 chars"},
  "question": {"type": "str", "avg_length": "42.81 chars"},
  "story": {"type": "str", "avg_length": "342.41 chars"}
}

Prompt format examples:
Example 1:
Jessica went to sit in her rocking chair. Today was her birthday and she was turning 80...

Analyzing tokenization...
Token length stats: Avg=142.17, Min=50, Max=623
Examples potentially needing truncation (>2048 tokens): 0 (0.00%)

Generation config: {
  "eos_token_id": [29889, 13, 0],
  "bad_words_ids": [[1055], [187, 1055], [13], [835], [187, 835], [936]]
}
```

### Common Test Issues and Solutions

1. **Missing Datasets**:
   - Ensure datasets are downloaded to the correct location
   - Check paths in `_settings.py`

2. **Tokenizer Errors**:
   - Add `HUGGING_FACE_HUB_TOKEN` to your environment
   - Ensure model weights are properly downloaded

3. **Long Sequence Issues**:
   - Look for "Examples potentially needing truncation" in the output
   - Consider implementing truncation in the dataset module

4. **Generation Config Errors**:
   - Ensure the dataset module has a proper `_generate_config` function
   - Check capitalization in module names (e.g., "SQuAD" vs "squad")

## Custom Dataset Integration

To add your own dataset to EigenScore, follow these steps:

1. **Create a new module** in the `dataeval` directory (e.g., `my_dataset.py`)

2. **Implement the required functions**:
   - `get_dataset(tokenizer, split='validation')`: Returns a HuggingFace Dataset
   - `_generate_config(tokenizer)`: Returns generation constraints

3. **Define the prompt format** appropriate for your dataset

4. **Add to pipeline/generate.py**:
   ```python
   def get_dataset_fn(data_name):
       # Add your dataset here
       if data_name == 'my_dataset':
           return my_dataset.get_dataset
       # Existing datasets...
   ```

### Template for Custom Dataset

```python
import datasets
from datasets import Dataset
import _settings

def get_dataset(tokenizer, split='validation'):
    # 1. Load your raw data
    # 2. Process into standard format
    # 3. Add prompt construction
    # 4. Tokenize for model input
    # 5. Set format and return
    
    # Example implementation:
    dataset = load_my_data()  # Your loading function
    
    def process_data(example):
        example['prompt'] = f"Your prompt format: {example['question']}"
        tokenized = tokenizer(example['prompt'], truncation=False, padding=False)
        example['input_ids'] = tokenized['input_ids']
        example['attention_mask'] = tokenized['attention_mask']
        return example
    
    dataset = dataset.map(process_data, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], 
                      output_all_columns=True)
    return dataset

def _generate_config(tokenizer):
    # Define EOS tokens and bad words
    eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', '.', '?']]
    eos_token_id += [tokenizer.eos_token_id]
    
    # Words/tokens to prevent in generation
    bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['Q:', 'Question:']]
    
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)
```

With this guide, you should be able to understand how EigenScore loads and processes different datasets, and how to integrate your own datasets into the framework. 