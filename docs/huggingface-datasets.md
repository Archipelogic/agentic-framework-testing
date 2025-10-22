# Hugging Face Datasets for Testing

## Quick Start

Install and load real data from Hugging Face:

```bash
# Install dependencies
pip install datasets

# Load all datasets
python scripts/load_huggingface_data.py --dataset all

# Or load specific dataset
python scripts/load_huggingface_data.py --dataset email
```

## Recommended Datasets by Use Case

### 1. Email Automation
- **[aeslc](https://huggingface.co/datasets/aeslc)** - Enron email subject line corpus
  - 14,436 emails with subject and body
  - Good for classification and prioritization
  
- **[enron_emails](https://huggingface.co/datasets/enron_emails)** - Full Enron email dataset
  - Real corporate emails
  - Rich metadata (sender, recipients, timestamps)

- **[spam](https://huggingface.co/datasets/spam)** - Email spam classification
  - 5,574 emails labeled as spam/ham
  - Perfect for testing filtering

### 2. GitHub Issue Triage
- **[codeparrot/github-issues](https://huggingface.co/datasets/codeparrot/github-issues)** 
  - Real GitHub issues from popular repos
  - Includes labels, comments, reactions
  
- **[bigcode/the-stack-issues](https://huggingface.co/datasets/bigcode/the-stack-issues)**
  - Massive collection of GitHub issues
  - Multiple programming languages/frameworks

### 3. Movie Recommendations
- **[movielens](https://huggingface.co/datasets/movielens)**
  - Classic recommendation dataset
  - User ratings and movie metadata
  
- **[imdb](https://huggingface.co/datasets/imdb)** 
  - 50,000 movie reviews
  - Binary sentiment classification
  
- **[rotten_tomatoes](https://huggingface.co/datasets/rotten_tomatoes)**
  - Movie reviews with ratings
  - Good for preference learning

### 4. Recipe Generation
- **[recipe_nlg](https://huggingface.co/datasets/recipe_nlg)**
  - 2.2M recipes with ingredients and instructions
  - Perfect for generation tasks
  
- **[m3hrdadfi/recipe_ingredients_and_instructions](https://huggingface.co/datasets/m3hrdadfi/recipe_ingredients_and_instructions)**
  - Detailed recipes with nutrition info
  - Multiple cuisines

### 5. Research Summary
- **[scientific_papers](https://huggingface.co/datasets/scientific_papers)**
  - ArXiv and PubMed papers
  - Abstracts and full text
  
- **[ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization)**
  - ArXiv papers with human summaries
  - Great for evaluation
  
- **[allenai/s2orc](https://huggingface.co/datasets/allenai/s2orc)**
  - Semantic Scholar Open Research Corpus
  - 81.1M academic papers

## Usage Example

```python
from datasets import load_dataset

# Load email dataset
emails = load_dataset("aeslc", split="train[:100]")

# Load GitHub issues
issues = load_dataset("codeparrot/github-issues", split="train[:100]")

# Load movie reviews
movies = load_dataset("imdb", split="train[:100]")

# Load recipes
recipes = load_dataset("recipe_nlg", split="train[:100]")

# Load research papers
papers = load_dataset("ccdv/arxiv-summarization", split="train[:50]")
```

## Data Processing Pipeline

1. **Load from Hugging Face**
   ```python
   dataset = load_dataset("dataset_name")
   ```

2. **Convert to Test Format**
   ```python
   test_cases = convert_to_test_format(dataset)
   ```

3. **Generate Ground Truth**
   ```python
   ground_truth = generate_ground_truth(dataset)
   ```

4. **Save for Testing**
   ```python
   save_test_data(test_cases, ground_truth)
   ```

## Running Tests with HF Data

```bash
# First, load the data
python scripts/load_huggingface_data.py

# Then run tests with the loaded data
python3 run_evaluation.py --live

# Or in mock mode
python3 run_evaluation.py --mock
```

## Benefits of Using Hugging Face

1. **Real Data**: Actual emails, issues, reviews, papers
2. **Diversity**: Wide variety of examples
3. **Scale**: Can load as much data as needed
4. **Preprocessing**: Data already cleaned and formatted
5. **Reproducibility**: Same datasets for all tests
6. **Free**: No API keys required for most datasets
