# Chinese-Vietnamese Machine Translation Sentence Alignment

This repository contains tools for aligning Chinese-Vietnamese sentence pairs using different alignment algorithms with SBERT (Sentence-BERT) embeddings.

## Repository Structure

```
zh-vn-mt/
├── README.md
├── vecalign_sbert.py              # An approach using Vecalign + LaBSE
├── vecalign_sbert_notebook.ipynb  # Another approach using Vecalign in Notebook
├── bertalign_sbert_notebook.ipynb # An approach using Bertalign in Notebook
├── vecalign/                      # Vecalign library (cloned from GitHub)
└── bertalign/                     # Bertalign library (cloned from GitHub)
```

## Features

- **LaBSE embeddings**: Uses Language-agnostic BERT Sentence Embedding for cross-lingual sentence representations
- **Multiple alignment methods**: 
  - Vecalign: Supports many-to-many alignments (1:1, 1:2, 2:1, 2:2)
  - Bertalign: Alternative alignment algorithm
- **Sentence segmentation**: Automatic segmentation for both Chinese and Vietnamese text

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for faster embedding generation)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/tb-tian/zh-vn-mt.git
   cd zh-vn-mt
   ```

2. Install dependencies:
   ```bash
   pip install sentence-transformers pysbd tqdm numpy cython
   ```

3. The alignment libraries (`vecalign` and `bertalign`) will be automatically cloned when you run the notebooks.

## Usage

### Using the Notebooks (Recommended)

1. **Vecalign + SBERT**: Open `vecalign_sbert_notebook.ipynb`
   - Follows the original Vecalign algorithm
   - Best for many-to-many sentence alignments

2. **Bertalign + SBERT**: Open `bertalign_sbert_notebook.ipynb`
   - Uses the Bertalign algorithm
   - Alternative approach for sentence alignment

### Using the Python Script

```python
from vecalign_sbert import VecalignSBERT

# Initialize the aligner
aligner = VecalignSBERT(similarity_threshold=0.5)

# Align your sentences
# ... (see script for detailed usage)
```

## Output Format

Aligned pairs are saved in JSON format with the following structure:

```json
{
  "zh": "Chinese sentence",
  "vi": "Vietnamese sentence", 
  "similarity": 0.85,
  "alignment_type": "1-1",
  "source": "source_file",
  "source_id": 0
}
```