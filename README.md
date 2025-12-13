# Chinese-Vietnamese Machine Translation Sentence Alignment

This repository contains tools for aligning Chinese-Vietnamese sentence pairs using different alignment algorithms with SBERT (Sentence-BERT) embeddings.

## Dataset preparation

Download these file and renamed it.
- JSON1: https://1drv.ms/u/c/5ee4098bb5c7cbac/IQCqzsK-ojEpRqnF7VCOoEoNAUuf4yEzuRNCE_olx_r3iuk?e=Le19rh
- JSON2: https://1drv.ms/u/c/5ee4098bb5c7cbac/IQBscKp3XN4CTKcyPHgEFfyrAVHAOT5uEh6PBRXNbkYaHdI?e=4aEzdc
- PDF1: https://1drv.ms/b/c/5ee4098bb5c7cbac/IQC9uPBwJnhxQqeiOSZtmgQIAdsz2nyOuhjk5hTBS6wzIyg?e=3lJQy0		

## Project structure

```
zh-vn-mt/
├── README.md                       # Project documentation
├── ocr/                            # OCR processing scripts and tools
├── align.ipynb                     # Main alignment notebook
├── json1.json                      # Downloaded and renamed dataset
├── json2.json                      # Downloaded and renamed dataset
└── pdf1.pdf                        # Downloaded and renamed dataset
```

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for faster embedding generation)

## Team

- Trương Bảo Thiên Ân - 23120019
- Phạm Ngọc Duy - 23120035
- Trần Kim Ngân - 23120060
- Vũ Duy Thụ - 23120093