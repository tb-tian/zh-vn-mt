# Chinese-Vietnamese Machine Translation Sentence Alignment

This repository contains tools for aligning Chinese-Vietnamese sentence pairs using different alignment algorithms with SBERT (Sentence-BERT) embeddings.

## Dataset preparation

Download these file and renamed it.
- JSON1: https://1drv.ms/u/c/5ee4098bb5c7cbac/IQCqzsK-ojEpRqnF7VCOoEoNAUuf4yEzuRNCE_olx_r3iuk?e=Le19rh
- JSON2: https://1drv.ms/u/c/5ee4098bb5c7cbac/IQBscKp3XN4CTKcyPHgEFfyrAVHAOT5uEh6PBRXNbkYaHdI?e=4aEzdc
- PDF1: https://1drv.ms/b/c/5ee4098bb5c7cbac/IQC9uPBwJnhxQqeiOSZtmgQIAdsz2nyOuhjk5hTBS6wzIyg?e=3lJQy0		

## Project structure
After unzipping the file, the project structure should look like this:
```
Prj_Mid_13_23120019_23120035_23120060_23120093/
├── README.md                       # Project documentation
├── ocr/                            # OCR processing scripts and tools
│   ├── ocr.ipynb                   # Notebook to convert PDF to JSON using OCR
│   ├── utils.py                    # Helper functions for image preprocessing
│   └── .env                        # Configuration for Tesseract/Poppler paths
├── align.ipynb                     # Main alignment notebook
├── json1.json                      # Downloaded and renamed dataset
├── json2.json                      # Downloaded and renamed dataset
├── pdf1.pdf                        # Downloaded and renamed dataset
└── report.pdf                      # Report
```

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for faster embedding generation)

### Install installer *Tesseract* for Windows
- [Link here to download Tesseract Installer for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
- Then download installers tesseract-ocr-w64-setup-5.5.0.20241111.exe (64 bit)

### Install installer *Tesseract* for Linux/Ubuntu
- There is a cell that instruct the user to download the necessary package
```bash
if os.name == 'posix':
    password = getpass.getpass("Enter sudo password: ")
    os.system(f'echo {password} | sudo -S apt install tesseract-ocr poppler-utils -y')
```

## Team

- Trương Bảo Thiên Ân - 23120019
- Phạm Ngọc Duy - 23120035
- Trần Kim Ngân - 23120060
- Vũ Duy Thụ - 23120093