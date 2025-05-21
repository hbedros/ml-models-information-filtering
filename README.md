# Optimizing Machine Learning Models for Information Filtering and Truth Validation: A Hybrid LDA-BERT Approach

## Overview  

This project evaluates and optimizes machine learning models for **information filtering and truth validation** using a **hybrid approach combining Latent Dirichlet Allocation (LDA) and BERT**. The goal is to enhance qualitative data processing by reducing irrelevant information before feeding it into Large Language Models (LLMs).

## Approach  

- **LDA**: Extracts key topics and thematic structures from text.
- **BERT**: Captures contextual meaning and deep semantic understanding.
- **Hybrid Model**: Combines both methods to improve accuracy and efficiency in filtering qualitative data.  

## Applications  

- Fake news detection  
- Document classification  
- Social media monitoring  
- Internal data filtering  

## Installation  

1. Clone the repository  
   ```bash
   git clone https://github.com/haigbedros/ml-models-information-filtering-archive.git
   cd ml-models-information-filtering-archive
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model
   ```bash
   python src/process_book.py
   ```

## Repository Structure  

```
ml-models-information-filtering-archive/
│
├── notebooks/              # Jupyter notebooks for EDA & model evaluation
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── pipeline/          # Data processing pipeline
│   ├── tokenizer/         # Text tokenization utilities
│   ├── utils/             # Helper functions
│   └── process_book.py    # Main processing script
│
├── tests/                 # Test suite
├── .gitignore              
├── LICENSE                
├── README.md               
└── requirements.txt        
```

## Requirements  

- Python >= 3.8  
- torch >= 2.0.0
- sentence-transformers >= 2.2.2
- scikit-learn >= 1.6.1
- pandas >= 2.2.3
- numpy >= 2.2.4
- nltk >= 3.8.1
- PyPDF2 >= 3.0.0
- pytest >= 7.0.0 (for testing)

## Testing

Run the test suite using pytest:
```bash
pytest
```

## License  

This project is licensed under the **MIT License**.  

## Contact  

Developed by **Haig Bedros**  
Email: haigbedros@gmail.com  
Website: [quento.ai](https://www.quento.ai)