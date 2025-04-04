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
   git clone https://github.com/your-username/ml-optimization-information-filtering.git
   cd ml-optimization-information-filtering
   ```

2. Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

3. Run the model
  ```bash
  python src/lda_bert_filtering.py
  ```

## Repository Structure  

```
ml-optimization-information-filtering/
│
├── data/                   # Sample datasets
├── notebooks/              # Jupyter notebooks for EDA & model evaluation
├── src/                    # Scripts for LDA + BERT pipeline
│   ├── lda_bert_filtering.py
│   ├── utils.py
│   ├── api.py              # Flask API for deployment
│
├── .gitignore              
├── LICENSE                
├── README.md               
└── requirements.txt        
```

## Requirements  

- Python >= 3.8  
- pytorch
- tenserflow  
- transformers  
- scikit-learn  
- pandas  
- numpy  
- nltk  

## License  

This project is licensed under the **MIT License**.  

## Contact  

Developed by **Haig Bedros**  
Email: haigbedros@gmail.com  
Website: [quento.ai](https://www.quento.ai)
