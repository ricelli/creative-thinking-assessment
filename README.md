# Creative Thinking Assessment

## Overview  
This repository houses the codebase and prompt materials developed as part of the *Creative Thinking Assessment* research project. It includes data processing pipelines, computational scripts, and evaluation prompt templates designed to analyse, classify, and interpret responses related to creative thinking in educational and experimental contexts.

## Project Structure  
- **Notebooks (.ipynb):** Implementation of various modelling approaches (e.g., BERT, DistilBERT, RoBERTa, SVM).  
- **PDFs:** Sets of prompt-templates without stimuli, segmented by dimension of PISA test (EII, GCI, GDI).  

## Getting Started  
### Prerequisites  
- Python 3.x (recommended)  
- Jupyter Notebook / Jupyter Lab  
- Key libraries: `transformers`, `scikit-learn`, `pandas`, `numpy` 

## Usage
Open the notebook corresponding to your modelling choice (e.g., DistilBERT.ipynb).
Run the data preparation section: load datasets, apply pre-processing, feature extraction.
Execute model training, validation and evaluation.
Explore the prompt-sets (PDFs) to review how the assessment prompts were structured and used.

## Research Context
This project addresses the assessment of creative thinking by combining prompt-design (to evoke creative responses) and computational modelling (to classify and interpret those responses). The aim is to support both experimental investigations and educational applications in measuring creativity within textual responses.

## Contributing
Contributions are welcome. If you wish to add new models, extend prompt sets, include further data or suggest improvements, please:
Fork the repository
Create a new branch for your feature
Submit a Pull Request with clear description of your changes
Ensure notebooks and scripts have appropriate docstrings/comments and pass any existing checks
