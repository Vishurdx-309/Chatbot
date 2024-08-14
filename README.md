# Chatbot Project

![Python](https://img.shields.io/badge/python-v3.x-blue.svg)

This project implements a chatbot using natural language processing techniques and machine learning. The chatbot processes input from an Excel file, generates responses, and saves the results back to the file.

## Features

- Text normalization and lemmatization
- Stop word removal
- TF-IDF vectorization for text representation
- Cosine similarity for finding the most relevant response
- Excel file processing for batch input/output

## Requirements

- Python 3.x
- NLTK
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- openpyxl

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/chatbot-project.git
   cd chatbot-project
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the necessary NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('wordnet')
   nltk.download('stopwords')
   nltk.download('omw-1.4')
   ```

## Usage

1. Prepare your input Excel file with an 'Input' column containing the user queries.
2. Ensure you have a `dialog_talk_agent.xlsx` file containing the chatbot's knowledge base.
3. Run the `process_excel.py` script:
   ```
   python process_excel.py
   ```
   This will process the input file (default: "Book1.xlsx") and generate responses in the 'Output' column.

## File Descriptions

- `final.py`: Contains the main chatbot logic, including text preprocessing, TF-IDF vectorization, and response generation.
- `process_excel.py`: Handles the Excel file processing, reading inputs, and writing outputs.
- `dialog_talk_agent.xlsx`: The knowledge base for the chatbot (not provided in this repository).

## How It Works

1. The chatbot reads the input from an Excel file.
2. For each input, it performs the following steps:
   - Normalizes the text (lowercase, remove special characters)
   - Tokenizes and lemmatizes the text
   - Removes stop words
   - Converts the text to TF-IDF vectors
   - Calculates cosine similarity with the knowledge base
   - Selects the most similar response
3. The generated responses are written back to the Excel file.

## Customization

You can customize the chatbot by modifying the `dialog_talk_agent.xlsx` file to include your own set of questions and responses.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- NLTK team for providing essential NLP tools
- scikit-learn developers for machine learning utilities

## Contact
Project Link: [https://github.com/Vishurdx-309/Chatbot](https://github.com/Vishurdx-309/Chatbot)
