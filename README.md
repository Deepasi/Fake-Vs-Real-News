# Fake-Vs-Real-News
# Fake News Detection using Logistic Regression

## Overview
This project implements a **Fake News Detection** model using **Logistic Regression**. The model is trained to classify news articles as **real or fake** based on textual content.

## Dataset
The dataset used for training and testing contains:
- **Real News** articles
- **Fake News** articles
- Labels indicating whether the news is real (0) or fake (1)

### Preprocessing Steps
1. **Handling Missing Data**: Replacing null values with empty strings.
2. **Text Processing**:
   - Removing non-alphabetic characters.
   - Converting text to lowercase.
   - Removing stopwords.
   - Applying stemming using the **Porter Stemmer**.
3. **Feature Extraction**: Using **TF-IDF Vectorization** to convert textual data into numerical representation.

## Model Training
- **Algorithm Used**: Logistic Regression
- **Dataset Split**: 80% training, 20% testing
- **Accuracy Score**: **98.64%** (0.9864 on training data)

## Installation & Execution
### Prerequisites
Ensure you have the following installed:
- Python (3.x)
- Required libraries: `numpy`, `pandas`, `nltk`, `sklearn`

### Steps to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python fake_news_detection.py
   ```

## Prediction Example
The model takes a news headline and predicts whether it is **real** or **fake**.
```python
X_new = X_test[3]
prediction = model.predict(X_new)
if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')
```

## Limitations & Future Enhancements
- The model is based only on textual features, and **does not** consider metadata like source credibility.
- A deep learning model (e.g., LSTM, Transformers) can improve accuracy further.

## License
This project is open-source and available under the **MIT License**.

