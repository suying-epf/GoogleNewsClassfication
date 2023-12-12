# ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## Project NLP Google news category analysis <br>YING SU
<img src='images.jpeg' width=700>

## Global Description
This dataset contains metadata of millions of news articles from Google News, including title, publisher, DateTime, link, and category.<br>
<br>
This is also an automation project in which data is scraped every day at 4am UTC on 8 major categories. This dataset is expected to have a monthly update, thus the data collected daily will be merged into a single monthly csv file and published on Kaggle at the end of each month. One may expect the value of the dataset to continuously grow through time.<br>
<br>

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Machine Learning](#machine-learning)
- [Deep Learning](#deep-learning)
- [Final Results](#final-results)
- [Conclusion](#conclusion)
- [Author](#author)

## Dataset
- Dataset Name: [Link to Dataset](https://www.kaggle.com/datasets/crxxom/daily-google-news)
1. Title:<br>
The title of the news article
2. Publisher:<br>
The publisher of the news article

3. DateTime:<br>
The DateTime of when the news article is published on Google News

4. Link:<br>
A link that will direct users to the corresponding article, one may feel free to dig deeper and scrape extended content by following the links

5. Category:<br>

8 major categories defined by Google News, particularly Business, Entertainment, Headlines, Health, Science, Sports, Technology and WorldWide.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/suying-epf/GoogleNewsClassfication.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd NLP_analysis
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
## Usage

**Evironment**:
Execute the following cells at the beginning of the notebook to ensure the correct directory structure is linked: (for mac)

```python
import sys
sys.path.append('../../googlenewsNLP/Scripts')
```
## Code Structure

This section provides an overview of the project's code structure and organization.
```plaintext
NLP_analysis/
├── data/
│   ├── clean_data.csv
│   ├── 2023_9.csv
│   
├── Notebooks/
│   ├── Baseline_model.ipynb
│   ├── Deep_learning.ipynb
│   ├── Exploratory_data.ipynb
│   ├── Improve_baseline.ipynb
│   └── best_model.h5
├── Scripts/
│   ├── Preprocesing.py
│   ├── Preprocessing.ipynb
│   ├── textclassfier.py
│   └── utils.py
├── README.md
└── requirements.txt
```

### data
1. `complete_cleaned_spellings_Restaurant_reviews.csv`
   - Preprocessed data
2. `Restaurant reviews.csv`:
   - Raw data
### Scripts
1. `preprocesing.py`&`preprocessing.ipynb`
   - The preprocessing pipeline
2. `textclassfier.py`:
   - The `TextClassifier` class

### Notebooks
1. `Exploratory_data.ipynb`:
   - Data exploration
2. `Baseline_model.ipynb`:
   - Creation and evaluation of the baseline model
3. `Improve_baseline.ipynb`:
   - Improve the baseline model
4. `Deep_Learning.ipynb`:
   - Use deep learning for better results


## Results
## Model Performances
| Model          | Accuracy (%) | F1 Score (%) | Number of training epochs
|----------------|----------|----------|-----|
| 1)Count Vectors + MultinomialNB | 0.87     | 0.867     | x|
| 2)TF-IDF + MultinomialNB | 0.87     | 0.871     | x |
| 3)TF-IDF2 + MultinomialNB| 0.87      | 0.870      | x |
| 4)TF-IDF + RidgeClassifier| 0.88     | 0.884     | x |
| 5)Word2Vec and TensorFlow| 0.83      | 0.829      | 20 |
| 6)Retrain model with callbacks| 0.83      | 0.829      | 20 |


**Accessing cleaned data**:
The cleaned data is available in the `data/clean_data` file. It is not recommended to run the preprocessor again because the Spellings run for a long time.

## General conclusions
This project is essentially a text classification problem that requires classification based on the characters in each sentence.

Since text data is a typical unstructured data, it may involve both feature extraction and classification model components.

Idea 1: TF-IDF + Machine Learning Classifier
Directly use TF-IDF to extract features from the text and use a classifier to classify it. For the choice of classifier, you can use SVM, LR, or XGBoost.

Idea 2: FastText
FastText is the starter word vector, and a classifier can be built quickly using the FastText tool provided by Facebook.

Idea 3: WordVec + Deep Learning Classifier
WordVec is an advanced word vector, and the classification is done by building a deep learning classification. The network structure for deep learning classification can choose TextCNN, TextRNN or BiLSTM.

Idea 4: Bert word vector
Bert is a high end word vector with powerful modelling learning capabilities.
### *Dataset Credits :*

https://blog.csdn.net/weixin_42691585/article/details/107981604

## Author
- YING SU(ying.su@epfedu.fr)