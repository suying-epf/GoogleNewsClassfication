# ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## Project NLP Google news category analysis <br>YING SU
<img src='images.jpeg' width=700>

## Global Description
This dataset contains metadata of millions of news articles from Google News, including title, publisher, DateTime, link, and category.<br>
<br>
This is also an automation project in which data is scraped every day at 4am UTC on 8 major categories. This dataset is expected to have a monthly update, thus the data collected daily will be merged into a single monthly csv file and published on Kaggle at the end of each month. One may expect the value of the dataset to continuously grow through time.<br>
<br>

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
