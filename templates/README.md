# Text Classification and Similarity 

## Overview
The **Text Classification** is a machine learning-based web application that classifies text into predefined categories.  This project implements a Text Classification model using TF-IDF (Term Frequency-Inverse Document Frequency) and Naive Bayes to classify documents into predefined categories. Additionally, it integrates Cosine Similarity to analyze how similar a query is to the training documents. This solution provides an interactive user interface where users can input text and get predictions, along with the calculated similarity scores.

This project demonstrates how to use supervised learning for classifying text data, leveraging models like Naive Bayes, and combining it with cosine similarity to rank similar documents.

## Features
- **Text Classification**: Classify text into categories such as drones, agriculture, sports, and AI.
- **Cosine Similarity**: Rank documents based on similarity to the input text.
- **Training and Testing**: Train a Naive Bayes classifier on provided documents and evaluate its performance on test data.
- **TF-IDF Vectorization**: Converts text data into numerical vectors for further analysis.
- **User Interface**: A simple web interface allows users to input a query and view results including the predicted category and similarity scores.
- **Model Evaluation**: The model provides training accuracy and tests the accuracy on unseen data.

## Technologies Used
- **Python 3.x**
- **Flask**: Web framework for building the app interface.
- **scikit-learn**: For machine learning models and vectorization.
- **Cosine Similarity**: To measure similarity between input and training data.
- **HTML/CSS**: For front-end design.

## How It Works
1. **Data Preprocessing and Vectorization**:
The training data is processed using TF-IDF vectorization. This method converts the text data into vectors of numbers, considering the importance of words in relation to the entire document corpus.

2. **Cosine Similarity**:
When a user inputs a query, Cosine Similarity is used to find the documents most similar to the query by comparing the query's vector with the vectors of the documents in the training set.

3. **Naive Bayes Classifier**:
Once the similarity is computed, the query is also classified using a Multinomial Naive Bayes classifier, which is trained on the same document vectors. The classifier predicts the most likely category of the input text.

4. **Result Display**:
The interface shows the most similar documents, the predicted category, and the similarity score between the input and the top matching document.

5. **Accuracy Calculation**:
The accuracy of the prediction is calculated based on the similarity between the predicted category and the most similar document. If the predicted category matches the most similar document's category, the prediction is considered correct.

## User Interface (UI)
The user interface (UI) is designed to be simple and interactive, providing users with an easy way to input their queries and view the results.

**Key UI Features**:
**Input Box**:
Users can type or paste a text query into the input box. This can be any text related to one of the predefined categories (e.g., "Agriculture," "Sports," etc.).
**Classify Button**:
Once the user enters their query, they click the "Classify" button. This triggers the model to process the input text and return the results.
Predicted Category:

The model predicts the category of the input text. This is displayed prominently on the page for the user to see.
Top Similar Documents:

The UI displays the documents from the training set that are most similar to the input query. Each document is accompanied by a similarity score (a percentage), helping users understand how close the query is to the documents in the training set.
Similarity Score:

The top-ranked document will also display the Cosine Similarity score, indicating how closely it matches the query. This score is used to measure the degree of similarity.
Accuracy Display:

The UI also includes the accuracy of the classification, calculated based on the similarity score between the predicted category and the most similar document. This is an additional measure of how well the model performed.
Responsive Design:

```
/text-classification
│
├── app.py                
├── text_classification.py 
├── templates/             
│   ├── index.html        
│
├── static/              
│   ├── style.css         
│
└── requirements.txt     


```