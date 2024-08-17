Introduction:

Social media, particularly Twitter, is crucial for understanding public sentiment and guiding business strategies. This project leverages Natural Language Processing (NLP) to analyze Twitter data, aiming to uncover customer sentiments and enhance business decision-making (Bird, Klein, & Loper, 2009; Jurafsky & Martin, 2021).

Business Problem:

Companies face the challenge of interpreting vast amounts of Twitter data to understand customer sentiment and address emerging issues. This project focuses on analyzing tweets to:

Identify key sentiments (positive, negative, neutral)
Detect recurring issues
Assess product reception (Potts, 2011)
Dataset Overview:

The dataset from Kaggle includes 74,681 tweets with attributes such as Tweet ID, Product, Sentiment, and Text. This data will be analyzed to derive insights into customer perceptions.

Methodology:

Text Preprocessing: Includes cleaning text data, removing irrelevant content, and standardizing text.
Text Representation: Utilizes techniques like Bag of Words (BoW) and TF-IDF for feature extraction.
Model Building: Naïve Bayes models are used with BoW and TF-IDF for sentiment classification.
Data Splitting: The dataset is divided into training and testing sets to evaluate model performance.
Results:

Bag of Words Model: Achieved 71.93% accuracy.
TF-IDF Model: Achieved 69.53% accuracy.
The BoW model slightly outperformed the TF-IDF model.
