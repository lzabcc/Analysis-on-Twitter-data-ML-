
Project Overview

This project is dedicated to performing sentiment analysis on Twitter data, primarily aiming to categorize tweets into three classes: Positive, Negative, or Neutral. The project encompasses several key stages: data preprocessing, feature engineering, model selection, evaluation, and addressing class imbalance issues.

Process of Completion

To achieve the aforementioned objectives, the project adhered to the following steps:

Data Preprocessing:
Extensive cleaning of raw data, including the removal of noise, abbreviations, spelling errors, and special characters.
Converting text to lowercase.
Removing Twitter usernames prefixed with '@'.
Stripping away hashtags marked with '#'.
Eliminating 'RT' markers indicating retweets.
Removing URLs.
Normalizing whitespace.
Retaining only alphabetical characters.
Removing deactivated words.
Stemming and lemmatizing words using Porter Stemmer and WordNet Lemmatizer.
Correcting spelling errors using TextBlob.

Feature Engineering:
Utilizing TF-IDF to transform textual data into numerical features that can be understood by computers.

Model Training:
Employing Logistic Regression and Multinomial Naive Bayes classifiers.
Computing category weights to adjust for the imbalance between positive, negative, and neutral sentiments.
Implementing Multilayer Perceptron (MLP) and achieving a significant improvement in F1-score after hyperparameter tuning.

Evaluation:
Evaluating the models using accuracy and F1 scores.
Analyzing the effectiveness of the models and making adjustments based on the evaluation metrics.

Key Technologies Used
Throughout the analysis process, three key technologies were employed:

Feature Engineering: Through TF-IDF, textual data was converted into numerical features, making the data usable for machine learning models. TF-IDF not only emphasized the importance of terms but also minimized the impact of common words.
Model Selection and Evaluation: Various machine learning algorithms were experimented with, including Logistic Regression, Multinomial Naive Bayes, and Support Vector Machines. Additionally, Multi-layer Perceptron (MLP) was utilized in deep learning, and after hyperparameter optimization, the predictive performance was improved.
Handling Class Imbalance: To tackle the issue of class imbalance where the quantities of positive, negative, and neutral samples varied significantly, category weights were adjusted to improve the model's performance. This ensured that even classes with fewer samples were treated fairly.

Technology Stack
Python: The primary programming language used for data manipulation and machine learning.
Pandas: For data manipulation and analysis.
NLTK: For natural language processing tasks.
Scikit-Learn: For machine learning models and evaluation metrics.
TensorFlow/Keras: For deep learning models.
Matplotlib: For data visualization.

![image](https://github.com/user-attachments/assets/327e0190-ebd8-460b-802d-8d494793857c)

The 'clean_dev', 'clean_train', and 'clean_test' files are cleaned from the original data using code. In the preprocessing part, you will notice commented-out code marked with 'first use'. This code is utilized to write to new CSV files, just the ones mentioned before.
