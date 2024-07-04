import pandas as pd
import os
print(os.getcwd())

# Load the dataset
file_path = 'C:/Users/Dell/Documents/suraj/codsoft-intern/spam-sms/spamm.csv'
data = pd.read_csv(file_path, encoding='latin-1')

# Display the first few rows of the dataset
data.head()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the dataset
file_path = 'C:/Users/Dell/Documents/suraj/codsoft-intern/spam-sms/spamm.csv'
data = pd.read_csv(file_path, encoding='latin-1')

# Display the first few rows of the dataset
print(data.head())

# Data Preprocessing
# Drop unnecessary columns
data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

# Rename columns for better understanding
data.columns = ['label', 'message']

# Encode the label (spam: 1, ham: 0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Display the first few rows after preprocessing
print(data.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model Training and Evaluation
# Function to evaluate model performance
def evaluate_model(model, X_test, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Classification Report:\n", class_report)
    print("\n" + "-"*60 + "\n")

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)
evaluate_model(nb, X_test, y_test, y_pred_nb)

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_tfidf, y_train)
y_pred_log_reg = log_reg.predict(X_test_tfidf)
evaluate_model(log_reg, X_test, y_test, y_pred_log_reg)

# Support Vector Machine
svc = SVC(random_state=42)
svc.fit(X_train_tfidf, y_train)
y_pred_svc = svc.predict(X_test_tfidf)
evaluate_model(svc, X_test, y_test, y_pred_svc)

