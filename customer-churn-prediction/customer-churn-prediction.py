import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the dataset
file_path = r'C:\Users\Dell\Documents\suraj\codsoft-intern\customer-churn-prediction\Churn_Modelling.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
# 1. Check for missing values (there are none in this dataset)
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# 2. Encode categorical variables (Geography and Gender)
data_encoded = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

# 3. Define features and target variable
X = data_encoded.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = data_encoded['Exited']

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
evaluate_model(log_reg, X_test, y_test, y_pred_log_reg)

# Random Forest
rand_forest = RandomForestClassifier(random_state=42)
rand_forest.fit(X_train_scaled, y_train)
y_pred_rand_forest = rand_forest.predict(X_test_scaled)
evaluate_model(rand_forest, X_test, y_test, y_pred_rand_forest)

# Gradient Boosting
grad_boost = GradientBoostingClassifier(random_state=42)
grad_boost.fit(X_train_scaled, y_train)
y_pred_grad_boost = grad_boost.predict(X_test_scaled)
evaluate_model(grad_boost, X_test, y_test, y_pred_grad_boost)
