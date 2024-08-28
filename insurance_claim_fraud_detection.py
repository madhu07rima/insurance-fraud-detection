import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# Load the dataset
df = pd.read_csv('insurance_claims.csv')

# Handle missing values
df.ffill()

# Encode categorical variables
df = pd.get_dummies(df, columns=['policyholder_gender', 'claim_type'], drop_first=True)

# Split data into features and target
X = df.drop(columns=['claim_id', 'policyholder_id', 'is_fraud'])
y = df['is_fraud']

# Example: Histogram of claim_amount
plt.hist(df['claim_amount'], bins=20, color='skyblue')
plt.xlabel('Claim Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Claim Amount')
plt.show()
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
print(confusion_matrix(y_test, y_pred))

# ROC-AUC score(Receiver Operating Characteristic- Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_prob)
print(f'ROC-AUC Score: {roc_auc:.2f}')


# Save the model and scaler
joblib.dump(model, 'fraud_detection_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Load the model and scaler (for later use)
loaded_model = joblib.load('fraud_detection_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


def prepare_new_claim(new_claim, columns):
    new_claim_df = pd.DataFrame([new_claim])
    new_claim_df = pd.get_dummies(new_claim_df, columns=['policyholder_gender', 'claim_type'], drop_first=True)
    
    for col in columns:
        if col not in new_claim_df.columns:
            new_claim_df[col] = 0
    
    new_claim_df = new_claim_df[columns]
    return new_claim_df

new_claim = {
    'claim_amount': 15000,
    'policyholder_age': 45,
    'policyholder_gender': 'Male',
    'claim_type': 'Auto',
    'previous_claims': 2
}

# Prepare new claim data
new_claim_df = prepare_new_claim(new_claim, X.columns)

# Standardize the new data
new_claim_scaled = loaded_scaler.transform(new_claim_df)

# Predict fraud probability
fraud_prob = loaded_model.predict_proba(new_claim_scaled)[:, 1]
print(f'Fraud Probability: {fraud_prob[0]:.2f}')
