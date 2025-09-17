import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load datasets
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# 2. Convert yes/no to 1/0
train_df["https"] = train_df["https"].map({"yes": 1, "no": 0})
test_df["https"] = test_df["https"].map({"yes": 1, "no": 0})

# 3. Select numeric features
numeric_features = ["url_len", "https", "js_len", "js_obf_len"]
X_train = train_df[numeric_features]
y_train = train_df["label"]

X_test = test_df[numeric_features]
y_test = test_df["label"]

# 4. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train SVM
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

# 6. Predict
y_pred = svm.predict(X_test)

# 7. Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🌀 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
