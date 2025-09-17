import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# === Load Data ===
print("Loading data...")
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# === Combine train + test for consistent encoding ===
full_df = pd.concat([train_df, test_df], axis=0)

# Drop large text columns not useful for numeric model
drop_cols = ["url", "ip_add", "content"]
full_df = full_df.drop(columns=drop_cols)
print("After dropping text columns:", full_df.shape)

# === Encode categorical features ===
label_encoders = {}
for col in full_df.select_dtypes(include=['object']).columns:
    if col != "label":  # don't encode target
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded column: {col}")

# Split back into train/test
train_processed = full_df.iloc[:len(train_df), :]
test_processed = full_df.iloc[len(train_df):, :]

X_train = train_processed.drop("label", axis=1)
y_train = train_processed["label"]

X_test = test_processed.drop("label", axis=1)
y_test = test_processed["label"]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# === Train SVM Model ===
print("\nTraining SVM model...")
svm_model = SVC(kernel="linear", C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# === Predict on Test Data ===
print("Making predictions...")
y_pred = svm_model.predict(X_test)

# === Evaluation ===
print("\n=== Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
