import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# === Quantum imports ===
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVM

# === Load Data ===
print("Loading data...")
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")
print("Original Train shape:", train_df.shape)
print("Original Test shape:", test_df.shape)

# === Reduce dataset size for faster quantum runs ===
train_df = train_df.sample(n=500, random_state=42)
test_df = test_df.sample(n=200, random_state=42)

# === Drop unused columns ===
drop_cols = ["url", "ip_add", "content"]
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

# === Encode categorical features ===
le_dict = {}
for col in train_df.select_dtypes(include=['object']).columns:
    if col != "label":
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        le_dict[col] = le

# === Convert label to numeric if not already ===
if train_df['label'].dtype == object:
    le_label = LabelEncoder()
    train_df['label'] = le_label.fit_transform(train_df['label'])
    test_df['label'] = le_label.transform(test_df['label'])

# === Split features and labels ===
X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]
X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

# === Scale features for QSVM ===
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ======================================================
# 1️⃣ Classical SVM
# ======================================================
print("\n=== Training Classical SVM ===")
svm_model = SVC(kernel="linear", C=1.0, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_classical = svm_model.predict(X_test)

print("\n--- Classical SVM Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_classical))
print("Report:\n", classification_report(y_test, y_pred_classical))

# ======================================================
# 2️⃣ Quantum SVM (QSVM)
# ======================================================
print("\n=== Training QSVM (Quantum SVM) ===")

# Quantum feature map
num_features = X_train_scaled.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)

# Quantum backend using Sampler
sampler = Sampler(backend=AerSimulator())


# Quantum kernel
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=sampler)

# QSVM
qsvm = QSVM(quantum_kernel)
qsvm_results = qsvm.run(X_train_scaled, y_train, X_test_scaled, y_test)

print("\n--- QSVM Results ---")
print("Accuracy:", qsvm_results["testing_accuracy"])
