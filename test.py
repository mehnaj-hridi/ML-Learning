# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import pandas as pd
import numpy as np
import time
import os
import sklearn
import seaborn as sns
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from mpl_toolkits.mplot3d import Axes3D
from random import randrange

# Disabling Warnings
warnings.filterwarnings('ignore')

# To make this script's output stable across runs
np.random.seed(42)

# Matplotlib settings
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
sns.set_palette(['green', 'red'])  # Fixing the Seaborn default palette

# Load dataset
def loadDataset(file_name):
    df = pd.read_csv(file_name)
    return df

if __name__ == "__main__":
    # Load training and test sets
    df_train = loadDataset("train_data.csv")
    df_test = loadDataset("test_data.csv")

    # Ensuring correct sequence of columns
    df_train = df_train[['url', 'ip_add', 'geo_loc', 'url_len', 'js_len', 'js_obf_len',
                         'tld', 'who_is', 'https', 'content', 'label']]
    df_test = df_test[['url', 'ip_add', 'geo_loc', 'url_len', 'js_len', 'js_obf_len',
                        'tld', 'who_is', 'https', 'content', 'label']]

    print("Training dataset shape:", df_train.shape)
    print("Test dataset shape:", df_test.shape)

    # Class Distribution
    print("\nClass distribution in training data:")
    print(df_train.groupby('label').size())

    pos, neg = df_train['label'].value_counts()
    total = neg + pos
    print('\nDataset Summary:')
    print('Total Samples: %s' % total)
    print('Positive: {} ({:.2f}% of total)'.format(pos, 100 * pos / total))
    print('Negative: {} ({:.2f}% of total)'.format(neg, 100 * neg / total))

    # =======================
    # Visualization
    # =======================
    fig = plt.figure(figsize=(12, 4))
    fig.subplots_adjust(top=0.85, wspace=0.3)

    # Bar Plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel("Class Labels")
    ax1.set_ylabel("Frequency")
    ax1.title.set_text('Bar Plot: Malicious & Benign Webpages')
    labels = df_train['label'].value_counts()
    bar = ax1.bar(list(labels.index), list(labels.values), color=['green', 'red'], edgecolor='black', linewidth=1)

    # Stack Plot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.title.set_text('Stack Plot: Malicious & Benign Webpages')
    df_train.assign(dummy=1).groupby(['dummy', 'label']).size().groupby(level=0).apply(
        lambda x: 100 * x / x.sum()).to_frame().unstack().plot(
            kind='bar', stacked=True, legend=False, ax=ax2,
            color={'red', 'green'}, linewidth=0.50, edgecolor='k'
        )
    ax2.set_xlabel('Benign/Malicious Webpages')
    ax2.set_xticks([])  # disable ticks in the x axis
    current_handles, _ = plt.gca().get_legend_handles_labels()
    reversed_handles = reversed(current_handles)
    correct_labels = reversed(['Malicious', 'Benign'])
    plt.legend(reversed_handles, correct_labels)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    # Save plots
    os.makedirs("imgs", exist_ok=True)
    plt.tight_layout()
    plt.savefig("imgs/Class_Distribution.png")
    plt.show()

    # =======================
    # ML Part: SVM Classifier
    # =======================
    print("\n===== MACHINE LEARNING WITH SVM =====")

    # Drop 'url' (not useful for ML directly)
    X_train = df_train.drop(columns=['url', 'label'])
    y_train = df_train['label']
    X_test = df_test.drop(columns=['url', 'label'])
    y_test = df_test['label']

    # Encode categorical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        le_dict[col] = le

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    clf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = clf.predict(X_test_scaled)

    # Evaluation
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label='malicious'))
    print("Recall:", recall_score(y_test, y_pred, pos_label='malicious'))
    print("F1 Score:", f1_score(y_test, y_pred, pos_label='malicious'))

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=['benign', 'malicious'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (SVM)")
    plt.savefig("imgs/SVM_Confusion_Matrix.png")
    plt.show()
