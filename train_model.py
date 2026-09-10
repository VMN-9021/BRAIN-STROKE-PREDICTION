import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("dataset/brain.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =====================================================
# 2. REMOVE ID
# =====================================================

if "id" in df.columns:
    df = df.drop("id", axis=1)


# =====================================================
# 3. HANDLE MISSING VALUES
# =====================================================

# BMI may contain missing values
if "bmi" in df.columns:
    df["bmi"] = pd.to_numeric(
        df["bmi"],
        errors="coerce"
    )

    df["bmi"] = df["bmi"].fillna(
        df["bmi"].median()
    )


# =====================================================
# 4. ENCODE CATEGORICAL COLUMNS
# =====================================================

categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)


encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

    encoders[column] = encoder


# =====================================================
# 5. SEPARATE FEATURES AND TARGET
# =====================================================

X = df.drop("stroke", axis=1)
y = df["stroke"]


print("\nTarget distribution:")
print(y.value_counts())


# =====================================================
# 6. HANDLE ANY REMAINING MISSING VALUES
# =====================================================

imputer = SimpleImputer(
    strategy="median"
)

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)


# =====================================================
# 7. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =====================================================
# 8. SCALE FEATURES
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# =====================================================
# 9. SMOTE
# =====================================================

smote = SMOTE(
    random_state=42
)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled,
    y_train
)


print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_balanced).value_counts())


# =====================================================
# 10. MODELS
# =====================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=200,
            random_state=42
        ),

    "Gaussian Naive Bayes":
        GaussianNB(),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            max_depth=8
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}


# =====================================================
# 11. TRAIN MODELS
# =====================================================

results = {}

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(
        X_train_balanced,
        y_train_balanced
    )

    y_pred = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    results[name] = {
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100
    }

    trained_models[name] = model


# =====================================================
# 12. COMPARISON
# =====================================================

results_df = pd.DataFrame(results).T

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.round(2))


# =====================================================
# 13. BEST MODEL
# =====================================================

best_model_name = results_df.index[0]

best_model = trained_models[
    best_model_name
]

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Model:",
    best_model_name
)

print(
    "Accuracy:",
    round(
        results_df.iloc[0]["Accuracy"],
        2
    ),
    "%"
)


# =====================================================
# 14. SAVE EVERYTHING
# =====================================================

joblib.dump(
    best_model,
    "stroke_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    imputer,
    "imputer.pkl"
)

joblib.dump(
    encoders,
    "encoders.pkl"
)

results_df.to_csv(
    "model_comparison.csv"
)


print("\nFiles saved:")
print("stroke_model.pkl")
print("scaler.pkl")
print("imputer.pkl")
print("encoders.pkl")
print("model_comparison.csv")
# -----------------------------------------
# Model Comparison Graph
# -----------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df.index,
    results_df["Accuracy"]
)

plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")

plt.xticks(rotation=30)
plt.ylim(0, 100)

# Display accuracy values on top of bars
for i, value in enumerate(results_df["Accuracy"]):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

# Save graph
plt.savefig("model_comparison.png", dpi=300)

# Show graph
plt.show()