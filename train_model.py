import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from imblearn.over_sampling import SMOTE


print("TRAINING STROKE PREDICTION MODEL....")

# Load dataset
df = pd.read_csv("brain_stroke_preprocessed.csv")

print("Dataset loaded successfully!")

# Split Features and Target
X = df.drop(columns=["stroke"])
y = df["stroke"]


# One-Hot Encoding
categorical_cols = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

print("One-hot encoding completed.")


# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("Scaler saved successfully.")


# SMOTE

smote = SMOTE(
    sampling_strategy="auto",
    random_state=42
)

X_resampled, y_resampled = smote.fit_resample(
    X_scaled,
    y
)

print("SMOTE completed.")
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())


# Base Models
knn = KNeighborsClassifier(
    n_neighbors=7
)

svm = SVC(
    kernel="linear",
    probability=True
)

cart = DecisionTreeClassifier(
    max_depth=14,
    random_state=42
)

estimators = [
    ("KNN", knn),
    ("SVM", svm),
    ("CART", cart)
]


# Stacking Model
model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

print("Training model...")

model.fit(
    X_resampled,
    y_resampled
)

print("Training completed.")


# Save model

joblib.dump(model, "stroke_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("Model trained and saved successfully!")
