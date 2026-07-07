# Generated from: Kodingan_Stroke_.ipynb
# Converted at: 2026-07-06T18:51:06.316Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn .ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
import plotly.express as px
from sklearn .ensemble import VotingClassifier , BaggingClassifier , StackingClassifier


from google.colab import files
uploaded=files.upload()
filename=list(uploaded.keys())[0]
df=pd.read_csv('brain_stroke.csv')

print("Nama file\t:",filename)
print("Nama kolom\t:",df.columns.tolist())
print("Banyak data\t:",df.shape)

df.head()

df.describe()

df.isnull().values.any()

df.isnull().sum().sum()

# Mengganti delimiter saat membaca CSV
df = pd.read_csv('brain_stroke.csv', delimiter=';')

# Daftar kolom kategorikal yang akan dikodekan
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
x = df.drop(columns=['stroke'])
y = df['stroke']
x_encoded = pd.get_dummies(x, columns=categorical_cols, drop_first=True)
scaler = StandardScaler()
scaler.fit(x_encoded)
standar = scaler.transform(x_encoded)

x=standar
y=df['stroke']

# ### Mengatasi Data Hilang (Missing Data)


missing_data = df.isnull().sum()

print("Jumlah data hilang di setiap kolom:")
print(missing_data)

if missing_data.any():
    print("\nAda data hilang!")
else:
    print("\nTidak ada data hilang!")

# 1. Menghapus Baris yang Mengandung Missing Data


df_cleaned = df.dropna()
df_cleaned

# ### Mengatasi Data Pencilan (Outlier)


plt.figure(figsize = (12, 6))
sns.boxplot(data=df_cleaned)  #sesauikan denganvar yg df.cleaned
plt.xticks(rotation=72)
plt.title("Boxplot untuk Setiap Kolom")  #judul gambar
plt.show() #menampilkan gambar

# 1. Menghapus Pencilan (Outlier) di Kolom Tertentu



df_cleaned_no_id = df_cleaned.drop(columns=['id'], errors='ignore')

df_no_outlier_avg = df_cleaned_no_id.copy()
Q1 = df_no_outlier_avg['avg_glucose_level'].quantile(0.33)
Q3 = df_no_outlier_avg['avg_glucose_level'].quantile(0.67)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

df_no_outlier_insulin = df_no_outlier_avg[(df_no_outlier_avg['avg_glucose_level'] >= lower_bound) & (df_no_outlier_avg['avg_glucose_level'] <= upper_bound)]  #inisialisasi

print(f"Jumlah data setelah menghapus outlier pada kolom avg_glucose_level: {df_no_outlier_insulin.shape[0]}")

plt.figure(figsize = (12, 6)) #ukuran
sns.boxplot(data = df_no_outlier_insulin) #disimpan di df_no
plt.xticks(rotation = 30)
plt.title("Boxplot untuk Setiap Kolom")
plt.show()

df_no_outlier_bmi = df_no_outlier_insulin.copy()
Q1 = df_no_outlier_bmi['bmi'].quantile(0.33)
Q3 = df_no_outlier_bmi['bmi'].quantile(0.67)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

df_no_outlier_bimo = df_no_outlier_bmi[(df_no_outlier_bmi['bmi'] >= lower_bound) & (df_no_outlier_bmi['bmi'] <= upper_bound)]  #inisialisasi

print(f"Jumlah data setelah menghapus outlier pada kolom bmi: {df_no_outlier_bimo.shape[0]}")

plt.figure(figsize = (12, 6)) #ukuran
sns.boxplot(data = df_no_outlier_bimo) #disimpan di df_no
plt.xticks(rotation = 30)
plt.title("Boxplot untuk Setiap Kolom")
plt.show()

# Mengisi Ulang Variabel x dan y


# Mengisi Ulang Variabel x dan y
# Menggunakan df_no_outlier_insulin setelah menghapus outlier pada kolom 'id'
x_no_outlier = df_no_outlier_bimo.drop(columns = ['stroke'])
y_no_outlier = df_no_outlier_bimo['stroke']
df_no_outlier_bimo.describe()

# ### Standarisasi Data


# 1. Standard Scaling


from sklearn.preprocessing import StandardScaler
import pandas as pd
#categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
x_no_outlier_encoded = pd.get_dummies(x_no_outlier, columns=categorical_cols, drop_first=True)
scaler = StandardScaler()
standard_scaled_data = scaler.fit_transform(x_no_outlier_encoded)
df_standard_scaled = pd.DataFrame(standard_scaled_data, columns=x_no_outlier_encoded.columns)
df_standard_scaled["stroke"] = y_no_outlier.reset_index(drop=True).values
df_standard_scaled

plt.figure(figsize = (12, 6))
sns.boxplot(data = df_standard_scaled)
plt.xticks(rotation = 30)
plt.title("Boxplot untuk Standard Scaler")
plt.show()
df_standard_scaled.describe()

# 2. MaxAbs Scaling


from sklearn.preprocessing import MaxAbsScaler
x_no_outlier_encoded = pd.get_dummies(x_no_outlier, columns=['gender','ever_married','work_type','Residence_type','smoking_status'])
scaler = MaxAbsScaler()
maxabs_scaled_data = scaler.fit_transform(x_no_outlier_encoded)
df_maxabs_scaled = pd.DataFrame(maxabs_scaled_data, columns=x_no_outlier_encoded.columns)
df_maxabs_scaled["stroke"] = y_no_outlier.values
df_maxabs_scaled

plt.figure(figsize = (12, 6))
sns.boxplot(data = df_maxabs_scaled)
plt.xticks(rotation = 70)
plt.title("Boxplot untuk MaxAbs Scaler")
plt.show()
df_maxabs_scaled.describe()

# Mengisi Ulang Variabel x dan y
x_standard_scaled = df_standard_scaled.drop(columns = ['stroke'])
y_standard_scaled = df_standard_scaled['stroke']

# ### Keseimbangan Data


print("Jumlah label:")
print(df_standard_scaled['stroke'].value_counts())

df_standard_scaled['stroke'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.xlabel('stroke')
plt.ylabel('Count')
plt.title('Distribusi stroke')
plt.xticks(rotation=0)
plt.show()

# 1. SMOTE: Synthetic Minority Over-sampling Technique


from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy = 'auto', random_state = 42)
x_oversampling, y_oversampling = smote.fit_resample(x_standard_scaled, y_standard_scaled)

print("Distribusi label setelah oversampling:")
print(pd.Series(y_oversampling).value_counts())

y_oversampling.value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.xlabel('stroke')
plt.ylabel('Count')
plt.title('Distribusi stroke')
plt.xticks(rotation=0)
plt.show()

# Mengisi Ulang Variabel x dan y
x = x_oversampling
y = y_oversampling

# ### **PERCANTAGE SPLIT**


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# ALGORTIMA SVM
SVM = svm.SVC(kernel='linear', probability=True)
SVM.fit(x_train,  y_train)
predictionSVM = SVM.predict(x_train)
trainingSVM = accuracy_score(predictionSVM ,y_train)
y_testSVM = SVM.predict(x_test)
test_accuracySVM = accuracy_score(y_test,y_testSVM)
accuracySVM = accuracy_score(y_test, y_testSVM) * 100
precisionSVM = precision_score(y_test, y_testSVM) * 100
recallSVM = recall_score(y_test, y_testSVM) * 100
f1_scoreSVM = f1_score(y_test, y_testSVM) * 100


KNN = KNeighborsClassifier(n_neighbors = 7)
KNN.fit(x_train,y_train)
predictionKNN = KNN.predict(x_train)
trainingKNN = accuracy_score(predictionKNN ,y_train)
y_testKNN = KNN.predict(x_test)
test_accuracyKNN = accuracy_score(y_test,y_testKNN)
accuracyKNN = accuracy_score(y_test, y_testKNN) * 100
precisionKNN = precision_score(y_test, y_testKNN) * 100
recallKNN = recall_score(y_test, y_testKNN) * 100


# ALGORTIMA CART Classifier
CART_C = DecisionTreeClassifier(max_depth= 14)
CART_C.fit(x_train,  y_train)
predictionCART_C = CART_C.predict(x_train)
trainingCART_C = accuracy_score(predictionCART_C ,y_train)
y_testCART_C = CART_C.predict(x_test)
test_accuracyCART_C = accuracy_score(y_test,y_testCART_C)
accuracyCART_C = accuracy_score(y_test, y_testCART_C) * 100
precisionCART_C = precision_score(y_test, y_testCART_C) * 100
recallCART_C = recall_score(y_test, y_testCART_C) * 100

# Model Ensemble VotingClassifier (Soft Voting)
ensemble_soft = VotingClassifier(estimators=[('SVM', SVM), ('KNN', KNN), ('CART_C', CART_C)], voting='soft')
ensemble_soft.fit(x_train, y_train)
prediction_ensemble_soft = ensemble_soft.predict(x_train)
training_ensemble_soft = accuracy_score(prediction_ensemble_soft, y_train)
y_test_ensemble_soft = ensemble_soft.predict(x_test)
test_accuracy_ensemble_soft = accuracy_score(y_test, y_test_ensemble_soft)
accuracy_ensemble_soft = accuracy_score(y_test, y_test_ensemble_soft) * 100
precision_ensemble_soft = precision_score(y_test, y_test_ensemble_soft) * 100
recall_ensemble_soft = recall_score(y_test, y_test_ensemble_soft) * 100
f1_score_ensemble_soft = f1_score(y_test, y_test_ensemble_soft) * 100

conf_matrixSVM = confusion_matrix(y_test, y_testSVM)
print("Confusion Matrix SVM:")
print(conf_matrixSVM)

conf_matrixKNN = confusion_matrix(y_test, y_testKNN)
print("Confusion Matrix KNN:")
print(conf_matrixKNN)

conf_matrixCART_C = confusion_matrix(y_test, y_testCART_C)
print("Confusion Matrix CART_C:")
print(conf_matrixCART_C)

conf_matrixensemble_soft  = confusion_matrix(y_test, y_test_ensemble_soft)
print("Confusion Matrix ENSEMBLE:")
print(conf_matrixensemble_soft)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.heatmap(conf_matrixSVM, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix - SVM (Percentage Split)')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_ylabel('True Label')

sns.heatmap(conf_matrixKNN, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix - KNN (Percentage Split)')
axes[0, 1].set_xlabel('Predicted Label')
axes[0, 1].set_ylabel('True Label')

sns.heatmap(conf_matrixCART_C, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix - CART (Percentage Split)')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_ylabel('True Label')

sns.heatmap(conf_matrixensemble_soft, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title('Confusion Matrix - ENSEMBLE (Percentage Split)')
axes[1, 1].set_xlabel('Predicted Label')
axes[1, 1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

print("\n")
print("Akurasi Data Testing SVM: {:.2f}%". format(test_accuracySVM*100))
print("Akurasi Data Testing KNN: {:.2f}%". format(test_accuracyKNN*100))
print("Akurasi Data Testing CART_C : {:.2f}%". format(test_accuracyCART_C*100))
print("Akurasi Data Testing Ensemble: {:.2f}%". format(test_accuracy_ensemble_soft*100))

print("\nclassification_report SVM:")
print(classification_report(y_test, y_testSVM))
print(f"Akurasi: {accuracySVM:.2f}%")
print(f"Presisi: {precisionSVM:.2f}%")
print(f"Recall: {recallSVM:.2f}%")

print("\nclassification_report KNN:")
print(classification_report(y_test, y_testKNN))
print(f"Akurasi: {accuracyKNN:.2f}%")
print(f"Presisi: {precisionKNN:.2f}%")
print(f"Recall: {recallKNN:.2f}%")

print("\nclassification_report CART_R:")
print(classification_report(y_test, y_testCART_C))
print(f"Akurasi: {accuracyCART_C:.2f}%")
print(f"Presisi: {precisionCART_C:.2f}%")
print(f"Recall: {recallCART_C:.2f}%")

print("\nclassification_report ENSEMBLE:")
print(classification_report(y_test, y_test_ensemble_soft))
print(f"Akurasi: {accuracy_ensemble_soft:.2f}%")
print(f"Presisi: {precision_ensemble_soft:.2f}%")
print(f"Recall: {recall_ensemble_soft:.2f}%")
print("\n")



# ### **K-FOLD Cross Validassion**


kfold = KFold(n_splits = 10, shuffle=True, random_state = 42)
acc_scores = []
for train_index, test_index in kfold.split(x, y):
  x_train, x_test = x.iloc[train_index], x.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# ALGORTIMA SVM
SVM = svm.SVC(kernel='linear', probability=True)
SVM.fit(x_train,  y_train)
predictionSVM = SVM.predict(x_train)
trainingSVM = accuracy_score(predictionSVM ,y_train)
y_testSVM = SVM.predict(x_test)
test_accuracySVM = accuracy_score(y_test,y_testSVM)
accuracySVM = accuracy_score(y_test, y_testSVM) * 100
precisionSVM = precision_score(y_test, y_testSVM ) * 100
recallSVM = recall_score(y_test, y_testSVM) * 100

# ALGORTIMA CART Classifier
CART_C = DecisionTreeClassifier(max_depth=14)
CART_C.fit(x_train,  y_train)
predictionCART_C = CART_C.predict(x_train)
trainingCART_C = accuracy_score(predictionCART_C ,y_train)
y_testCART_C = CART_C.predict(x_test)
test_accuracyCART_C = accuracy_score(y_test,y_testCART_C)
accuracyCART_C = accuracy_score(y_test, y_testCART_C) * 100
precisionCART_C = precision_score(y_test, y_testCART_C) * 100
recallCART_C = recall_score(y_test, y_testCART_C) * 100

# Algoritma KNN
KNN = KNeighborsClassifier(n_neighbors = 7)
KNN.fit(x_train,y_train)
predictionKNN = KNN.predict(x_train)
trainingKNN = accuracy_score(predictionKNN ,y_train)
y_testKNN = KNN.predict(x_test)
test_accuracyKNN = accuracy_score(y_test,y_testKNN)
accuracyKNN = accuracy_score(y_test, y_testKNN) * 100
precisionKNN = precision_score(y_test, y_testKNN) * 100
recallKNN = recall_score(y_test, y_testKNN) * 100


# Voting Ensemble (Soft)
ensemble_model = VotingClassifier(estimators=[('SVM', SVM), ('KNN', KNN),('CART_C', CART_C)], voting='soft')
ensemble_model.fit(x_train, y_train)
predictionensemble_model = ensemble_model.predict(x_train)
trainingensemble_model = accuracy_score(predictionensemble_model ,y_train)
y_testensemble_model = ensemble_model.predict(x_test)
test_accuracyensemble_model = accuracy_score(y_test,y_testensemble_model)
accuracyensemble_model = accuracy_score(y_test, y_testensemble_model) * 100
precisionensemble_model = precision_score(y_test, y_testensemble_model) * 100
recallensemble_model = recall_score(y_test, y_testensemble_model) * 100

conf_matrixSVM = confusion_matrix(y_test, y_testSVM)
print("Confusion Matrix SVM:")
print(conf_matrixSVM)

conf_matrixALR = confusion_matrix(y_test, y_testKNN)
print("Confusion Matrix KNN:")
print(conf_matrixKNN)

conf_matrixRF  = confusion_matrix(y_test, y_testCART_C)
print("Confusion Matrix CART_C :")
print(conf_matrixCART_C)

conf_matrixensemble_model = confusion_matrix(y_test, y_testensemble_model)
print("Confusion Matrix ENSEMBLE:")
print(conf_matrixensemble_model)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Use the correct variable names defined in the previous cell
sns.heatmap(conf_matrixSVM, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix - SVM (K-Fold)')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_ylabel('True Label')

sns.heatmap(conf_matrixKNN, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix - KNN (K-Fold)')
axes[0, 1].set_xlabel('Predicted Label')
axes[0, 1].set_ylabel('True Label')

sns.heatmap(conf_matrixCART_C, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix - CART (K-Fold)')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_ylabel('True Label')

sns.heatmap(conf_matrixensemble_model, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title('Confusion Matrix - ENSEMBLE (K-Fold)')
axes[1, 1].set_xlabel('Predicted Label')
axes[1, 1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

print("\n")
print("Akurasi Data Testing SVM: {:.2f}%". format(test_accuracySVM*100))
print("Akurasi Data Testing KNN: {:.2f}%". format(test_accuracyKNN*100))
print("Akurasi Data Testing CART_C: {:.2f}%". format(test_accuracyCART_C*100))
print("Akurasi Data Testing Ensemble: {:.2f}%". format(test_accuracyensemble_model*100))

print("\nclassification_report SVM:")
print(classification_report(y_test, y_testSVM))
print(f"Akurasi: {accuracySVM:.2f}%")
print(f"Presisi: {precisionSVM:.2f}%")
print(f"Recall: {recallSVM:.2f}%")

print("\nclassification_report KNN:")
print(classification_report(y_test, y_testKNN))
print(f"Akurasi: {accuracyKNN:.2f}%")
print(f"Presisi: {precisionKNN:.2f}%")
print(f"Recall: {recallKNN:.2f}%")

print("\nclassification_report CART_C :")
print(classification_report(y_test, y_testCART_C))
print(f"Akurasi: {accuracyCART_C:.2f}%")
print(f"Presisi: {precisionCART_C:.2f}%")
print(f"Recall: {recallCART_C:.2f}%")

print("\nclassification_report ENSEMBLE:")
print(classification_report(y_test, y_testensemble_model))
print(f"Akurasi: {accuracyensemble_model:.2f}%")
print(f"Presisi: {precisionensemble_model:.2f}%")
print(f"Recall: {recallensemble_model:.2f}%")