import pandas as pd
import numpy as np
import xlsxwriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Input file dan memasukkan kedalam variabel
df_Data = pd.read_excel('DataSetTB3_SHARE.xlsx', sheet_name="Data")
X = df_Data.iloc[:, 2:]
Y = df_Data['label']

# split isi dari sheet Data menjadi 70 : 30
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)

# -----------------------------------------------------------------------------------------------------------------------------------
# Analisis Data latih dan Penentapan KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform',
                           algorithm='auto', leaf_size=30, metric='euclidean')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# -----------------------------------------------------------------------------------------------------------------------------------
# # Membuat file excel "OutputLatih.xlsx"
df1 = pd.DataFrame(X_test, columns=['idData'])
numpy_array = np.array(y_pred_knn)
print(df1.index)
workbook = xlsxwriter.Workbook('OutputLatih.xlsx')
worksheet = workbook.add_worksheet("Data")
worksheet.write(0, 0, 'idData')
worksheet.write(0, 1, 'Klasifikasi')
worksheet.write('C1', 'Akurasi')
worksheet.write('C2', accuracy_score(y_test, y_pred_knn))
start = 1
for i in range(300):
    worksheet.write(start, 0, df1.index[i])
    worksheet.write(start, 1, numpy_array[i])
    start += 1
workbook.close()
# -----------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------------------

# Input sheet submit
df_submit = pd.read_excel('DataSetTB3_SHARE.xlsx', sheet_name="Submit")

X_submit = df_submit.iloc[:, 1:]
# Analisis Data Submit
knn.fit(X, Y)
y_pred_knn_submit = knn.predict(X_submit)
print(y_pred_knn_submit)

# Membuat file excel "OutputSubmit.xlsx"
df2 = pd.DataFrame(X_submit, columns=['idData'])
numpy_array_submit = np.array(y_pred_knn_submit)
print(df1.index)
workbook = xlsxwriter.Workbook('OutputSubmit.xlsx')
worksheet = workbook.add_worksheet("Submit")
worksheet.write(0, 0, 'idData')
worksheet.write(0, 1, 'Klasifikasi')
start = 1
for i in range(500):
    worksheet.write(start, 0, df2.index[i]+1)
    worksheet.write(start, 1, numpy_array_submit[i])
    start += 1
workbook.close()

# -----------------------------------------------------------------------------------------------------------------------------------

# ===========================================================================================
# Eksperimen metric dan nilai k agar bisa mendapatkan nilai akurasi terbaik
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform',
                           algorithm='auto', leaf_size=30, metric='euclidean')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Akurasi euclidean dengan k = 5 adalah",
      accuracy_score(y_test, y_pred_knn))
knn = KNeighborsClassifier(n_neighbors=8, weights='uniform',
                           algorithm='auto', leaf_size=30, metric='euclidean')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Akurasi euclidean dengan k = 8 adalah",
      accuracy_score(y_test, y_pred_knn))
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform',
                           algorithm='auto', leaf_size=30, metric='manhattan')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Akurasi manhattan dengan k = 5 adalah",
      accuracy_score(y_test, y_pred_knn))
knn = KNeighborsClassifier(n_neighbors=8, weights='uniform',
                           algorithm='auto', leaf_size=30, metric='manhattan')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Akurasi manhattan dengan k = 8 adalah",
      accuracy_score(y_test, y_pred_knn))
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform',
                           algorithm='auto', leaf_size=30, metric='minkowski')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Akurasi minkowski dengan k = 5 adalah",
      accuracy_score(y_test, y_pred_knn))
knn = KNeighborsClassifier(n_neighbors=8, weights='uniform',
                           algorithm='auto', leaf_size=30, metric='minkowski')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Akurasi minkowski dengan k = 8 adalah",
      accuracy_score(y_test, y_pred_knn))
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform',
                           algorithm='auto', leaf_size=30, metric='chebyshev')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Akurasi chebyshev dengan k = 5 adalah",
      accuracy_score(y_test, y_pred_knn))
knn = KNeighborsClassifier(n_neighbors=8, weights='uniform',
                           algorithm='auto', leaf_size=30, metric='chebyshev')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Akurasi chebyshev dengan k = 8 adalah",
      accuracy_score(y_test, y_pred_knn))
# ===========================================================================================
