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