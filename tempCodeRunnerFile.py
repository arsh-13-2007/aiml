X_num = X[['ApplicantIncome']].values
# print(type(X_num))

# X = np.hstack((X_cat , X_num))
# print(type(X))

# X= pd.DataFrame(X)
# print(type(X))
# print(X.head())
# """ label encoding"""

# y = le.fit_transform(y)   # label encoding is only applies on output column not input columns (important)

# print(type(y))
# y= pd.DataFrame(y)
# print(type(y))
# print(y.head())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)