sic = SimpleImputer(strategy='constant', fill_value='missing')
# X_train['Cabin'] = sic.fit_transform(X_train[['Cabin']])
# X_test['Cabin'] = sic.transform(X_test[['Cabin']])