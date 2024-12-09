# Going down the line: Decision Tree, Isolation Forest, K-means Clustering, KNN, Linear Regression, Logistic Regression, Naive Bayes, Random Forest (copy pasted without graphs or logs, while also changing generic variables, now prefixed with initials (eg: isolation forest = if_))

from preprocessing import * # import np, pd, sklearn fn's --and df (https://www.kaggle.com/datasets/kartik2112/fraud-detection/)

pca = PCA(n_components = 10)
X_Train_pca = pca.fit_transform(X_train_balanced)
X_Test_pca = pca.transform(X_test_balanced)

print("pca done")

"""Decision Tree.ipynb"""
#Instantiating DecisionTreeClassifier object
ccfd_decisiontree = DecisionTreeClassifier()

#Training the model
ccfd_decisiontree.fit(X_train_balanced, y_train_balanced)

#Using newly trained model with X_test_balanced to predict for Credit Card Fraud
decisiontree_prediction = ccfd_decisiontree.predict(X_test_balanced)

#Evaluating accuracy of model by comparing predicted labels with y_test values (actual)
score_dt = metrics.accuracy_score(y_test_balanced, decisiontree_prediction)

#Estimating the probability of Credit Card Fraud Label
decisiontree_prediction_probability = ccfd_decisiontree.predict_proba(X_test_balanced)

#Calculating true positive rate(tpr) and false positive rate(fpr)
dt_fpr, dt_tpr, dt_thresholds = metrics.roc_curve(y_test_balanced, decisiontree_prediction_probability[:,1], pos_label=1)

#Calculating Area Under Curve
dt_AUC = metrics.auc(dt_fpr, dt_tpr)

#Instantiating DecisionTreeClassifier object
ccfd_decisiontree_pca = DecisionTreeClassifier(random_state = 7)

#Training the model
ccfd_decisiontree_pca.fit(X_Train_pca,  y_train_balanced)

#Using newly trained model with X_test_balanced to predict for Credit Card Fraud
decisiontree_prediction_pca = ccfd_decisiontree_pca.predict(X_Test_pca)

#Evaluating accuracy of model by comparing predicted labels with y_test values (actual)
score_dt_pca = metrics.accuracy_score(y_test_balanced, decisiontree_prediction_pca)

#Estimating the probability of Credit Card Fraud Label
decisiontree_prediction_probability_pca = ccfd_decisiontree_pca.predict_proba(X_Test_pca)

#Calculating true positive rate(tpr) and false positive rate(fpr)
dt_pca_fpr, dt_pca_tpr, pdt_pca_thresholds = metrics.roc_curve(y_test_balanced, decisiontree_prediction_probability_pca[:,1], pos_label=1)

#Calculating Area Under Curve
dt_pca_AUC = metrics.auc(dt_pca_fpr, dt_pca_tpr)

print("Decision Tree done")

"""Isolation Forest.ipynb"""
# Train Isolation Forest
ccfd_iso_forest = IsolationForest(
    n_estimators=100,
    contamination='auto',  # Automatically estimate contamination (fraud rate)
    random_state=42
)
ccfd_iso_forest.fit(X_train_balanced)

# Predict Outliers
cached_X_train_balanced = X_train_balanced.copy() # cannot add column directly to X_train_balanced AND also run ccfd_iso_forest.predict on itself
cached_X_train_balanced['anomaly_score'] = ccfd_iso_forest.decision_function(X_train_balanced)  # Higher scores = normal
cached_X_train_balanced['is_anomaly'] = ccfd_iso_forest.predict(X_train_balanced)  # -1 = anomaly, 1 = normal

# Map Isolation Forest anomalies to Fraud label
cached_X_train_balanced['predicted_fraud'] = cached_X_train_balanced['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Calculate ROC curve
if_fpr, if_tpr, if_thresholds = metrics.roc_curve(y_train_balanced, -cached_X_train_balanced['anomaly_score'])  # Negative scores for alignment with anomaly detection
if_roc_auc = metrics.auc(if_fpr, if_tpr)  # Calculate AUC

"""K-means Clustering.ipynb"""
#Instantiating KMeans classifier
ccfd_Kmeans = KMeans(n_clusters=2, random_state=4) # 1 cluster for fraud, 1 cluster for legit

#Training the model
ccfd_Kmeans.fit(X_train_balanced)

label_clustered = ccfd_Kmeans.labels_

#Using newly trained model with X_test_balanced to predict for Credit Card Fraud
KMeans_predictions = ccfd_Kmeans.predict(X_test_balanced)

#Evaluating accuracy of model by comparing predicted labels with y_test values (actual)
score_gnb = metrics.accuracy_score(y_test_balanced, KMeans_predictions)

print("Isolation Tree done")

"""KNN.ipynb"""
# Too Large a Dataset for KNN, ENSURE PCA Done First (Principal Component Analysis - an unsupervised dimensionality reduction)
#Instantiating KNeighborsClassifier object
k=3
ccfd_kneighbors = KNeighborsClassifier(n_neighbors=k, weights = 'uniform')

#Training the model
ccfd_kneighbors.fit(X_Train_pca,  y_train_balanced)

#Using newly trained model with X_test_balanced to predict for Credit Card Fraud
kneighbors_prediction = ccfd_kneighbors.predict(X_Test_pca)

#Evaluating accuracy of model by comparing predicted labels with y_test values (actual)
score_kn = metrics.accuracy_score(y_test_balanced, kneighbors_prediction)

#Estimating the probability of Credit Card Fraud Label
kneighbors_prediction_probability = ccfd_kneighbors.predict_proba(X_Test_pca)

#Calculating true positive rate(tpr) and false positive rate(fpr)
knn_pca_fpr, knn_pca_tpr, knn_pca_thresholds = metrics.roc_curve(y_test_balanced, kneighbors_prediction_probability[:,1], pos_label=1)

#Calculating Area Under Curve
knn_AUC = metrics.auc(knn_pca_fpr, knn_pca_tpr)

print("KNN done")

"""Linear Regression.ipynb"""
ccfd_linear_regression = linear_model.LinearRegression()
ccfd_linear_regression.fit(X_train_balanced, y_train_balanced)

# Make predictions on the test set
ccfd_linear_regression_predictions = ccfd_linear_regression.predict(X_test_balanced)

# Evaluate the model
mse = metrics.mean_squared_error(y_test_balanced, ccfd_linear_regression_predictions)
r_squared = metrics.r2_score(y_test_balanced, ccfd_linear_regression_predictions)

print("Linear Regression done")

"""Logistic Regression.ipynb"""
#Instantiating LogisticRegressionClassifier object
ccfd_logisticregression = linear_model.LogisticRegression(random_state = 7)

#Training the model
ccfd_logisticregression.fit(X_train_balanced,  y_train_balanced)

#Using newly trained model with X_test_balanced to predict for Credit Card Fraud
logisticregression_prediction = ccfd_logisticregression.predict(X_test_balanced)

#Evaluating accuracy of model by comparing predicted labels with y_test values (actual)
score_lr = metrics.accuracy_score(y_test_balanced, logisticregression_prediction)

#Estimating the probability of Credit Card Fraud Label
logisticregression_prediction_probability = ccfd_logisticregression.predict_proba(X_test_balanced)

#Calculating true positive rate(tpr) and false positive rate(fpr)
lr_fpr, lr_tpr, lr_thresholds = metrics.roc_curve(y_test_balanced, logisticregression_prediction_probability[:,1], pos_label=1)

#Calculating Area Under Curve
lr_AUC = metrics.auc(lr_fpr, lr_tpr)

#Instantiating LogisticRegressionClassifier object
ccfd_logisticregression_pca = linear_model.LogisticRegression(random_state = 7)

#Training the model
ccfd_logisticregression_pca.fit(X_Train_pca,  y_train_balanced)

#Using newly trained model with X_test_balanced to predict for Credit Card Fraud
logisticregression_prediction_pca = ccfd_logisticregression_pca.predict(X_Test_pca)

#Evaluating accuracy of model by comparing predicted labels with y_test values (actual)
score_lr_pca = metrics.accuracy_score(y_test_balanced, logisticregression_prediction_pca)

#Estimating the probability of Credit Card Fraud Label
logisticregression_prediction_probability_pca = ccfd_logisticregression_pca.predict_proba(X_Test_pca)

#Calculating true positive rate(tpr) and false positive rate(fpr)
lr_pca_fpr, lr_pca_tpr, lr_pca_threshold = metrics.roc_curve(y_test_balanced, logisticregression_prediction_probability_pca[:,1], pos_label=1)

#Calculating Area Under Curve
pca_lr_AUC = metrics.auc(lr_pca_fpr, lr_pca_tpr)

print("Logistic Regression done")

"""Naive Bayes.ipynb"""
#Instantiating GaussianNBClassifier object
ccfd_gaussianNB = GaussianNB()

#Training the model
ccfd_gaussianNB.fit(X_train_balanced,  y_train_balanced)

#Using newly trained model with X_test_balanced to predict for Credit Card Fraud
gaussianNB_prediction = ccfd_gaussianNB.predict(X_test_balanced)

#Evaluating accuracy of model by comparing predicted labels with y_test values (actual)
score_gnb = metrics.accuracy_score(y_test_balanced, gaussianNB_prediction)

#Estimating the probability of Credit Card Fraud Label
gaussianNB_prediction_probability = ccfd_gaussianNB.predict_proba(X_test_balanced)

#Calculating true positive rate(tpr) and false positive rate(fpr)
nb_fpr, nb_tpr, nb_thresholds = metrics.roc_curve(y_test_balanced, gaussianNB_prediction_probability[:,1], pos_label=1)

#Calculating Area Under Curve
nb_AUC = metrics.auc(nb_fpr, nb_tpr)

#Instantiating GaussianNBClassifier object
ccfd_gaussianNB_pca = GaussianNB()

#Training the model
ccfd_gaussianNB_pca.fit(X_Train_pca,  y_train_balanced)

#Using newly trained model with X_test_balanced to predict for Credit Card Fraud
gaussianNB_prediction_pca = ccfd_gaussianNB_pca.predict(X_Test_pca)

#Evaluating accuracy of model by comparing predicted labels with y_test values (actual)
score_gnb_pca = metrics.accuracy_score(y_test_balanced, gaussianNB_prediction_pca)

#Estimating the probability of Credit Card Fraud Label
gaussianNB_prediction_probability_pca = ccfd_gaussianNB_pca.predict_proba(X_Test_pca)

#Calculating true positive rate(tpr) and false positive rate(fpr)
nb_pca_fpr, nb_pca_tpr, nb_pca_thresholds = metrics.roc_curve(y_test_balanced, gaussianNB_prediction_probability_pca[:,1], pos_label=1)

#Calculating Area Under Curve
nb_pca_AUC = metrics.auc(nb_pca_fpr, nb_pca_tpr)

print("Naive Bayes done")

"""Random Forest.ipynb"""
#Instantiating RandomForestClassifier object
ccfd_randomforest = RandomForestClassifier(n_estimators = 19, bootstrap = True, random_state = 7)

#Training the model
ccfd_randomforest.fit(X_train_balanced,  y_train_balanced)

#Using newly trained model with X_test_balanced to predict for Credit Card Fraud
randomforest_prediction = ccfd_randomforest.predict(X_test_balanced)

#Evaluating accuracy of model by comparing predicted labels with y_test values (actual)
score_rf = metrics.accuracy_score(y_test_balanced, randomforest_prediction)

#Estimating the probability of Credit Card Fraud Label
randomforest_prediction_probability = ccfd_randomforest.predict_proba(X_test_balanced)

#Calculating true positive rate(tpr) and false positive rate(fpr)
rf_fpr, rf_tpr, rf_thresholds = metrics.roc_curve(y_test_balanced, randomforest_prediction_probability[:,1], pos_label=1)

#Calculating Area Under Curve
rf_AUC = metrics.auc(rf_fpr, rf_tpr)

#Instantiating RandomForestClassifier object
ccfd_randomforest_pca = RandomForestClassifier(n_estimators = 19, bootstrap = True, random_state = 7)

#Training the model w new pca training set
ccfd_randomforest_pca.fit(X_Train_pca,  y_train_balanced)

#Using newly trained model with X_test_balanced to predict for Credit Card Fraud using new pca testing set
randomforest_prediction_pca = ccfd_randomforest_pca.predict(X_Test_pca)

#Evaluating accuracy of model by comparing predicted labels with y_test values (actual)
score_rf_pca = metrics.accuracy_score(y_test_balanced, randomforest_prediction_pca)

#Estimating the probability of Credit Card Fraud Label
randomforest_prediction_probability_pca = ccfd_randomforest_pca.predict_proba(X_Test_pca)

#Calculating true positive rate(tpr) and false positive rate(fpr)
rf_pca_fpr, rf_pca_tpr, rf_pca_thresholds = metrics.roc_curve(y_test_balanced, randomforest_prediction_probability_pca[:,1], pos_label=1)

#Calculating Area Under Curve
rf_pca_AUC = metrics.auc(rf_pca_fpr, rf_pca_tpr)

print("Random Forest done")
