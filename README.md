***This project is done for practice purpose only***

Project: **Fraud Credit Card Transaction Detector**

Dataset: creditcard.csv (https://www.kaggle.com/mlg-ulb/creditcardfraud)

Steps:

1.  Import the libraries.
2.  Import the dataset.
3.  Explore the dataset.
4.  Setting X and y for deploying the Machine Learning algorithms.
5.  Plotting Bar Plot for comparing number of Fraud Transactions vs Normal Transactions.
6.  Plotting Seaborn Heatmap to check the corelation between the attributes (columns).
7.  Splitting the dataset to Train and Test data (X_train, y_train, X_test, y_test) in the ratio 80:20.
8.  Standardization of X_train and X_test with StandardScaler.
9.  Use Decision tree Classfiier to classify between Normal and Fraud Transaction and fit classifier on X_train and y_train.
10. Based on the fitted classifier, predict the classfication values for given X_test data.
11. Build Confusion Matrix for the Performance Evaluation of the Decision Tree Classfier built above.
12. Find Accuracy, Error Rate, Precision, Recall based on the Confusion Matrix.
13. Now, use Support Vector Machine Classifier for the same data to classfiy between Fraud and Normal transactions. Fit the SVM on X_train, y_train.
14. Now predict the classification of X_test.
15. Build Confusion Matrix for Performance Evaluation of SVM Classifier.
16. Find Accuracy, Error Rate, Precision, Recall based on the Confusion Matrix above.
17. Compare the Performance of both the Algorithms and select the one with better performance.
