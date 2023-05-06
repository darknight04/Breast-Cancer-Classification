# Breast-Cancer-Classification

This project focuses on building a Decision Tree classifier to predict whether a breast mass is malignant or benign using the Breast Cancer Dataset provided by scikit-learn. The dataset includes 569 samples with 30 features each, and has two classes: malignant and benign.

## Dependencies
Python 3.10.4
Scikit-learn
Numpy
Pandas
Matplotlib
Project Overview
The project is divided into four parts:

## Part 1 & 2:
In this part, we explored the dataset and computed statistics for the attribute values. This gave us an idea of the range and distribution of the attribute values in the dataset.

## Part 3:
To build the initial Decision Tree classifier, we split the data into training and testing sets, and standardized the data using the StandardScaler method. The initial Decision Tree classifier achieved an accuracy of 91.93%.

## Part 4:
In this part, we tried three additional preprocessing methods to improve the results: feature selection, PCA, and oversampling. For feature selection, we used the SelectKBest method to select the top 10 features with the highest correlation with the target variable. This improved the accuracy of the model to 94.16%. Next, we tried using PCA to reduce the dimensionality of the dataset. We found that reducing the number of components to 10 improved the accuracy to 95.32%. Finally, we used oversampling techniques to balance the class distribution in the dataset. We tried both the SMOTE and ADASYN methods and found that SMOTE gave us the best results with an accuracy of 95.77%.

## Conclusion
Using a combination of preprocessing methods, we were able to improve the accuracy of the initial Decision Tree classifier by over 3%, achieving an accuracy of 95.77%. Feature selection helped to reduce the noise in the data and improve the accuracy. PCA reduced the dimensionality of the dataset, which allowed the model to better generalize to new data. Oversampling helped to balance the class distribution, which was important since the dataset had more samples of one class than the other. It is interesting to note that while oversampling improved the results, it did not give us as big of a boost as we had expected, which suggests that the model was already doing a good job of generalizing to new data even without the balanced class distribution. Overall, this project highlights the importance of careful preprocessing in achieving high accuracy for classification tasks.
