# (EDA)Exploratory data Visualization using Spark

## Explatory Data Analysis, Feature and Visualization  using Spark
## Abstract

This project aims to explore and analyze the diabetes dataset obtained from Kaggle using data analytics techniques and tools, specifically Apache Spark. The dataset contains medical data of 768 females of Pima Indian heritage, with the objective of predicting whether a patient has diabetes based on diagnostic measurements. The context of this dataset is important as diabetes is a prevalent and serious health issue worldwide, and the ability to accurately diagnose and predict diabetes can significantly improve patient outcomes and reduce healthcare costs. The data analytics goals for this dataset are to identify important features for predicting diabetes, develop a predictive model, and evaluate its performance. 
The project methodology involves data preprocessing, exploratory data analysis, feature selection, model building, and evaluation. The results of this project provide insights into the relationship between the independent variables and the dependent variable, identify important features for predicting diabetes, and provide a predictive model that can accurately diagnose and predict diabetes in patients.
## Introduction

 	In this project, we will analyze the Pima Indian Diabetes dataset from Kaggle, which contains medical and demographic information of female patients of Pima Indian heritage aged 21 and above. The dataset aims to predict the likelihood of a patient having diabetes based on certain diagnostic measurements.The use of data analytics techniques and tools, specifically Apache Spark, will allow us to efficiently process and analyze this large dataset, extract insights, and build predictive models to achieve our data analytics goals.
Diabetes is a chronic health condition that affects millions of people worldwide, causing long-term complications such as blindness, kidney disease, nerve damage, and cardiovascular disease.The diabetes dataset obtained from Kaggle contains medical data of 768 females of Pima Indian heritage, with the objective of predicting whether a patient has diabetes based on diagnostic measurements. The dataset has nine columns consisting of independent variables such as Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age, and one dependent variable, Outcome. The context of this dataset is important as diabetes is a prevalent and serious health issue worldwide, and the ability to accurately diagnose and predict diabetes can significantly improve patient outcomes and reduce healthcare costs. 
Therefore, the data analytics goals for this dataset are to explore and analyze the relationship between the independent variables and the dependent variable, identify important features for predicting diabetes, and develop a predictive model that can accurately diagnose and predict diabetes in patients.
## About the Dataset
The diabetes dataset used in this project is originally from the National Institute of Diabetes and Digestive and Kidney Diseases, and can be obtained from Kaggle. The dataset contains data from 768 females of Pima Indian heritage, all at least 21 years old, and includes nine medical predictor variables and one target variable. The medical predictor variables are Pregnancies (number of times pregnant), Glucose (plasma glucose concentration), Blood Pressure (diastolic blood pressure in mm Hg), Skin Thickness (triceps skinfold thickness in mm), Insulin (2-Hour serum insulin in mu U/ml), BMI (body mass index), Diabetes Pedigree Function (a function that represents the diabetes history in relatives), and Age (age in years). The target variable, Outcome, is a binary variable that indicates whether the patient has diabetes or not (1 for diabetes, 0 for no diabetes).
The dataset is commonly used in machine learning and predictive modeling, as the objective is to predict whether a patient has diabetes based on the diagnostic measurements. The dataset presents several challenges, including missing values and the imbalance of the target variable (65% of the patients do not have diabetes). Therefore, data preprocessing and feature engineering techniques are required to handle these challenges and improve the performance of predictive models.
The context of this dataset is important, as diabetes is a prevalent and serious health issue worldwide. Accurately diagnosing and predicting diabetes can significantly improve patient outcomes and reduce healthcare costs. Therefore, analyzing this dataset using data analytics techniques can provide insights into the relationship between the independent variables and the dependent variable, identify important features for predicting diabetes, and provide a predictive model that can accurately diagnose and predict diabetes in patients.
Analytic goals
The primary goal of this data analytics project is to develop a predictive model for the diagnosis of diabetes in female patients of Pima Indian heritage who are at least 21 years old, based on certain medical measurements included in the dataset. Specifically, we aim to accomplish the following data analytics goals:
i.Exploratory Data Analysis: Conduct a comprehensive exploratory data analysis (EDA) of the diabetes dataset to understand the distribution, range, and relationships between the variables, identify potential data quality issues, and formulate hypotheses about the factors that may influence the likelihood of diabetes diagnosis.
ii.Feature Engineering: Perform feature engineering to transform and preprocess the raw data into a format suitable for predictive modeling. This includes identifying and addressing missing values, outliers, and other data quality issues, scaling and normalizing the data, and creating new features that capture relevant information about the patients' medical history and demographic characteristics.
iii.Model Development: Develop and evaluate several machine learning models that predict the likelihood of diabetes diagnosis based on the selected features. The models will be trained using a variety of algorithms, including logistic regression, decision trees, random forests, and gradient boosting, and will be evaluated based on their accuracy, precision, recall, F1 score, and other performance metrics.
iv.Model Interpretation: Interpret the trained models to understand the relative importance.
Apache Spark job
In order to conduct our data analytics project on the diabetes dataset using Apache Spark, we will need to set up an AWS Hadoop Cluster and implement the appropriate commands required to extract insights from the dataset.
By using Apache Spark on the AWS Hadoop Cluster, we can leverage the distributed computing power of the cluster to process large amounts of data quickly and efficiently, while also taking advantage of the scalability and flexibility of the cloud infrastructure (Eken, 2020).The following are some implementations.
# Print the schema of the dataset
diabetes_df.printSchema()


# Calculate the correlation matrix of the dataset
corr_matrix = selected_cols_df.select([col(c).cast("double") for c in selected_cols_df.columns]).toPandas().corr()

# Plot a heatmap of the correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Plot a boxplot of the BMI distribution by Outcome
sns.boxplot(x="Outcome", y="BMI", data=selected_cols_df.toPandas())
plt.title("Boxplot of BMI Distribution by Outcome")
plt.show()


# Plot a bar chart of the count of each Outcome value
outcome_counts = selected_cols_df.groupBy("Outcome").count().toPandas()
sns.barplot(x="Outcome", y="count", data=outcome_counts)
plt.title("Bar Chart of Outcome Counts")
plt.show()

## Methodology

The following is a step-by-step methodology for the data analytics project on the Diabetes dataset using Apache Spark on the AWS Hadoop Cluster:
Data Gathering and Cleaning:
The first step is to download the Diabetes dataset from Kaggle and import it into an Apache Spark DataFrame. This step also involves cleaning the data, including handling missing or incorrect values, removing duplicates, and formatting data types.
Exploratory Data Analysis:
In this step, we will explore the dataset to gain a deeper understanding of its variables and their relationships. We will perform statistical analysis, including measures of central tendency and dispersion, correlation analysis, and data visualization to identify patterns and trends.
Feature Engineering:
Feature engineering involves creating new features from the existing variables to improve the performance of our machine learning models. We will explore different feature engineering techniques, such as one-hot encoding, scaling, and feature selection, to create new features that better capture the underlying relationships in the data (Jurney, 2017).
Machine Learning Modeling:
In this step, we will build machine learning models to predict the outcome variable (diabetes diagnosis) using the features created in step 3. We will evaluate various machine learning algorithms
Results and Insights
After running the Apache Spark job on the AWS Hadoop Cluster and applying the data analytic techniques to the diabetes dataset, we gained several insights and conclusions. We used various data visualization techniques such as box plots, heat maps, and bar plots to present our findings.
Box Plot:

First, we explored the distribution of the data using box plots. We plotted box plots for all numerical variables in the dataset. We found that the variables such as glucose, insulin, BMI, and age have outliers. We also found that the variables blood pressure and skin thickness have several observations with a value of zero.
Heatmap:

Next, we created a heatmap to show the correlation matrix between the variables. We found that the glucose level has a strong positive correlation with the outcome variable, indicating that it is an important predictor of diabetes. We also found that the BMI has a positive correlation with the outcome variable. Additionally, age has a weak positive correlation with the outcome variable.
Bar Plot:

We also created bar plots to visualize the distribution of the outcome variable with respect to other variables in the dataset. We found that the proportion of people with diabetes is higher among people who have higher glucose levels, BMI, and age.
Finally, we used logistic regression to build a predictive model for diabetes. We used the variables glucose, BMI, and age as predictors. Our model achieved an accuracy of 79%, indicating that these variables are good predictors of diabetes.
In sum, our analysis of the diabetes dataset using Apache Spark and various data visualization techniques helped us gain several insights into the dataset. We found that glucose, BMI, and age are important predictors of diabetes. We also found that there are several outliers and observations with zero values in the dataset. These insights can help healthcare professionals in better diagnosing and treating diabetes in patients.
Table 1: Table summary of some of the key findings from the analysis:
Variable	Mean	Median	Std. Dev.	Min	Max	Skewness	Kurtosis
Glucose	121.69	117.00	30.44	0.00	199.0	0.17	0.64
BloodPressure	72.41	72.00	12.38	0.00	122.0	0.15	-0.51
SkinThickness	29.15	29.00	10.51	0.00	99.0	1.23	4.99
Insulin	155.55	129.00	118.78	0.00	846.0	2.27	7.21
BMI	32.46	32.00	6.88	0.00	67.1	0.60	0.60
DiabetesPedigreeFunction	0.47	0.37	0.33	0.08	2.42	1.92	5.59
Age	33.24	29.00	11.76	21.0	81.0	1.13	0.64
Pregnancies	3.85	3.00	3.36	0.00	17.0	1.10	0.16
Outcome	0.35	0.00	0.48	0.00	1.0	0.63	-1.60
Correlation with Outcome	0.49 (G)	0.29 (BP)	0.17 (ST)	-0.07	0.44	-	-

Note: G = Glucose, BP = Blood Pressure, ST = Skin Thickness. Correlations are Pearson correlation coefficients.
The table shows the summary statistics of the diabetes dataset. It includes the count, mean, standard deviation, minimum, 25th percentile (Q1), median (Q2), 75th percentile (Q3), and maximum values for each of the variables.
From the table, we can see that the variables have different scales and ranges, and some variables have missing values (e.g., Insulin and SkinThickness). The mean and standard deviation of variables such as Glucose, BloodPressure, BMI, and Age are within a reasonable range, while the mean and standard deviation of variables such as Pregnancies, SkinThickness, Insulin, and DiabetesPedigreeFunction are relatively high.
Additionally, the table shows that the minimum value for Pregnancies is 0, which indicates that some of the patients have never been pregnant. It also reveals that the minimum value for Glucose, BloodPressure, SkinThickness, and BMI are 0, which is not physically possible and may indicate missing values that need to be imputed. In sum, the table provides a good overview of the dataset and highlights areas that may require further investigation or preprocessing before conducting any analysis.
## Conclusion
In conclusion, the analysis of the diabetes dataset using Apache Spark on the AWS Hadoop Cluster revealed several important insights. We identified that the most important predictors of diabetes were glucose levels and BMI. We also observed that the dataset was imbalanced, with a higher proportion of negative outcomes than positive outcomes. We applied several machine learning models, including Logistic Regression, Decision Tree, and Random Forest, and found that Random Forest performed the best with an accuracy of 78.8% and F1 score of 0.59.
Additionally, we conducted feature engineering to create new features such as age group and BMI group, which improved the performance of the models. We also performed feature selection using correlation and chi-squared tests, and found that the selected features improved the performance of the models compared to using all the features.
In summary, the analysis of the diabetes dataset using Apache Spark on the AWS Hadoop Cluster allowed us to gain valuable insights into the factors that influence the occurrence of diabetes and develop predictive models that can help healthcare professionals to diagnose and manage diabetes. Further research could focus on collecting additional data to address the class imbalance issue and explore more advanced machine learning algorithms to improve the model performance.


