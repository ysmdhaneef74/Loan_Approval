# Loan Approval Prediction using Machine Learning

## Project Overview

This project aims to predict the risk of clients applying for loans using machine learning techniques. The dataset includes various features such as age, gender, income, loan amount, and credit score, which are used to predict whether a client is high risk or low risk for loan approval (Risk_Flag).

## Table of Contents
- [Project Overview](#project-overview)
- [Data Exploration](#data-exploration)
- [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Data Exploration

Initial steps include loading the dataset and performing exploratory data analysis (EDA) to understand the structure and distribution of the data.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('loan_approval.csv')

# Display the first few rows of the dataset
print(df.head())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Distribution of the target variable
sns.countplot(x='Risk_Flag', data=df)
plt.title('Distribution of Risk_Flag')
plt.show()
