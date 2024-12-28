# -*- coding: utf-8 -*-
"""Instructor Session01 | Exploratory Data Analysis | CIOL Winter ML BootCamp

### What is **Exploratory Data Analysis (EDA)**?  
Exploratory Data Analysis (EDA) is a process of examining and analyzing data to uncover patterns, detect anomalies, test hypotheses, and summarize its main characteristics, often using visual and statistical techniques. It is typically the first step in the data analysis pipeline and helps in understanding the data structure, relationships, and underlying patterns.

### Why is EDA Important?  
EDA is critical for several reasons:  
1. **Data Understanding**:  It provides a deep understanding of the dataset, such as its shape, size, types of variables, and missing values.  
2. **Data Cleaning**:   Identifies and resolves issues like missing, duplicate, or erroneous data, ensuring the quality of the dataset.
3. **Hypothesis Formation**:   Helps formulate questions or hypotheses for further analysis or modeling.
4. **Feature Selection and Engineering**:   Reveals which features (columns) are relevant, enabling efficient and effective feature engineering.  
5. **Guiding Model Selection**:    Gives insights into which types of models may perform best (e.g., regression vs. classification models).  
6. **Preventing Biases**:    Detects skewness, outliers, or imbalances that could lead to biased or misleading results.

### Why is EDA Done in Every Data Analysis or Research Project?  
EDA is essential in every project because:  
1. **Ensures Data Integrity**:   Without EDA, undetected issues can lead to incorrect conclusions or poorly performing models.  
2. **Foundation for Decision-Making**:   The insights gained through EDA drive informed decisions for preprocessing and model building.  
3. **Improves Efficiency**:    Helps prioritize resources by identifying irrelevant or redundant features early in the pipeline.  
4. **Identifies Limitations**:    Reveals limitations of the data, such as insufficient samples or biases, guiding more realistic analyses.

### What is Done in EDA?  
1. **Data Overview**:  Checking dataset size, column types, and data summary (mean, median, etc.).  
2. **Handling Missing Values**:   Identifying missing values and deciding on imputation or removal strategies.  
3. **Detecting Outliers**:   Using methods like box plots or statistical thresholds to detect anomalies.  
4. **Univariate Analysis**:   Analyzing individual variables using histograms, box plots, and descriptive statistics.  
5. **Bivariate and Multivariate Analysis**:   Exploring relationships between variables through scatter plots, correlation matrices, or pair plots.  
6. **Data Visualization**:   Visual tools (e.g., bar charts, line plots, heatmaps) to better understand data distributions and relationships.  
7. **Identifying Patterns and Trends**:   Observing patterns over time or across categories.  
8. **Hypothesis Testing**:   Testing initial hypotheses to validate assumptions.

---

# Download/Upload Data
"""

!git clone https://github.com/ciol-researchlab/CIOL-Winter-ML-Bootcamp.git

"""# 2. Setting up the enviroment

- [Pandas](https://pandas.pydata.org/docs/user_guide/index.html)
"""

# Tabular Data Analysis
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utility
import time
import warnings
warnings.filterwarnings('ignore')

"""# 3. Basic Python"""

# Python Data Types

# Integers and Floats
x = 10         # Integer
y = 3.14       # Float

# Strings
name = "Python"  # String

# Booleans
is_valid = True  # Boolean (True or False)

# Lists (ordered, mutable collections)
numbers = [1, 2, 3, 4, "A"]

# Tuples (ordered, immutable collections)
coordinates = (10, 20)

# Sets (unordered, unique elements)
unique_numbers = {1, 2, 3}

# Dictionaries (key-value pairs)
person = {"name": "Alice", "age": 25}

# None (represents "no value")
empty_value = None

# Check Data Types
print(type(unique_numbers))

# Conditional Logic

age = 20

if age < 18:
    print("You are a minor.")
elif 18 <= age <= 60:
    print("You are an adult.")
else:
    print("You are a senior citizen.")

x = 15

if x > 0:
    if x % 2 == 0:
        print("Positive even number")
    else:
        print("Positive odd number")
else:
    print("Not a positive number")

# Loop

# Iterating over a list
fruits = ["apple", "banana", "cherry"]
for i in fruits:
    print(i)

# Range-based loop
for i in range(2,5,2):  # 0 to 4
    print(i)

# Loop until a condition is met
count = 0
while count < 5:
    print(count)
    count += 1

# Break example
for i in range(10):
    if i == 5:
        break
    print(i)  # Stops when i == 5

# Continue example
for i in range(10):
    if i % 2 == 0:
        continue  # Skip even numbers
    print(i)  # Only prints odd numbers

person = {"name": "Alice", "age": 25, "city": "New York"}

# Accessing values
print(person["name"])  # Output: Alice

# Using get() to avoid KeyError
print(person.get("fathers name", "Key not found"))  # Output: 25

person["job"] = "Engineer"  # Add new key-value pair
person["age"] = 30          # Update existing value
print(person)

"""---

# 4. Load the dataset
"""

df = pd.read_csv("/content/CIOL-Winter-ML-Bootcamp/datasets/session1/main/spaceship-titanic/train.csv")

df.head(3)

df["HomePlanet"]

df[['PassengerId','RoomService','FoodCourt']]

"""# 5. Exploratory Data Analysis (EDA)

## 5.1. **Data Overview**
"""

#Check Dataset Size: The number of rows and columns in the dataset.

print(f"Dataset Size: {df.shape}")  # Rows and Columns

# Check Column Types: Information about columns and their data types.

df.dtypes

# Basic Summary Statistics

df.describe()

df.iloc[:, :-1].describe().T.sort_values(by='std' , ascending = False)\
                     .style.background_gradient(cmap='GnBu')\
                     .bar(subset=["max"], color='#BB0000')\
                     .bar(subset=["mean",], color='green')

# Unique Values: Number of unique values for each column.

df.nunique()

# Data Types and Memory Usage: Detailed information about the DataFrame.

df.info()

# Select numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
print("Numerical Columns:", numerical_columns)

# Select categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical Columns:", categorical_columns)

categorical_columns.remove('PassengerId')

# Filtering

df[df['VIP']==True]

"""## 5.2. **Handling Missing Values**"""

df.isnull()

# Check for missing values in each column
print(df.isnull().sum())

# Check for missing values as a percentage of the total
print(df.isnull().mean() * 100)

# Visualize missing values using a heatmap (requires seaborn)
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()

"""**Insights:**
- We can understand that there is no specific missing data pattern. All missing data seems random.

How to handle them?
"""

# Remove rows with missing values:

df_cleaned = df.dropna()
df_cleaned.shape

# Remove columns with missing values:

df_cleaned = df.dropna(axis=1)
df_cleaned.shape

# Dummy df
dfx = pd.DataFrame()

# Fill with a constant value:

dfx['HomePlanet'] = df['HomePlanet'].fillna(0)  # Replace missing with 0
print(df['HomePlanet'].isnull().sum())
print(dfx['HomePlanet'].isnull().sum())

df['RoomService'].max()

#Fill with the mean, median, or mode:

# Mean
dfx['RoomService'] = df['RoomService'].fillna(df['RoomService'].mean())

# Median
dfx['RoomService'] = df['RoomService'].fillna(df['RoomService'].median())

# Mode (most frequent value)
dfx['RoomService'] = df['RoomService'].fillna(df['HomePlanet'].mode()[0])

"""Use packages [scikit-learn](https://scikit-learn.org/1.5/modules/impute.html)

# 5.3. **Detecting and Fixing Outliers**
"""

df['RoomService']

sns.boxplot(data=df[numerical_columns])
plt.title('Box Plot of Multiple Columns')
plt.show()

# Violin plot for multiple columns
sns.violinplot(data=df[numerical_columns])
plt.title('Violin Plot of Multiple Columns')
plt.show()

"""**Insights:**
- There are a good number of outliers in these numeral variables.

### Fixing Outliers
"""

df['RoomService'].quantile(0.90)

# IQR Method (Interquartile Range)

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['RoomService'].quantile(0.25)
Q3 = df['RoomService'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['RoomService'] < lower_bound) | (df['RoomService'] > upper_bound)]
outliers['RoomService']

# Remove rows with outliers
df_cleaned = df[(df['RoomService'] >= lower_bound) & (df['RoomService'] <= upper_bound)]
df_cleaned.shape

"""## 5.4. **Univariate Analysis**

### Numerical Variables
"""

# Plot histogram for individual numerical columns
sns.histplot(df['RoomService'], kde=True, bins=10, color="green")
plt.title(f'Histogram of RoomService')
plt.show()

# Plot multiple histograms together
df[numerical_columns].hist(bins=20, figsize=(10, 10), layout=(3, 2), color="blue", edgecolor="black")
plt.suptitle('Histograms of Numerical Variables')
plt.tight_layout()
plt.show()

from scipy import stats

def normality_test(df, column, alpha=0.05):

    data = df[column]
    # Perform the Shapiro-Wilk test for normality
    stat, p_value = stats.shapiro(data)

    # Check if p-value is less than alpha (significance level)
    if p_value > alpha:
        return f"Data of {column} column is normally distributed (p-value = {p_value:.4f})"
    else:
        return f"Data of {column} column is NOT normally distributed (p-value = {p_value:.4f})"

normality_test(df, numerical_columns[0])

"""### Categorical Variables"""

print(df[categorical_columns[2]].value_counts())

few_category_columns=['HomePlanet','CryoSleep','Destination','VIP']

# Set up the plotting area (adjust size as needed)
plt.figure(figsize=(12, 10))

# Plot each categorical column's value counts
for i, col in enumerate(few_category_columns):
    plt.subplot(3, 3, i+1)  # Adjust grid size (3x3 here)
    sns.countplot(x=col, data=df, palette='Set2')
    plt.title(f'Value Counts for {col}')
    plt.xticks(rotation=45)

# Tight layout for better spacing
plt.tight_layout()
plt.show()

"""## 5. 5. **Bivariate and Multivariate Analysis**

###  GroupBy
"""

# Group by 'HomePlanet' and calculate the mean of 'RoomService'
df.groupby('HomePlanet')['RoomService'].max()

# Group by 'CryoSleep' and calculate the sum of 'FoodCourt'
df.groupby('CryoSleep')['FoodCourt'].sum()

# Group by 'Destination' and 'VIP' and calculate the mean of 'Spa'
df.groupby(['Destination', 'VIP'])['Spa'].mean()

# combined Complex One

# Group by 'HomePlanet', 'CryoSleep', and 'VIP'
df.groupby(['HomePlanet', 'CryoSleep', 'VIP']).agg({
    'RoomService': ['mean', 'sum', 'count'],   # Mean, sum, and count of 'RoomService'
    'FoodCourt': ['mean', 'sum']      # Mean, sum, and count of 'FoodCourt'
}).reset_index()

"""### Scatterplots"""

# Plot scatter plot for two variables (e.g., 'Age' and 'RoomService')
sns.scatterplot(x='FoodCourt', y='RoomService', data=df)
plt.title('Scatter Plot: Age vs RoomService')
plt.xlabel('Age')
plt.ylabel('RoomService')
plt.show()

"""### Corelation and Heatmap"""

# Calculate the correlation matrix
corr_matrix = df[numerical_columns].corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Create a clustermap for correlation matrix
sns.clustermap(corr_matrix, annot=True, cmap='coolwarm', figsize=(10, 8))
plt.title('Clustermap: Correlation of Numerical Variables')
plt.show()

"""### Pairplots"""

# Plot pair plot for numerical columns
sns.pairplot(df[numerical_columns])
plt.suptitle('Pair Plot of Numerical Variables', y=1.02)
plt.show()

# KDE plot for two variables
sns.kdeplot(x='Age', y='RoomService', data=df, cmap='Blues', shade=True, fill=True)
plt.title('KDE Plot: Age vs RoomService')
plt.show()

# Box plot for numerical variable ('Age') across categories ('CryoSleep')
sns.boxplot(x='CryoSleep', y='Age', data=df)
plt.title('Box Plot: CryoSleep vs Age')
plt.show()

"""## 5.6. **Data Visualization**"""

# Bar plot for categorical data (e.g., distribution of 'HomePlanet')
sns.countplot(x='HomePlanet', data=df)
plt.title('Bar Plot: Distribution of HomePlanet')
plt.show()

# Line plot for numerical data (e.g., 'Age' over 'RoomService')
sns.lineplot(x='Age', y='RoomService', data=df)
plt.title('Line Plot: Age vs RoomService')
plt.show()

# Histogram for numerical variable (e.g., 'Age')
sns.histplot(df['Age'], kde=True)
plt.title('Histogram: Distribution of Age')
plt.show()

# FacetGrid for visualizing the distribution of 'Age' across different 'CryoSleep' categories
g = sns.FacetGrid(df, col='CryoSleep', height=5)
g.map(sns.histplot, 'Age', kde=True)
g.set_titles('CryoSleep = {col_name}')
plt.show()

# Joint plot for two numerical variables ('Age' vs 'RoomService')
sns.jointplot(x='Age', y='RoomService', data=df, kind='scatter', color='blue')
plt.suptitle('Joint Plot: Age vs RoomService', y=1.02)
plt.show()

# Ridgeline plot for distribution of 'Age' across different 'CryoSleep' categories
sns.violinplot(x='CryoSleep', y='Age', data=df, inner="stick")
plt.title('Ridgeline Plot: CryoSleep vs Age')
plt.show()

# Dot plot for categorical data ('CryoSleep')
sns.stripplot(x='HomePlanet', data=df, jitter=True, size=5, color='blue')
plt.title('Dot Plot: Distribution of CryoSleep')
plt.show()

# Stacked bar plot for categorical data
df.groupby(['CryoSleep', 'HomePlanet']).size().unstack().plot(kind='bar', stacked=True)
plt.title('Stacked Bar Plot: CryoSleep vs HomePlanet')
plt.show()

"""## 5.7. **Identifying Patterns and Trends**:"""

# Regression plot for two variables ('Age' vs 'RoomService')
sns.regplot(x='Age', y='RoomService', data=df, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
plt.title('Regression Plot: Age vs RoomService')
plt.show()

# Calculate the correlation matrix
corr_matrix = df[numerical_columns].corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

"""### Aggregating Data to Identify Patterns"""

# Grouping by 'CryoSleep' and calculating the mean of 'RoomService'
aggregated_data = df.groupby('CryoSleep')['RoomService'].mean().reset_index()
aggregated_data

# Box plot for 'RoomService' across 'CryoSleep'
sns.boxplot(x='CryoSleep', y='RoomService', data=df)
plt.title('RoomService Across CryoSleep Categories')
plt.show()

# Bar plot for 'RoomService' mean across 'CryoSleep'
sns.barplot(x='CryoSleep', y='RoomService', data=df, estimator='mean')
plt.title('Mean RoomService Across CryoSleep Categories')
plt.show()

from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Selecting numerical columns for clustering
X = df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]

# Impute missing values by the most frequent value (mode)
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)

# Applying KMeans clustering
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(X_imputed)

# Visualizing clusters with a scatter plot
plt.scatter(df['Age'], df['RoomService'], c=df['Cluster'], cmap='viridis')
plt.title('KMeans Clustering: Age vs RoomService')
plt.xlabel('Age')
plt.ylabel('RoomService')
plt.show()

"""## 5.8. **Hypothesis Testing**

### **Numerical Example: One-Sample t-Test**
We'll use **Age** as the variable and test whether the average age is significantly different from 30.

#### **Hypotheses**
- **H₀**: The mean age is 30. (μ = 30)
- **H₁**: The mean age is not 30. (μ ≠ 30)

f `p-value <= 0.05`, we reject the null hypothesis and conclude that the mean age is significantly different from 30.
"""

# Hypothetical Data: Age column
age_data = df['Age'].dropna()  # Drop any NaN values

# Perform a one-sample t-test
from scipy import stats

t_stat, p_value = stats.ttest_1samp(age_data, 30)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

"""### **Numerical Example: Two-Sample t-Test**
We'll compare the **RoomService** spending between two groups: people who have **CryoSleep** and people who don't.

**Hypotheses**
- **H₀**: The mean RoomService spending is the same for both CryoSleep and non-CryoSleep passengers.
- **H₁**: The mean RoomService spending is different for CryoSleep and non-CryoSleep passengers.


If `p-value <= 0.05`, we reject the null hypothesis and conclude that there is a significant difference in RoomService spending between the two groups.

"""

# Data: RoomService spending and CryoSleep status
cryo_sleep_yes = df[df['VIP'] == True]['RoomService'].dropna()
cryo_sleep_no = df[df['VIP'] == False]['RoomService'].dropna()

# Perform a two-sample t-test
t_stat, p_value = stats.ttest_ind(cryo_sleep_yes, cryo_sleep_no)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

"""### **Categorical Example: Chi-Square Test of Independence**
We'll test whether there is an association between **Destination** and **CryoSleep** status.

**Hypotheses**
- **H₀**: There is no association between Destination and CryoSleep.
- **H₁**: There is an association between Destination and CryoSleep.


If `p-value <= 0.05`, we reject the null hypothesis and conclude that there is an association between **Destination** and **CryoSleep** status.

"""

from scipy.stats import chi2_contingency

# Data: Destination and CryoSleep status
contingency_table = pd.crosstab(df['VIP'], df['CryoSleep'])

# Perform the Chi-Square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2_stat}")
print(f"P-value: {p_value}")

"""### **Categorical Example: Chi-Square Test for VIP and CryoSleep**
We'll test whether there is a relationship between **VIP status** and **CryoSleep** (assumed to be a column in the dataset).

**Hypotheses**
- **H₀**: There is no association between VIP status and CryoSleep.
- **H₁**: There is an association between VIP status and CryoSleep.

 If `p-value <= 0.05`, we reject the null hypothesis and conclude that there is an association between **VIP status** and **CryoSleep**.

"""

# Data: VIP status and Survival status (assuming Survival is a column in the dataset)
contingency_table = pd.crosstab(df['VIP'], df['CryoSleep'])

# Perform the Chi-Square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2_stat}")
print(f"P-value: {p_value}")

"""# 7. **Working with Dates**"""

date_df = pd.read_csv("/content/CIOL-Winter-ML-Bootcamp/datasets/session1/main/date/data_date.csv")
date_df.head()

# Convert 'Date' column to datetime
date_df['Date'] = pd.to_datetime(date_df['Date'])

date_df.dtypes

# Extracting year, month, and day from the 'Date' column
date_df['Year'] = date_df['Date'].dt.year
date_df['Month'] = date_df['Date'].dt.month
date_df['Day'] = date_df['Date'].dt.day

# Display the updated dataframe
date_df.head()

bangladesh_data = date_df[date_df['Country']=='Bangladesh']

# Plot the 'Value' column over time
plt.figure(figsize=(10, 6))
plt.plot(bangladesh_data['Date'], bangladesh_data['AQI Value'], marker='o', linestyle='-', color='b')
plt.title('Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

## Resampling and Aggregating by Date

# Set 'Date' as the index for easier resampling
bangladesh_data.set_index('Date', inplace=True)

# Resample by month and calculate the mean AQI for each month
monthly_aqi = bangladesh_data.resample('M')['AQI Value'].mean()

# Plot the monthly average AQI for Bangladesh
plt.figure(figsize=(10, 6))
monthly_aqi.plot(marker='o', linestyle='-', color='b')
plt.title("Monthly Average AQI for Bangladesh")
plt.xlabel("Month")
plt.ylabel("Average AQI Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# Resample by week and calculate the mean AQI for each week
weekly_aqi = bangladesh_data.resample('W')['AQI Value'].mean()

# Plot the weekly average AQI for Bangladesh
plt.figure(figsize=(10, 6))
weekly_aqi.plot(marker='o', linestyle='-', color='b')
plt.title("Weekly Average AQI for Bangladesh")
plt.xlabel("Week")
plt.ylabel("Average AQI Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

"""# Final report

500 words

---

Thank you!!

---

If you use it, cite:

*Azmine Toushik Wasi. (2024). CIOL Presnts Winer ML BootCamp. https://github.com/ciol-researchlab/CIOL-Winter-ML-Bootcamp*

```
@misc{wasi2024CIOL-WMLB,
      title={CIOL Presnts Winer ML BootCamp},
      author={Azmine Toushik Wasi},
      year={2024},
      url={https://github.com/ciol-researchlab/CIOL-Winter-ML-Bootcamp},
}```
"""