import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the training data
train_df = pd.read_csv('./data/train.csv')

# Data Overview
print(train_df.head())
print(train_df.info())

# Distribution of Sale Prices
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('Distribution of Sale Prices')
plt.show()

# Correlation matrix
corr_matrix = train_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Scatter plot of SalePrice vs. GrLivArea (Above grade living area)
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train_df)
plt.title('Sale Price vs. Above Grade Living Area')
plt.show()