import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_users = 500
data = {
    'user_id': range(1, num_users + 1),
    'subscription_length': np.random.randint(1, 13, size=num_users), #months
    'avg_daily_usage': np.random.normal(loc=2, scale=1, size=num_users),
    'num_logins': np.random.poisson(lam=10, size=num_users),
    'churned': np.random.binomial(1, 0.2, size=num_users) # 20% churn rate
}
df = pd.DataFrame(data)
# Add some more realistic features
df['customer_segment'] = np.random.choice(['A','B','C'], size=num_users, p=[0.3,0.5,0.2])
df['age'] = np.random.randint(18, 65, size=num_users)
df['country'] = np.random.choice(['USA','Canada','UK'], size=num_users, p=[0.6,0.25,0.15])
# --- 2. Feature Importance Analysis ---
# Using Chi-squared test for categorical features and correlation for numerical features.
# This is a simplified example; more sophisticated methods could be used.
# Contingency table for churn vs. customer segment
contingency_table = pd.crosstab(df['churned'], df['customer_segment'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-squared test for churn vs. customer segment: chi2={chi2:.2f}, p={p:.3f}")
# Correlation between churn and numerical features
correlation = df[['churned', 'subscription_length', 'avg_daily_usage', 'num_logins', 'age']].corr()
print("\nCorrelation Matrix:")
print(correlation)
# --- 3. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features with Churn')
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")
plt.figure(figsize=(8,6))
sns.countplot(x='customer_segment', hue='churned', data=df)
plt.title('Churn Rate by Customer Segment')
plt.savefig('churn_by_segment.png')
print("Plot saved to churn_by_segment.png")
# --- 4. (Simplified) Sequential Pattern Mining (Illustrative) ---
#  This section would typically involve more advanced algorithms like GSP or PrefixSpan.
#  Here, we'll just show a simplified example using pandas' groupby.
#  This part requires further development for a real-world application.
# Group by user and sort by date (assuming we had a date column)
# This is a placeholder; a real implementation would need time series data.
#  df = df.sort_values(by=['user_id', 'Date']) #needs a date column
# Example:  Identifying users who had low usage before churning (simplified)
# This is a highly simplified example and needs a proper time series analysis for a real application.
# df['low_usage_before_churn'] = (df.groupby('user_id')['avg_daily_usage'].transform(lambda x: x.iloc[-2] < 1 if len(x) >=2 else False) & df['churned'])
# print(df.head())
#Further analysis and modeling (e.g., Logistic Regression, Random Forest) would follow here, using the identified features.