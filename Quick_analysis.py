import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('train.csv')

# Statistical exploration
print("Dataset Info:")
df.info()
print("\nStatistical Summary:")
print(df.describe())
print("\nSurvival Distribution:")
print(df['Survived'].value_counts())

# Key visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Survival by gender
df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=ax1, color=['lightcoral', 'lightblue'])
ax1.set_title('Survival Rate by Gender')

# Correlation heatmap
sns.heatmap(df[['Survived', 'Pclass', 'Age', 'Fare']].corr(), annot=True, ax=ax2)
ax2.set_title('Correlation Matrix')

# Age distribution
df['Age'].hist(bins=20, ax=ax3, alpha=0.7)
ax3.set_title('Age Distribution')

# Pairplot data prep
pairplot_data = df[['Survived', 'Age', 'Fare']].dropna()
ax4.scatter(pairplot_data['Age'], pairplot_data['Fare'], 
           c=pairplot_data['Survived'], alpha=0.6)
ax4.set_xlabel('Age')
ax4.set_ylabel('Fare')
ax4.set_title('Age vs Fare by Survival')

plt.tight_layout()
plt.savefig('analysis_results.png')
plt.show()

# Key findings
print(f"\nKey Findings:")
print(f"Female survival: {df[df['Sex']=='female']['Survived'].mean():.1%}")
print(f"Male survival: {df[df['Sex']=='male']['Survived'].mean():.1%}")
print(f"1st class survival: {df[df['Pclass']==1]['Survived'].mean():.1%}")
print(f"3rd class survival: {df[df['Pclass']==3]['Survived'].mean():.1%}")