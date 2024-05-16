import pandas as pd
import matplotlib.pyplot as plt

#Adult Income dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
adult_df = pd.read_csv(data_url, names=column_names)

# Counting the number of individuals by gender
gender_counts = adult_df['sex'].value_counts()

# Plotting
plt.figure(figsize=(8, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
plt.title('Distribution of Genders in the Adult Income Dataset')
plt.axis('equal')  
plt.tight_layout()
plt.show()
