import pandas as pd


df = pd.read_csv('data/know.csv', nrows=10)

#print(df.head())
#print(df.describe())
#print(df.shape)
#print(df.isnull().sum())

for i,row in df.iterrows():
    
    with open(f"data/{i}.txt", 'w') as f:
        f.write(row['primary_question'] + "\n" + row["answer"])
        f.write("\n")
