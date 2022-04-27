import pandas as pd 

df = pd.read_csv('../data/creditcard.csv')
normal_df = df[df.Class == 0].sample(frac=0.05)
fraud_df = df[df.Class == 1].sample(frac=1)
sample_df = pd.concat([normal_df, fraud_df], axis=0)

sample_df.to_csv('../data/test_sample.csv', index=False)