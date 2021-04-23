import pandas as pd

def func(df):
    test_df = df.iloc[:2]
    test_df[1] *= 10

    return 1


df = pd.DataFrame([[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4]])

func(df.sample(frac=1))

print(df)