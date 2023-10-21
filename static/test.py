import pandas as pd
df = pd.read_excel('lipshades.xlsx')
print(df.head())
df['hexcode'] = df['hexcode'].map(lambda x: str(x))
df['wordsearch'] = df['name'] + df['color'] + df['hexcode']
print(df['wordsearch'])