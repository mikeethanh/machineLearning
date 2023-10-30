import pandas as pd

df = pd.read_csv('pandass/pokemon_data.csv')
# ngoai ra con xlsx(excel),

# neu la filr txt : 
# df = pd.read_csv('pokemmon_data.txt',deliminter = '\t')

# 
print(df)
# 
print(df.head(3))
# 
print(df.tal(5))

# READING DATA FROM PANDAS

# Read Headers
df.columns
# Index(['#', 'Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk',
#       'Sp. Def', 'Speed', 'Generation', 'Legendary'],
#       dtype='object')

# read each column
print(df['Name'][0:5])
#in ten cua cac pokemon tu 1 den 5

# 
print(df[['Name', 'Type 1', 'HP']])
# in 3 cot name, type 1 , HP


#read each row 
print(df.head(5))
# print 5 hang dau tien 

print(df.iloc[1])
# in ra tat ca thong ti ncua pokemon dung thu 1

print(df.iloc[1:4])

print(df.iloc[2,1])
# in ra hang 2 cot 1

for index, row in df.iterrows():
    print(index , row['Name'])
    # chuyen nganh thanh doc 

# giong kieu so sanh
df.loc[df['Type 1'] == "Fire"]
# in ra tat ca thong tin co Type 1 == fire

# SORTING/DESCRiBiNG DATA

df.describe()

# sort by name (theo thu tu giam dan , mac dinh la tang dan)
df.sort_values('Name',ascending = False )

# sap xem theo Type 1 xong r den HP, Type theo tang dan , hp = giam dan 
df.sort_values(['Type 1', 'HP'],ascending=[1,0])


# Making changes to the data

df['Total'] = df['HP'] + df['DDEfense']+ df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
# tao ra 1 cot vos header = total

# xoa 1 cot nao do 
df = df.drop(columns=['Total'])

# cach khac de tao ra 1 cot 
df['Total'] = df.iloc[:, 4:10].sum(axis=1)

# do cho cac cot 
cols = list(df.columns)
df = df[cols[0:4] + [cols[-1]]+cols[4:12]]

# SAVNG OUR DATA 
df.to_csv('modified.csv',index = False)
# xoa bo index o dau

# neu la file duoi xlsx
df.to_excel('modified.xlsx',index = False)

# neu la file dduoi txt 
df.to_csv('modified.txt', index=False, sep='\t')

# FILTERING DATA
new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type2'] == 'Poion')& (df['HP'] > 70)]
print(new_df)

# reset index 
new_df = new_df.reset_index(drop=True,inplace=True)

# save fle 
new_df.to_csv('filtred.csv')

# loc nhung pokemon ma ten cua chong co chua mega
# neu muon reverse : ~
df.loc[~df['Name'].str.contains('Mega')]

import re 
df.loc[df['Type 1'].str.contains('Fire|Grass',regex = True)]
# in ra cac pokemond ma type co chua fire OR grass


#   CONDITIONAL CHANGES

# neu total > 50 thi gen va legend = test 1  ,test 2
df.loc[df['Total'] > 500, ['Generation','Legendary']] = ['Test 1', 'Test 2']


# Aggregate Statistics (Groupby)

df.groupby(['Type 1']).mean().sort_values('HP',ascending=False)
# thong ke cac type 1 

df.groupby(['Type 1']).sum()

df.groupby(['Type 1']).count()




