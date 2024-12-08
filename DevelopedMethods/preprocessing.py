# Data Exploration.ipynb —— all cells pasted, minus graphs & logging   --so just data preprocessing lines
from DF import * # import np, pd, sklearn fn's --and df (https://www.kaggle.com/datasets/kartik2112/fraud-detection/)

# make exploration easier, drop redundant columns:
df.drop(['Unnamed: 0', 'unix_time', 'trans_num'], axis = 1, inplace = True) # "Unnamed: 0" always == row index  &  "unix_time" isn't as readable as "trans_date_trans_time", "trans_num" is unique in every row (basically an id, so useless)

# manually encode binary/simple labels
# categorical features:
# df["trans_date_trans_time"] = df["trans_date_trans_time"].apply(lambda x: 1 if x == "2019-01-01 00:00:18" else 0)
# df["merchant"] = df["merchant"].apply(lambda x: 1 if x == "fraud_Rippin, Kub and Mann" else 0)
# df["category"] = df["category"].apply(lambda x: 1 if x == "misc_net" else 0)
# df["first"] = df["first"].apply(lambda x: 1 if x == "Jennifer" else 0)
# df["last"] = df["last"].apply(lambda x: 1 if x == "Banks" else 0)
df["gender"] = df["gender"].apply(lambda x: 1 if x == "M" else 0)
# df["street"] = df["street"].apply(lambda x: 1 if x == "561 Perry Cove" else 0)
# df["city"] = df["city"].apply(lambda x: 1 if x == "Moravian Falls" else 0)
# df["state"] = df["state"].apply(lambda x: 1 if x == "NC" else 0)
# df["job"] = df["job"].apply(lambda x: 1 if x == "Psychologist, counselling" else 0)
# df["dob"] = df["dob"].apply(lambda x: 1 if x == "1988-03-09" else 0)

# split trans_date_trans_time into y-m-d-h-m-s columns
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['year'] = df['trans_date_trans_time'].dt.year
df['month'] = df['trans_date_trans_time'].dt.month
df['day'] = df['trans_date_trans_time'].dt.day
df['hour'] = df['trans_date_trans_time'].dt.hour
df['minute'] = df['trans_date_trans_time'].dt.minute
df['second'] = df['trans_date_trans_time'].dt.second
df = df.drop(columns=['trans_date_trans_time'])

# one-hot encoding features w/ few values
df = pd.get_dummies(df, columns=['category', 'state'], drop_first=True) # 14, 51 unique values

# Label Encoding (convert value into unique int) - good for ft. w/ many unique values
label_enc = LabelEncoder()
df['merchant'] = label_enc.fit_transform(df['merchant']) # 693 unique values
df['first'] = label_enc.fit_transform(df['first']) # 355 unique values
df['last'] = label_enc.fit_transform(df['last']) # 486 unique values
df['street'] = label_enc.fit_transform(df['street']) # 999 unique values
df['city'] = label_enc.fit_transform(df['city']) # 906 unique values
df['job'] = label_enc.fit_transform(df['job']) # 497 unique values
df['dob'] = label_enc.fit_transform(df['dob']) # 984 unique values

df, df_train, df_test = resplit_train_test(df) # currently df is df_train + df_test,  tracked by the added 'source' feature (train or test)         --also removes the source column from df

X_train = df_train.drop(columns=['is_fraud'])
X_test = df_test.drop(columns=['is_fraud'])
y_train = df_train['is_fraud']
y_test = df_test['is_fraud']

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.2, random_state=42)  # Resample minority class to 20% of majority class
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
X_test_balanced, y_test_balanced = smote.fit_resample(X_test, y_test)

# this ran before # Feature Matrix (X) & Target/label Vector (y), but we cannot SMOTE (imbalance handling) non-discrete/continuous values (ie: scaling converts nearly everything into a decimal)
# so, we now have to apply to: X_train, X_test (opposed to df_train & df_test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test_balanced)
# Re-wrap the scaled data into a DataFrame with original column names and indices (scaler transforms into numpy.ndarray (NOT a dataframe))
X_train_balanced = pd.DataFrame(X_train_scaled, columns=X_train_balanced.columns, index=X_train_balanced.index)
X_test_balanced = pd.DataFrame(X_test_scaled, columns=X_test_balanced.columns, index=X_test_balanced.index)
