from imports import * # import np, pd, sklearn fn's

# df_train = pd.read_csv('../data/fraud-detection/fraudTrain.csv') # <--- relative path only works if u ran: /opt/anaconda3/bin/jupyter_mac.command  @  project root
# df_test = pd.read_csv('../data/fraud-detection/fraudTest.csv') # <--- relative path only works if u ran: /opt/anaconda3/bin/jupyter_mac.command  @  project root
df_train = pd.read_csv('~/Desktop/CS4661_Project/data/fraud-detection/fraudTrain.csv') # hardcoded. can also use kaggle notebook path: /kaggle/input/fraud-detection/fraudTrain.csv (if using online kaggle notebook)
df_test = pd.read_csv('~/Desktop/CS4661_Project/data/fraud-detection/fraudTest.csv') # hardcoded. can also use kaggle notebook path: /kaggle/input/fraud-detection/fraudTrain.csv (if using online kaggle notebook)

# track indices of each dataset so we can later separate them back into df_train, df_test after modifying (encoding, feature drops, etc.)
df_train['source'] = 'train'
df_test['source'] = 'test'

# make df entire dataset (train+test)
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)  # axis=0 means row-wise concatenation

def resplit_train_test(df): # df -> train_df & test_df with original indices
    df_train = df[df['source'] == 'train'].drop(columns='source').reset_index(drop=True)
    df_test = df[df['source'] == 'test'].drop(columns='source').reset_index(drop=True)
    df.drop(['source'], axis=1, inplace=True)
    return df, df_train, df_test

# df attributes:
# ['T', 'abs', 'add', 'add_prefix', 'add_suffix', 'agg', 'aggregate', 'align', 'all',
# 'amt', 'any', 'apply', 'applymap', 'asfreq', 'asof', 'assign', 'astype', 'at', 'at_time',
# 'attrs', 'axes', 'backfill', 'between_time', 'bfill', 'bool', 'boxplot', 'category', 'cc_num',
# 'city', 'city_pop', 'clip', 'columns', 'combine', 'combine_first', 'compare', 'convert_dtypes',
# 'copy', 'corr', 'corrwith', 'count', 'cov', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe',
# 'diff', 'div', 'divide', 'dob', 'dot', 'drop', 'drop_duplicates', 'droplevel', 'dropna', 'dtypes',
# 'duplicated', 'empty', 'eq', 'equals', 'eval', 'ewm', 'expanding', 'explode', 'ffill', 'fillna',
# 'filter', 'first', 'first_valid_index', 'flags', 'floordiv', 'from_dict', 'from_records', 'ge',
# 'gender', 'get', 'groupby', 'gt', 'head', 'hist', 'iat', 'idxmax', 'idxmin', 'iloc', 'index',
# 'infer_objects', 'info', 'insert', 'interpolate', 'is_fraud', 'isetitem', 'isin', 'isna',
# 'isnull', 'items', 'iterrows', 'itertuples', 'job', 'join', 'keys', 'kurt', 'kurtosis',
# 'last', 'last_valid_index', 'lat', 'le', 'loc', 'long', 'lt', 'map', 'mask', 'max', 'mean',
# 'median', 'melt', 'memory_usage', 'merch_lat', 'merch_long', 'merchant', 'merge', 'min', 'mod',
# 'mode', 'mul', 'multiply', 'ndim', 'ne', 'nlargest', 'notna', 'notnull', 'nsmallest', 'nunique',
# 'pad', 'pct_change', 'pipe', 'pivot', 'pivot_table', 'plot', 'pop', 'pow', 'prod', 'product',
# 'quantile', 'query', 'radd', 'rank', 'rdiv', 'reindex', 'reindex_like', 'rename', 'rename_axis',
# 'reorder_levels', 'replace', 'resample', 'reset_index', 'rfloordiv', 'rmod', 'rmul', 'rolling',
# 'round', 'rpow', 'rsub', 'rtruediv', 'sample', 'select_dtypes', 'sem', 'set_axis', 'set_flags',
# 'set_index', 'shape', 'shift', 'size', 'skew', 'sort_index', 'sort_values', 'squeeze', 'stack',
# 'state', 'std', 'street', 'style', 'sub', 'subtract', 'sum', 'swapaxes', 'swaplevel', 'tail',
# 'take', 'to_clipboard', 'to_csv', 'to_dict', 'to_excel', 'to_feather', 'to_gbq', 'to_hdf',
# 'to_html', 'to_json', 'to_latex', 'to_markdown', 'to_numpy', 'to_orc', 'to_parquet', 'to_period',
# 'to_pickle', 'to_records', 'to_sql', 'to_stata', 'to_string', 'to_timestamp', 'to_xarray', 'to_xml',
# 'trans_date_trans_time', 'trans_num', 'transform', 'transpose', 'truediv', 'truncate', 'tz_convert',
# 'tz_localize', 'unix_time', 'unstack', 'update', 'value_counts', 'values', 'var', 'where', 'xs', 'zip']
