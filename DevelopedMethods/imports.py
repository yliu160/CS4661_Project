import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier # knn algo
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split # data splitter helper
from sklearn.preprocessing import StandardScaler # normalize features
from sklearn import linear_model # LogisticRegression LinearRegression
from sklearn import metrics # accuracy score
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
# %matplotlib inline

# defaults
pd.set_option('display.max_columns', None) # don't limit # columns shown
