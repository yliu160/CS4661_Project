import pandas as pd
import numpy as np

# sklearn classifier's
from sklearn.neighbors import KNeighborsClassifier # knn algo
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model # LogisticRegression LinearRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans # K-means Clustering
from sklearn.naive_bayes import GaussianNB # naive bayes

# sklearn misc.
from sklearn.preprocessing import StandardScaler, LabelEncoder        # StandardScaler = normalize features             ——LabelEncoder =  Label Encoding (convert value into unique int) - good for ft. w/ many unique values (to not create many dummy columns)
from sklearn import metrics                                           # accuracy score
from sklearn.model_selection import cross_val_score, train_test_split # cross_val_score = average(repeatedly split dataset into training/testing, .fit(), accuracy_score(.predict())         ——train_test_split example:   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=x)
from sklearn.pipeline import make_pipeline                            #
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA # (principal component analysis (unsupervised dimensionality reduction))

# plotting
import matplotlib.pyplot as plt                                       # matplotlib. (graphs/plots)
import seaborn as sns                                                 # matplotlib alternative

# defaults
pd.set_option('display.max_columns', None) # don't limit # columns shown
