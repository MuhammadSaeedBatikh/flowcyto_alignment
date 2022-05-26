from pipeline import Pipeline
from helpers import load_data
import numpy as np
path = '../../data/080322-ENU-Mouse-Normalisation-Data-Gated.csv'
X,Y, columns_names = load_data(path)
print(np.unique(Y))
pipeline = Pipeline(X, Y, 15, channel_names=columns_names[1:])
