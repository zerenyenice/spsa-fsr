import Safs
from sklearn.svm import LinearSVC
import pandas as pd

d = pd.read_csv('test/sonar.csv', header=None)
y = d.iloc[:, [60]].values.ravel()
x = d.drop([60], axis=1).values

engine = Safs.Safs(x=x, y=y, model=LinearSVC())
rs = engine.run(num_features_selected=10, num_cores=1).results
rs.keys() #result list
rs.get('features')
