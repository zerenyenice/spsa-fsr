from Safs import Safs
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

m = KNeighborsClassifier(n_neighbors=1)
d = pd.read_csv('sonar.csv', header=None)
y = d.iloc[:, [60]].values.ravel()
x = d.drop([60], axis=1).values

engine = Safs(x=x, y=y, model=m)
rs = engine.run().results
rs.keys() #result list
rs.get('features')
