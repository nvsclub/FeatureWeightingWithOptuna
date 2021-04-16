# Imports
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

# Regression
## Abalone (from UCI)
abalone_data = pd.read_csv('data/raw/abalone.csv')
abalone_data.to_csv('data/standardized/r_abalone.csv', index=False)

## Bike sharing (from UCI)
bike_data = pd.read_csv('data/raw/bikesharing-day.csv').drop(['instant','dteday','casual','registered'], axis = 1)
bike_data.to_csv('data/standardized/r_bikesharing.csv', index=False)

## Boston housing (from pandas)
boston_data = datasets.load_boston()
df_boston = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
df_boston['target'] = pd.Series(boston_data.target)
df_boston.to_csv('data/standardized/r_boston.csv', index=False)

## Diabetes (from pandas)
diabetes_data = datasets.load_diabetes()
df_diabetes = pd.DataFrame(diabetes_data.data,columns=diabetes_data.feature_names)
df_diabetes['target'] = pd.Series(diabetes_data.target)
df_diabetes.to_csv('data/standardized/r_diabetes.csv', index=False)

## Forest fires (from UCI)
fire_data = pd.read_csv('data/raw/forestfires.csv')
d = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
fire_data.month = fire_data.month.map(d)
d = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
fire_data.day = fire_data.day.map(d)
fire_data.target = np.log(fire_data.target+1) # log normalization recomended by the author
fire_data.to_csv('data/standardized/r_forestfires.csv', index=False)

## Machine (from UCI)
machine_data = pd.read_csv('data/raw/machine.csv')
machine_data.to_csv('data/standardized/r_machine.csv', index=False)

## Student performance (from UCI)
student_data = pd.read_csv('data/raw/student-mat.csv', sep = ';')
student_data.school = (student_data.school == 'GP') * 1
student_data.sex = (student_data.sex == 'F') * 1
student_data.address = (student_data.address == 'U') * 1
student_data.famsize = (student_data.famsize == 'LE3') * 1
student_data.Pstatus = (student_data.Pstatus == 'T') * 1
student_data.Mjob = (student_data.Mjob == 'teacher') * 1
student_data.Fjob = (student_data.Fjob == 'teacher') * 1
student_data.reason = (student_data.reason == 'home') * 1
student_data.schoolsup = (student_data.schoolsup == 'yes') * 1
student_data.famsup = (student_data.famsup == 'yes') * 1
student_data.paid = (student_data.paid == 'yes') * 1
student_data.activities = (student_data.activities == 'yes') * 1
student_data.nursery = (student_data.nursery == 'yes') * 1
student_data.higher = (student_data.higher == 'yes') * 1
student_data.internet = (student_data.reason == 'yes') * 1
student_data.romantic = (student_data.romantic == 'yes') * 1
student_data.guardian = (student_data.guardian == 'mother') * 1
student_data = student_data.drop(['G1','G2'], axis = 1)
student_data.to_csv('data/standardized/r_student.csv', index=False)

# Classification
## Australian (from UCI - https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval))
australian_data = pd.read_csv('data/raw/australian.csv')
australian_data.to_csv('data/standardized/c_australian.csv', index=False)

## Balance-scale (from UCI - https://archive.ics.uci.edu/ml/datasets/balance+scale)
df = pd.read_csv('data/raw/balance-scale.csv')
df.to_csv('data/standardized/c_balance-scale.csv', index=False)

## Breast cancer (from pandas)
breast_cancer_data = datasets.load_breast_cancer()
df_breast_cancer = pd.DataFrame(breast_cancer_data.data,columns=breast_cancer_data.feature_names)
df_breast_cancer['target'] = pd.Series(breast_cancer_data.target)
df_breast_cancer.to_csv('data/standardized/c_breastcancer.csv', index=False)

## Breast tissue (from UCI - http://archive.ics.uci.edu/ml/datasets/breast+tissue)
df = pd.read_csv('data/raw/breast-tissue.csv')
df.to_csv('data/standardized/c_breast-tissue.csv', index=False)

## PI Diabetes (from Kaggle - https://www.kaggle.com/uciml/pima-indians-diabetes-database)
df = pd.read_csv('data/raw/pi-diabetes.csv')
df.to_csv('data/standardized/c_pi-diabetes.csv', index=False)

## Ecoli (from UCI - https://archive.ics.uci.edu/ml/datasets/ecoli)
ecoli_data = pd.read_csv('data/raw/ecoli.csv').drop('a1', axis = 1)
ecoli_data.to_csv('data/standardized/c_ecoli.csv', index=False)

## Glass (from UCI - https://archive.ics.uci.edu/ml/datasets/glass+identification)
glass_data = pd.read_csv('data/raw/glass.csv').drop('a1', axis = 1)
glass_data.to_csv('data/standardized/c_glass.csv', index=False)

## Haberman (from UCI - https://archive.ics.uci.edu/ml/datasets/haberman's+survival)
df = pd.read_csv('data/raw/haberman.csv').drop('a1', axis = 1)
df.to_csv('data/standardized/c_haberman.csv', index=False)

## Statlog heart (from UCI - http://archive.ics.uci.edu/ml/datasets/statlog+(heart))
df = pd.read_csv('data/raw/heart.csv').drop('a1', axis = 1)
df.to_csv('data/standardized/c_heart.csv', index=False)

## Ionosphere (from UCI - https://archive.ics.uci.edu/ml/datasets/ionosphere)
ionosphere_data = pd.read_csv('data/raw/ionosphere.csv')
ionosphere_data.to_csv('data/standardized/c_ionosphere.csv', index=False)

## Iris (from pandas)
iris_data = datasets.load_iris()
df_iris = pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
df_iris['target'] = pd.Series(iris_data.target)
df_iris.to_csv('data/standardized/c_iris.csv', index=False)

## Liver disorder (from UCI - https://archive.ics.uci.edu/ml/datasets/liver+disorders)
df = pd.read_csv('data/raw/liver-disorder.csv').drop('disc1', axis = 1)
df.to_csv('data/standardized/c_liver-disorder.csv', index=False)

## Pendigits (from UCI - https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)
df = pd.read_csv('data/raw/pendigits.csv')
df.to_csv('data/standardized/c_pendigits.csv', index=False)

## Post-operative (from UCI - https://archive.ics.uci.edu/ml/datasets/Post-Operative+Patient)
df = pd.read_csv('data/raw/post-operative.csv')
df.to_csv('data/standardized/c_post-operative.csv', index=False)

## Sonar (from UCI - https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))
sonar_data = pd.read_csv('data/raw/sonar.csv')
sonar_data.to_csv('data/standardized/c_sonar.csv', index=False)

## Transfusion (from UCI - https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center)
df = pd.read_csv('data/raw/transfusion.csv')
df.to_csv('data/standardized/c_transfusion.csv', index=False)

## Wine (from pandas)
wine_data = datasets.load_wine()
df_wine = pd.DataFrame(wine_data.data,columns=wine_data.feature_names)
df_wine['target'] = pd.Series(wine_data.target)
df_wine.to_csv('data/standardized/c_wine.csv', index=False)

## Vote (from UCI - https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)
df = pd.read_csv('data/raw/votes.csv')
df.to_csv('data/standardized/c_votes.csv', index=False)

## Vote (from UCI - https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)
df = pd.read_csv('data/raw/vehicle.csv')
df.to_csv('data/standardized/c_vehicle.csv', index=False)

## Yeast (from UCI - http://archive.ics.uci.edu/ml/datasets/yeast)
df = pd.read_csv('data/raw/yeast.csv').drop('disc1', axis = 1)
df.to_csv('data/standardized/c_yeast.csv', index=False)

## Car (from UCI - https://archive.ics.uci.edu/ml/datasets/car+evaluation) test for one hot encoded variables
car_data = pd.read_csv('data/raw/car.csv')
car_features = car_data.drop('target', axis = 1)
enc = OneHotEncoder()
enc.fit(car_features)
enc_car_data = pd.DataFrame(enc.transform(car_features).toarray(), columns = list(enc.get_feature_names()))
enc_car_data['target'] = car_data.target
enc_car_data.to_csv('data/standardized/c_car.csv', index=False)

## Zoo (from UCI - http://archive.ics.uci.edu/ml/datasets/Zoo)
df = pd.read_csv('data/raw/yeast.csv').drop('disc1', axis = 1)
df.to_csv('data/standardized/c_yeast.csv', index=False)