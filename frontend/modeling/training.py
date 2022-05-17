import time

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score, f1_score


# train test split function: gets imput from user selection of test and train size from the UI 
def my_train_test_split(train_size, test_size):
	data=pd.read_csv("/storage/preprocessed_water_potability.csv")
	train_x,test_x,train_y,test_y = model_selection.train_test_split(
		data.iloc[:,:-1], data.iloc[:, -1], test_size=test_size, train_size=train_size, random_state=42)
	
	return train_x,test_x,train_y,test_y	

def training(model, test_size,train_size):
	data = pd.read_csv("/storage/preprocessed_water_potability.csv")
	train_x,test_x,train_y,test_y = model_selection.train_test_split(
	data.iloc[:,:-1], data.iloc[:, -1], test_size=test_size, train_size=train_size, random_state=42)
	#model = RandomForestClassifier(random_state=42,bootstrap=True,criterion='gini',max_depth=5,min_samples_leaf=10)
	t0 = time.time()
	trained_model= model.fit(train_x,train_y)
	duration = time.time() - t0
	duration = np.round(duration,3)
	y_test_pred = trained_model.predict(test_x)

	test_accuracy = np.round(accuracy_score(test_y, y_test_pred), 3)
	test_f1 = np.round(f1_score(test_y, y_test_pred, average="weighted"), 3)
	acc = trained_model.score(test_x,test_y)

	return test_accuracy,test_f1, duration
