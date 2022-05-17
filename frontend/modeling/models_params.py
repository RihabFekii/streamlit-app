import streamlit as st 
from sklearn.ensemble import RandomForestClassifier

# RandomForest prams selector function 
def rf_param_selector():

	criterion = st.selectbox("criterion", ["gini", "entropy"])
	n_estimators = st.selectbox("n_estimators", options=[5, 10, 50, 100])
	max_depth = st.selectbox("max_depth", [5, 10, 50])
	min_samples_split = st.selectbox("min_samples_split", [10, 20, 5])

	params = {
		"random_state":42,
		"bootstrap":True,
		"criterion": criterion,
		"n_estimators": n_estimators,
		"max_depth": max_depth,
		"min_samples_split": min_samples_split,
		"n_jobs": -1,
	}

	model = RandomForestClassifier(**params)
	return model