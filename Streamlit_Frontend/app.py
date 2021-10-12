import streamlit as st 
from streamlit import caching
import pandas as pd 
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import logging
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title ="AI service App",initial_sidebar_state="expanded", layout="wide", page_icon="💦")

#@st.cache(persist=False,allow_output_mutation=True,suppress_st_warning=True,show_spinner= True)
  
# change the color of button widget
Color = st.get_option("theme.secondaryBackgroundColor")
s = f"""
<style>
div.stButton > button:first-child {{background-color: #fffff ; border: 2px solid {Color}; border-radius:5px 5px 5px 5px; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)



# Read CSV -> outputs Pandas Dataframe
def load_csv(ds):
	df_input = pd.DataFrame()
	df_input=pd.read_csv(ds)  
	return df_input

# CSV File uploader
def file_upload():
	FILE_TYPES = ["CSV"]
	#FILE_TYPES = ["CSV", "py", "png", "jpg"]
	file = st.file_uploader("Upload file", type=FILE_TYPES)
	show_file = st.empty()
	if not file:
		show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))

	return file

# Dataframe columns types 
def column_types(df):
	with st.spinner('Loading data..'):
		col=df.dtypes
	return col 

# Bar plot with Altair 
def altair_plot(chart_type: str, df):
	""" return altair plots """
	fig = None

	if chart_type == "Bar":
		fig = (
		alt.Chart(df).mark_bar().encode(
		x='Potability:O',
		y= 'count(Potability):O',
		color='Potability:N'
		).interactive()
		)
	
	return fig

# KDE Plot with Seaborn and Matplotlib 
def sns_plot(chart_type: str, df):
	if chart_type=="Kernel Density Estimate":
		non_potabale = df.query("Potability == 0")
		potabale     = df.query("Potability == 1")
		fig , ax = plt.subplots(3,3)
		#fig.suptitle("Kernel Density Estimate")
		sns.kdeplot(ax=ax[0,0], x=non_potabale["Chloramines"],label='Non Potabale')
		sns.kdeplot(ax=ax[0,0], x=potabale["Chloramines"],label='Potabale')
		sns.kdeplot(ax=ax[0,1], x=non_potabale["Hardness"],label='Non Potabale')
		sns.kdeplot(ax=ax[0,1], x=potabale["Hardness"],label='Potabale')
		sns.kdeplot(ax=ax[0,2], x=non_potabale["Solids"],label='Non Potabale')
		sns.kdeplot(ax=ax[0,2], x=potabale["Solids"],label='Potabale')
		#####
		sns.kdeplot(ax=ax[1,0], x=non_potabale["ph"],label='Non Potabale')
		sns.kdeplot(ax=ax[1,0], x=potabale["ph"],label='Potabale')
		sns.kdeplot(ax=ax[1,1], x=non_potabale["Sulfate"],label='Non Potabale')
		sns.kdeplot(ax=ax[1,1], x=potabale["Sulfate"],label='Potabale')
		sns.kdeplot(ax=ax[1,2], x=non_potabale["Conductivity"],label='Non Potabale')
		sns.kdeplot(ax=ax[1,2], x=potabale["Conductivity"],label='Potabale')
		######
		sns.kdeplot(ax=ax[2,0], x=non_potabale["Organic_carbon"],label='Non Potabale')
		sns.kdeplot(ax=ax[2,0], x=potabale["Organic_carbon"],label='Potabale')
		sns.kdeplot(ax=ax[2,1], x=non_potabale["Trihalomethanes"],label='Non Potabale')
		sns.kdeplot(ax=ax[2,1], x=potabale["Trihalomethanes"],label='Potabale')
		sns.kdeplot(ax=ax[2,2], x=non_potabale["Turbidity"],label='Non Potabale')
		sns.kdeplot(ax=ax[2,2], x=potabale["Turbidity"],label='Potabale')
		plt.tight_layout(pad=1)
		plt.legend(loc="upper left")
		st.pyplot(fig)
	return


# create plots
def show_plot(df,chart_type):
	if chart_type != "Kernel Density Estimate":
		plot = altair_plot(chart_type, df)
		st.altair_chart(plot, use_container_width=True)
	if chart_type == "Kernel Density Estimate":
		sns_plot(chart_type, df)

# Plot of heatmap for missing values of a dataframe 
def plot(df):
			fig= plt.figure()
			plt.title('Missing Values Per Feature')
			nans = df.isna().sum().sort_values(ascending=False).to_frame()
			sns.heatmap(nans,annot=True,vmin=None,fmt='d',cmap='Blues')
			st.pyplot(fig)
			return(fig)

# Plot dataset after handling missing values -> replaces zeros with the mean of the values in that column coreesponding to the same target class. 
def deal_with_missing_values(df):
	for col in df.columns:	
		missing_label_0 = df.query('Potability == 0')[col][df[col].isna()].index
		df.loc[missing_label_0,col] = df.query('Potability == 0')[col][df[col].notna()].mean()

		missing_label_1 = df.query('Potability == 1')[col][df[col].isna()].index
		df.loc[missing_label_1,col] = df.query('Potability == 1')[col][df[col].notna()].mean()
	return df

# Plot dataset after handling missing values
def plot_notmissing(df):
	df = deal_with_missing_values(df)
	fig = plot(df)
	return(fig)

def save_dataset(df,file):
	new_df = deal_with_missing_values(df)
	# save dataset after preprocessing for training 
	name = file.name
	new_df.to_csv("/storage/preprocessed_" + name )
	return df

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

# train test split function: gets imput from user selection of test and train size from the UI 
def my_train_test_split(train_size, test_size):
	data=pd.read_csv("/storage/preprocessed_water_potability.csv")
	X=data.iloc[:,:-1]
	y=data.iloc[:, -1]
	train_x,test_x,train_y,test_y = model_selection.train_test_split(
		data.iloc[:,:-1], data.iloc[:, -1], test_size=test_size, train_size=train_size, random_state=42)
	return (train_x,test_x,train_y,test_y)	

def training(model, test_size,train_size):
	data=pd.read_csv("/storage/preprocessed_water_potability.csv")
	train_x,test_x,train_y,test_y = model_selection.train_test_split(
	data.iloc[:,:-1], data.iloc[:, -1], test_size=test_size, train_size=train_size, random_state=42)
	#model = RandomForestClassifier(random_state=42,bootstrap=True,criterion='gini',max_depth=5,min_samples_leaf=10)
	t0 = time.time()
	trained_model= model.fit(train_x,train_y)
	duration = time.time() - t0
	duration=np.round(duration,3)
	y_test_pred= trained_model.predict(test_x)

	test_accuracy = np.round(accuracy_score(test_y, y_test_pred), 3)
	test_f1 = np.round(f1_score(test_y, y_test_pred, average="weighted"), 3)
	acc=trained_model.score(test_x,test_y)
	return test_accuracy,test_f1, duration

# train model and calculate metrics
def train_model(model, train_x,test_x,train_y,test_y):
    t0 = time.time()
	#st.write(type(train_x))
	#st.write(type(train_y))
    trained_model=model.fit(train_x, train_y)
    duration = time.time() - t0
    y_train_pred = trained_model.predict(train_x)
    y_test_pred = trained_model.predict(test_x)

    train_accuracy = np.round(accuracy_score(train_y, y_train_pred), 3)
    train_f1 = np.round(f1_score(train_y, y_train_pred, average="weighted"), 3)

    test_accuracy = np.round(accuracy_score(test_y, y_test_pred), 3)
    test_f1 = np.round(f1_score(test_y, y_test_pred, average="weighted"), 3)

    return trained_model, train_accuracy, train_f1, test_accuracy, test_f1, duration

# User input: ID for the GET request to the Context Broker (to get the entities)
def user_input(default_id):
	id = st.text_input("User input", default_id)
	return id 

# GET attributes from the Context broker corresponding to id
def get_attributes(url):

	#url="http://backend.docker:8000/get_entities/" + id

	headers = {
	'Link': '<http://context/water-ngsi.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
	'Accept': 'application/ld+json'
	}

	entities = requests.request("GET",url,headers=headers)

	st.write(entities.json())
	att = entities.json()  
	return att

#notify context broker of Prediction attribute (Potability) update 
def notify_pred(id: str , resp:str):
	#url = "http://backend.docker:8000/patch_prediction/" + id
	url_notification= "http://backend.docker:8000/prediction/" + id + "/" + resp

	response = requests.request("PATCH", url_notification)

	return(response.text)

# Extracts attributes from the GET entities request payload, which will be injected to the ML model 
def extract_attributes(att):	
	result = dict(Ph=att['Ph'],Hardness=att['Hardness'], Solids = att['Solids'],Chloramines = att['Chloramines'], 
	Sulfate = att['Sulfate'], Conductivity = att['Conductivity'], Organic_carbon = att['Organic_carbon'],
	Trihalomethanes = att['Trihalomethanes'], Turbidity = att['Turbidity'] )
	return result 

# ML model prediction function using the prediction API request
def predict(result): 
	 
	header = {'Content-Type': 'application/json'}
	url3= "http://backend.docker:8000/prediction"
	#url= "http://backend_aliases:8000/prediction"

	payload=json.dumps(result)
	
	response = requests.request("POST", url3, headers=header, data=payload)
	response = response.text

	return response


def main():
	
	st.title("AI service for water quality assessment powered by FIWARE ")
	st.header("Water potability prediction 💦 ")
	st.write("This application enables to classify the potability of water based on the water composition and water quality metrics")
	
	#caching.clear_cache()

	activities = ["About this AI application","Data upload and visualisation","Data preprocessing","Modeling", "Prediction"]
	st.sidebar.title("Navigation")
	choices = st.sidebar.radio("",activities)

	sign=False
	test=False
	train_X=pd.DataFrame()
	test_X=pd.DataFrame()
	train_Y=pd.DataFrame()
	test_Y = pd.DataFrame()
	accuracy="Not yet defined before training"
	f1="Not yet defined before training"
	duration="..."
 #************************* Start About this AI application ***************************  
		
	if choices == 'About this AI application':

		st.image("/storage/img2.jpg")
		st.header("Context")
		st.write("Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level.")
		st.header("About this application")
		st.write("This application will explore the different features related to water potability, Modeling, and predicting water potability. It presents an in-depth analysis of what separates potable water from non-potable using statistics, bayesian inference, and other machine learning approaches.")
		#st.markdown("""The ML algorithm used is **[Sickit learn](https://facebook.github.io/prophet/)**.""")

#********************** End About this AI application *********************************

#********************** Start Data upload and visualisation ***************************  
		
	if choices == 'Data upload and visualisation':

		st.subheader('1. Data loading 🏋️')

		st.write("Import your water potability CSV dataset")
		with st.beta_expander("Water potability metrics dataset information"):
			st.write("This dataset contains water quality metrics for 3276 different water bodies. The target is to classify water Potability based on the 9 attributes ")

		
		file = file_upload()
		st.session_state.file_upload = file
		if st.session_state.file_upload:
			indice=True
			file = st.session_state.file_upload
		else: 
			indice=True
			st.session_state.file_upload = file

		if file:
			if 'load_csv' in st.session_state:
				df=st.session_state.load_csv
				st.write(file.name + " " +  "is loaded") 

			else:
				df = load_csv(file)
				st.session_state.load_csv = df

			
		# Dataframe columns and types	
		if file:
			st.write("Columns and their types:")
			col = column_types(df)
			st.write(col)

			# Show Dataframe 
			st.text("Display uploaded dataframe")
			if st.checkbox("Show dataset"):
				st.dataframe(df)

		st.subheader('2. Data visualisation 👀')
		st.text("Data visualisation")

		plot_types = ("Kernel Density Estimate","Bar")
		
		# User choose type
		chart_type = st.selectbox("Choose your chart type", plot_types)

		with st.beta_container():
			st.subheader(f"Showing:  {chart_type}")
			if chart_type== "Kernel Density Estimate":
				st.write("A kernel density estimate (KDE) plot is a method for visualizing the distribution of the water metrics in the dataset.")
			else: 
				st.write("Distribution of potability target class")


		if file:
			show_plot(df,chart_type)
				

#********************** End Data upload and visualisation ***************************  


#**************************** Start Data preprocessing *******************************


	if choices == 'Data preprocessing':
		st.header("Data preprocessing")
		st.subheader("1. Handeling missing values")
		with st.beta_expander("Impact of missing values on ML algorithms"):
			st.write("Training a model with a dataset that has a lot of missing values can drastically impact the machine learning model's quality. There are multiple ways to deal with missing values and they depend on the data and its quality (e.g deleting rows/column with missing values, imputing, etc...)")
		#the dataframe saved from page 1 in the session state is called now in page 2
		if st.session_state.file_upload:
			file = st.session_state.file_upload
			df = st.session_state.load_csv
			if 'plot' in st.session_state:
				st.button(label='View missing values distribution')
				fig=st.session_state.plot
				st.pyplot(fig)
			elif st.button(label='View missing values distribution'):
				fig=plot(df)
				#st.pyplot(fig)
				st.session_state.plot = fig
	
			dd=df	
			if 'plot_notmissing' in st.session_state:
				st.button(label='Deal with missing values')
				fig2=st.session_state.plot_notmissing
				st.pyplot(fig2)
				sign= True
			elif st.button(label='Deal with missing values'):
				sign= True
				fig2=plot_notmissing(dd)
				#st.pyplot(fig)
				st.session_state.plot_notmissing = fig2

			if 'save_dataset' in st.session_state:
				st.button("Save this dataset")
				dd = st.session_state.save_dataset
				st.info("Processed dataset saved!")
			elif st.button(label='Save Dataset') and sign:
				st.spinner("saving..")
				dd = save_dataset(dd,file)
				st.session_state.save_dataset = dd
				st.info("Dataset saved and will be used for training")
							

		else: 
			st.warning("Please upload file first")

#**************************** End Data preprocessing *******************************
		
#********************************* Start Modeling ***********************************		

	if choices == 'Modeling':
		st.header("Modeling 💡")

		col1, col2 = st.beta_columns((1, 1))

		with col2:

			st.subheader("2. Train Test split")
		
			train_size = st.selectbox("Train size", options=[70,80,90] , index=1)
			test_size = st.selectbox("Test size", options=[30,20,10], index=1)

			if (train_size + test_size != 100):
				st.error("Train and test size sum must be equal to 100")

			train_test_samples = my_train_test_split(test_size=test_size,train_size=train_size)

			if "my_train_test_split" in st.session_state:
				train_test_samples = st.session_state.my_train_test_split
				st.success("Dataset splitted!")
			elif st.button(label="Train-test-split"):
				with st.spinner("Split loading.."):
					train_test_samples = my_train_test_split(test_size=test_size,train_size=train_size)
					st.session_state.my_train_test_split = train_test_samples
				(train_X,test_X,train_Y,test_Y) = st.session_state.my_train_test_split
				st.success("Dataset splitted!")


		with col1:
			st.subheader("1. Classifier algorithm")

			st.selectbox("Select classification algorithm", options=["RandomForestClassifer"])

			st.info("Link to scikit-learn official documentation about the chosen model [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)")

				
		Col1, Col2 = st.beta_columns((1, 1))

		with Col1:
			st.subheader("3. Model parameters")
			model = rf_param_selector()
			if st.button("Train"):
				acc , f1, duration=training(model,test_size,train_size)
				accuracy = acc




		with Col2:
			st.subheader("4. Model metrics")
			
			st.write("Model accuracy")

			st.warning("Accuracy =  " + str(accuracy) )

			st.write("Model F1 score" )

			st.warning("F1 score =  " + str(f1))

			st.write("Training duration")

			st.warning("Traning took =  " + str(duration) + " seconds")

		Column1, Column2 = st.beta_columns((1, 1))

		with Column1:
			st.subheader("5. Saving trained model")
			if st.button("Save model"):
				st.success("model saved")



	
#******************************* End modeling **********************************


#****************************** Start prediction *******************************

	if choices == 'Prediction':
		st.header("Real-time prediction")

		default_id="urn:ngsi-ld:WaterPotabilityMetrics:<id>"

		st.write("Please provide the id of your request to the Context Broker")
		
		# Get ID from user function
		id = user_input(default_id)

		url_entities="http://backend.docker:8000/get_entities/" + id
	
			
		if 'get_attributes' in st.session_state:
			st.button(label='Get actual water metrics parameters')
			att=st.session_state.get_attributes
			att=get_attributes(url_entities)
			#test=False
		elif st.button(label='Get actual water metrics parameters'):
			att=get_attributes(url_entities)
			st.session_state.get_attributes = att
			#test=True

		if 'extract_attributes' in st.session_state:
			st.button(label='Process actual data')
			result= st.session_state.extract_attributes
			st.write("Extracted test sample")
			result = extract_attributes(att)
			st.write(result)
		elif st.button(label='Process actual data'):
			
			st.write("Extracted test sample")
			result = extract_attributes(att)
			st.write(result)
			st.session_state.extract_attributes = result
		

		def pred_and_patch(id,result):
			response = predict(result)
			notify_pred(id , response[1:-1])


		#Prediction 

		if 'predict' in st.session_state:
			st.button(label='Predict')
			response = predict(result)
			notify_pred(id , response[1:-1])
			if "Not Potable" in response:
				st.error(response[1:-1])
			else:
				st.success(response[1:-1])
			if st.button("Notify prediction"):
				notify_pred(id , response[1:-1])
		elif st.button(label='Predict'):
			response = predict(result)
			notify_pred(id , response[1:-1])
			st.session_state.predict = response
			if "Not Potable" in response:
				st.error(response[1:-1])
			else:
				st.success(response[1:-1])
			if st.button("Notify prediction"):
				notify_pred(id , response[1:-1])


		
if __name__ == '__main__':
	main()
