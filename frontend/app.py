import pandas as pd
import streamlit as st

from modeling.models_params import rf_param_selector
from modeling.training import my_train_test_split, training
from prediction.input_manager import (extract_attributes, get_attributes,
                                      user_input)
from prediction.predict import notify_pred, predict
from processing.data_manager import (column_types, file_upload, load_csv,
                                     save_dataset)
from visualisation.visualisation import plot, plot_notmissing, show_plot

#***************************** Start HTML configuration *****************************

st.set_page_config(page_title ="AI service App",initial_sidebar_state="expanded", layout="wide", page_icon="💦")
  
# change the color of button widget
Color = st.get_option("theme.secondaryBackgroundColor")
s = f"""
<style>
div.stButton > button:first-child {{background-color: #fffff ; border: 2px solid {Color}; border-radius:5px 5px 5px 5px; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)

#************************* End HTML configuration ***************************

def main():
	
	st.title("AI service for water quality assessment powered by FIWARE ")
	st.header("Water potability prediction 💦 ")
	st.write("This application enables to classify the potability of water based on the water composition and water quality metrics")
	
	activities = ["About this AI application","Data upload and visualisation","Data preprocessing","Modeling", "Prediction"]
	st.sidebar.title("Navigation")
	choices = st.sidebar.radio("",activities)

	sign = False
	train_X = pd.DataFrame()
	test_X = pd.DataFrame()
	train_Y = pd.DataFrame()
	test_Y = pd.DataFrame()
	accuracy = "Not yet defined before training"
	f1 = "Not yet defined before training"
	duration = "..."


 #************************* Start About this AI application ***************************  

	if choices == 'About this AI application':

		st.image("/storage/img2.jpg")
		st.header("Context")
		st.write("Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level.")
		st.header("About this application")
		st.write("This application will explore the different features related to water potability, Modeling, and predicting water potability. It presents an in-depth analysis of what separates potable water from non-potable using statistics, bayesian inference, and other machine learning approaches.")

#********************** End About this AI application *********************************


#********************** Start Data upload and visualisation ***************************  
		
	if choices == 'Data upload and visualisation':

		st.subheader('1. Data loading 🏋️')

		st.write("Import your water potability CSV dataset")
		with st.expander("Water potability metrics dataset information"):
			st.write("This dataset contains water quality metrics for 3276 different water bodies. The target is to classify water Potability based on the 9 attributes ")

		
		file = file_upload()
		st.session_state.file_upload = file
		if st.session_state.file_upload:
			file = st.session_state.file_upload
		else: 
			st.session_state.file_upload = file

		if file:
			if 'load_csv' in st.session_state:
				df = st.session_state.load_csv
				st.write(file.name + " " +  "is loaded") 

			else:
				df = load_csv(file)
				st.session_state.load_csv = df
			

		# Dataframe columns and types	
		if file:
			st.write("Columns and their types:")
			df_col = column_types(df)
			st.write(df_col)

			# Show Dataframe 
			st.text("Display uploaded dataframe")
			if st.checkbox("Show dataset"):
				st.dataframe(df)

		st.subheader('2. Data visualisation 👀')
		st.text("Data visualisation")

		plot_types = ("Kernel Density Estimate","Bar")
		
		# User choose type
		chart_type = st.selectbox("Choose your chart type", plot_types)

		with st.container():
			st.subheader(f"Showing:  {chart_type}")
			if chart_type == "Kernel Density Estimate":
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
		with st.expander("Impact of missing values on ML algorithms"):
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
				fig = plot(df)
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

		col1, col2 = st.columns((1, 1))

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
				train_X, test_X, train_Y, test_Y = st.session_state.my_train_test_split
				st.success("Dataset splitted!")


		with col1:
			st.subheader("1. Classifier algorithm")

			st.selectbox("Select classification algorithm", options=["RandomForestClassifer"])

			st.info("Link to scikit-learn official documentation about the chosen model [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)")

				
		Col1, Col2 = st.columns((1, 1))

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

		Column1, Column2 = st.columns((1, 1))

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

		elif st.button(label='Get actual water metrics parameters'):
			att=get_attributes(url_entities)
			st.session_state.get_attributes = att

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
