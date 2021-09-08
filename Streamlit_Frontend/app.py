import streamlit as st 
from streamlit import caching
import pandas as pd 
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import logging


st.set_page_config(page_title ="AI service App",initial_sidebar_state="expanded", layout="wide", page_icon="💦")

#@st.cache(persist=False,allow_output_mutation=True,suppress_st_warning=True,show_spinner= True)
  
df =  pd.DataFrame()


def main():
 
	st.title("Powered by FIWARE AI service for water quality assessment")
	st.header("Water potability prediction 💦 ")
	st.write("This application enables to classify the potability of water based on the water composition and water quality metrics")
	
	#caching.clear_cache()


	activities = ["About this AI application","Data upload and visualisation","Data preprocessing","Modeling", "Prediction"]
	
	st.sidebar.title("Navigation")
	choices = st.sidebar.radio("",activities)

    
		
	if choices == 'About this AI application':

		st.image("/storage/img2.jpg")
		st.header("Context")
		st.write("Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level.")
		st.header("About this application")
		st.write("This application will explore the different features related to water potability, Modeling, and predicting water potability. It presents an in-depth analysis of what separates potable water from non-potable using statistics, bayesian inference, and other machine learning approaches.")
		#st.markdown("""The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)**.""")

		

		
	if choices == 'Data upload and visualisation':

	#********************** Degin Data upload ****************************

		if "df" not in st.session_state:
			st.session_state.df = pd.read_csv("/storage/data.csv")
			st.session_state.columns = ["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity","Potability"]
			st.session_state.dtypes = [ "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[int64]'>" ]

		
		
		#def load_csv(ds):
			#df_input = pd.DataFrame()
			#df_input=pd.read_csv(ds)  
			#return df_input


		st.subheader('1. Data loading 🏋️')

		st.write("Import your water potability CSV dataset")
		with st.beta_expander("Water potability metrics dataset information"):
			st.write("This dataset contains water quality metrics for 3276 different water bodies. The target is to classify water Potability based on the 9 attributes ")
		
		input = st.file_uploader('', key="dataframe")
		

		if input:
			with st.spinner('Loading data..'):
				#df = load_csv(input)
				df = st.session_state.df


				st.write("Columns and their types:")
				st.write(df.dtypes)

		
	#********************** End data upload ***************************

		
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

		# create plots
		def show_plot(df):
			if chart_type != "Kernel Density Estimate":
				plot = altair_plot(chart_type, df)
				st.altair_chart(plot, use_container_width=True)
			if chart_type == "Kernel Density Estimate":
				sns_plot(chart_type, df)

		if input:
			show_plot(df)
				
	
	#*********************** End plotting ******************************

	#*********************** Start Data preprocessing ******************************


	if choices == 'Data preprocessing':
		st.subheader("Data preprocessing")

		st.write("Handeling missing values")

		ddd=pd.read_csv("/storage/data.csv")

		def plot(ddd):
			fig= plt.figure()
			plt.title('Missing Values Per Feature')
			nans = ddd.isna().sum().sort_values(ascending=False).to_frame()
			sns.heatmap(nans,annot=True,vmin=None,fmt='d',cmap='Blues')
			st.pyplot(fig)
			return(fig)


		if 'plot' in st.session_state:
			st.button(label='View missing values distribution')
			fig=st.session_state.plot
			st.pyplot(fig)
		elif st.button(label='View missing values distribution'):
			fig=plot(ddd)
			#st.pyplot(fig)
			st.session_state.plot = fig

		def plot_notmissing(dd):
			for col in dd.columns:
				
				missing_label_0 = dd.query('Potability == 0')[col][dd[col].isna()].index
				dd.loc[missing_label_0,col] = dd.query('Potability == 0')[col][dd[col].notna()].mean()

				missing_label_1 = dd.query('Potability == 1')[col][dd[col].isna()].index
				dd.loc[missing_label_1,col] = dd.query('Potability == 1')[col][dd[col].notna()].mean()

			fig= plt.figure()
			plt.title('Missing Values Per Feature')
			nans = dd.isna().sum().sort_values(ascending=False).to_frame()
			sns.heatmap(nans,annot=True,vmin=None,fmt='d',cmap='Blues')
			st.pyplot(fig)
			return(fig)
			
		dd=ddd
		if 'plot_notmissing' in st.session_state:
			fig2=st.session_state.plot_notmissing
			st.pyplot(fig2)
		elif st.button(label='Deal with missing values'):
			fig2=plot_notmissing(dd)
			#st.pyplot(fig)
			st.session_state.plot_notmissing = fig2
		
	#***************************** Start modeling *******************************		


	if choices == 'Modeling':
		st.subheader("Modeling")

		default_id="urn:ngsi-ld:WaterPotabilityMetrics:<id>"

		st.write("Please provide the id of your request to the Context Broker")

		id = st.text_input("User input", default_id)

		def get_attributes(id):

			url1="http://backend.docker:8000/get_entities/" + id

			entities = requests.request("GET",url1)
			#logging.info(entities)

			st.write(entities.json())

			p = entities.json()  
			return p
	

		if 'get_attributes' in st.session_state:
			st.button(label='Get actual water metrics parameters')
			att=st.session_state.get_attributes
			att=get_attributes(id)
		elif st.button(label='Get actual water metrics parameters'):
			att=get_attributes(id)
			st.session_state.get_attributes = att

		if st.button("click me"):
			st.write(att)
	
	#*********************** End modeling **********************************


	#*********************** Start prediction ******************************

	if choices == 'Prediction':
		st.subheader("Prediction")

		default_id="urn:ngsi-ld:WaterPotabilityMetrics:<id>"

		st.write("Please provide the id of your request to the Context Broker")

		id = st.text_input("User input", default_id)

		def get_attributes(id):

			url1="http://backend.docker:8000/get_entities/" + id

			entities = requests.request("GET",url1)
			#logging.info(entities)

			st.write(entities.json())

			p = entities.json()  
			return p
	

		if 'get_attributes' in st.session_state:
			st.button(label='Get actual water metrics parameters')
			att=st.session_state.get_attributes
			att=get_attributes(id)
		elif st.button(label='Get actual water metrics parameters'):
			att=get_attributes(id)
			st.session_state.get_attributes = att


		if st.button("click"):
	
			st.write("Extracted Test sample")
			p=att	
			result = dict(ph=p['Ph'],Hardness=p['Hardness'], Solids = p['Solids'],Chloramines = p['Chloramines'], 
			Sulfate = p['Sulfate'], Conductivity = p['Conductivity'], Organic_carbon = p['Organic_carbon'],
			Trihalomethanes = p['Trihalomethanes'], Turbidity = p['Turbidity'] )

			st.write(result)
				

		#Prediction 
		header = {'Content-Type': 'application/json'}
		url3= "http://backend.docker:8000/prediction"
		#url= "http://backend_aliases:8000/prediction"

		payload=json.dumps(
			{
			"ph":3.716080,
			"Hardness":204.890455,
			"Solids":20791.318980,
			"Chloramines":7.300211,
			"Sulfate":368.516441,
			"Conductivity":564.308654,
			"Organic_carbon":10.379783,
			"Trihalomethanes":86.990970,
			"Turbidity":2.963135
			}
		)
		
	

		if st.button("predict"):
			response = requests.request("POST", url3, headers=header, data=payload)
			st.success(response.text)


		
if __name__ == '__main__':
	main()