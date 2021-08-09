import streamlit as st 
from streamlit import caching
import pandas as pd 
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json


st.set_page_config(page_title ="AI service App",initial_sidebar_state="expanded", layout="wide", page_icon="💦")

#@st.cache(persist=False,allow_output_mutation=True,suppress_st_warning=True,show_spinner= True)
  


def main():
 
	st.title("Powered by FIWARE AI service for water quality assessment")
	st.header("Water potability prediction 💦 ")
	st.write("This application enables to classify the potability of water based on the water composition and water quality metrics")
	
	#caching.clear_cache()
	df =  pd.DataFrame()

	activities = ["About this AI application","Data visualisation","Data preprocessing","Modeling", "Prediction"]

	choices = st.sidebar.radio('Tabs',activities)

    
		
	if choices == 'About this AI application':

		st.header("Context")
		st.write("Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level.")
		st.header("About this application")
		st.write("This application will explore the different features related to water potability, Modeling, and predicting water potability. It presents an in-depth analysis of what separates potable water from non-potable using statistics, bayesian inference, and other machine learning approaches.")
		#st.markdown("""The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)**.""")

		

		
	if choices == 'Data visualisation':

		#********************** Degin Data upload ****************************

		if "df" not in st.session_state:
			st.session_state.df = pd.read_csv("/storage/data.csv")
			st.session_state.columns = ["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity","Potability"]
			st.session_state.dtypes = [ "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[float64]'>", "<class 'numpy.dtype[int64]'>" ]

		#reading a csv with pandas
		
		def load_csv(ds):
			df_input = pd.DataFrame()
			df_input=pd.read_csv(ds)  
			#df_input=pd.read_csv(input)
			#df_input=pd.read_csv(st.session_state.dataframe)
			return df_input


		st.subheader('1. Data loading 🏋️')

		st.write("Import water potability csv file")
		with st.beta_expander("Drinking water potability metrics dataset information"):
			st.write("This dataset contains water quality metrics for 3276 different water bodies. The target is to classify water Potability based on the 9 attributes ")
		
		input = st.file_uploader('', key="dataframe")
		
	
		if input:
			with st.spinner('Loading data..'):
				#df = load_csv(st.session_state.df)
				df = st.session_state.df

				st.write("Columns:")
				st.write(st.session_state.columns)

				st.write("Columns types:")
				st.write(st.session_state.dtypes)

		
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
			#st.write("")

		
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



	if choices == 'Data preprocessing':
		st.subheader("Data preprocessing")


	if choices == 'Modeling':
		st.subheader("Modeling")

	if choices == 'Prediction':
		st.subheader("Prediction")
		url= "http://backend.docker:8000/prediction"
		#url= "http://backend_aliases:8000/prediction"

		payload=json.dumps(
			{
			"ph":3.716080,
			"Hardness":204.8904554713363,
			"Solids":20791.318980747023,
			"Chloramines":7.300211873184757,
			"Sulfate":368.51644134980336,
			"Conductivity":564.3086541722439,
			"Organic_carbon":10.3797830780847,
			"Trihalomethanes":86.9909704615088,
			"Turbidity":2.9631353806316407
			}
		)

		headers = {'Content-Type': 'application/json'}

		st.write("Test sample")
		sample={
			"ph":3.716080,
			"Hardness":204.8904554713363,
			"Solids":20791.318980747023,
			"Chloramines":7.300211873184757,
			"Sulfate":368.51644134980336,
			"Conductivity":564.3086541722439,
			"Organic_carbon":10.3797830780847,
			"Trihalomethanes":86.9909704615088,
			"Turbidity":2.9631353806316407
			}
		st.write(sample)
	

		if st.button("predict"):
			response = requests.request("POST", url, headers=headers, data=payload)
			st.write(response.text)


		
if __name__ == '__main__':
	main()