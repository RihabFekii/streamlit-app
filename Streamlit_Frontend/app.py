import streamlit as st 
from streamlit import caching
import pandas as pd 
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title ="AI service App",initial_sidebar_state="collapsed",page_icon="🔮")
#st.set_option('wideMode' , True)
@st.cache(persist=False,allow_output_mutation=True,suppress_st_warning=True,show_spinner= True)
  

def main():

	st.title("Powered by FIWARE AI service for water quality assessment")
	st.header("Water potability prediction 💦 ")
	st.write("This application enables to classify the potability of water based on the water composition and water quality metrics")
	
	caching.clear_cache()
	df =  pd.DataFrame()

	activities = ["About this AI application","Data visualisation","Data preprocessing","Modeling", "Prediction"]

	choices = st.sidebar.radio('Tabs',activities)

    
		
	if choices == 'About this AI application':

		st.header("Context")
		st.write("Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level.")
		st.header("About this application")
		st.write("This application will explore the different features related to water potability, Modeling, and predicting water potability. It presents an in-depth analysis of what separates potable water from non-potable using statistics, bayesian inference, and other machine learning approaches.")
		#st.markdown("""The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)**.""")

		#********************** Degin Data upload ****************************
		#reading a csv with pandas
		def load_csv():
			df_input = pd.DataFrame()  
			df_input=pd.read_csv(input)
			return df_input

		#saving into a csv a pandas dataframe 
		def save_csv(df_input):
			df=pd.DataFrame(input)
			df_input.to_csv(path_or_buf="/app/data/data.csv")


		st.subheader('1. Data loading 🏋️')
		st.write("Import a time series csv file")
		#with st.beta_expander("Data format"): 
			#st.write("The dataset can contain multiple columns but you will need to select a column to be used as dates and a second column containing the metric you wish to forecast. The columns will be renamed as **ds** and **y** to be compliant with Prophet. Even though we are using the default Pandas date parser, the ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric.")
		with st.beta_expander("Minimum Daily Temperatures Dataset information"):
			st.write("This dataset contains water quality metrics for 3276 different water bodies. The target is to classify water Potability based on the 9 attributes ")
		input = st.file_uploader('')

		if input:
			with st.spinner('Loading data..'):
				df = load_csv()

				st.write("Columns:")
				st.write(list(df.columns))

				st.write("Columns types:")
				st.write(list(df.dtypes))

		
		#********************** End data upload ***************************

		
		#********************** Begin Plotting ****************************
		
		
		st.text("Display uploaded dataframe")
		if st.checkbox("Show dataset"):
			st.dataframe(df)
		
	if choices == 'Data visualisation':
		
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
				non_potabale = df.query('Potability == 0')
				potabale     = df.query('Potability == 1')
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

		
		show_plot(state.df)
				
	
		#*********************** End plotting ******************************


		#********************** Begin Data preprocessing ********************

	
		#st.subheader("3. Data processing")

		#Area to select columns based on user input 
		#Link: https://stackoverflow.com/questions/66885387/how-to-plot-bar-chart-by-allowing-the-user-to-choose-the-columns-using-plotly-an
		#if input:
			#columns= list(df.columns) 

			#selected_columns = st.multiselect("select column", options=columns)"""

		#*********************** End Data preprocessing *******************

	if choices == 'Data preprocessing':
		st.subheader("4. Saving updated dataset")
		if st.button("save data"):
			save_csv(state.df)

		

	if choices == 'Modeling':
		st.subheader("Modeling")
if __name__ == '__main__':
	main()