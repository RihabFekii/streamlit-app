import streamlit as st 
from streamlit import caching
import pandas as pd 
import altair as alt


st.set_page_config(page_title ="AI service App",initial_sidebar_state="collapsed",page_icon="🔮")

@st.cache(persist=False,allow_output_mutation=True,suppress_st_warning=True,show_spinner= True)
  

def main():

	#reading a csv with pandas
	def load_csv():
		df_input = pd.DataFrame()  
		df_input=pd.read_csv(input)
		return df_input

	#saving into a csv a pandas dataframe 
	def save_csv(df_input):
		df=pd.DataFrame(input)
		df_input.to_csv(path_or_buf="/app/data/data.csv")


	st.title("Powered by FIWARE AI service for water quality assessment")
	st.header("Water potability prediction 💦 ")
	st.write("This application enables to classify the potability of water based on the water composition and water quality metrics")
	st.header("Context")
	st.write("Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level.")
	st.header("About this application")
	st.write("This application will explore the different features related to water potability, Modeling, and predicting water potability. It presents an in-depth analysis of what separates potable water from non-potable using statistics, bayesian inference, and other machine learning approaches.")
	#st.markdown("""The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)**.""")

	caching.clear_cache()
	df =  pd.DataFrame()

	activities = ["AI application","Plots"]

	choices = st.sidebar.selectbox('Select Activities',activities)


		
	if choices == 'AI application':

		#********************** Degin Data upload ****************************

		st.subheader('1. Data loading 🏋️')
		st.write("Import a time series csv file")
		#with st.beta_expander("Data format"): 
			#st.write("The dataset can contain multiple columns but you will need to select a column to be used as dates and a second column containing the metric you wish to forecast. The columns will be renamed as **ds** and **y** to be compliant with Prophet. Even though we are using the default Pandas date parser, the ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric.")
		with st.beta_expander("Minimum Daily Temperatures Dataset information"):
			st.write("This dataset describes the minimum daily temperatures over 10 years (1981-1990) in the city Melbourne, Australia. The units are in degrees Celsius and there are 3650 observations. The source of the data is credited as the Australian Bureau of Meteorology.")
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
		
		st.subheader('2. Data visualisation 👀')
		st.text("Display uploaded dataframe")
		if st.checkbox("Show dataset"):
			st.dataframe(df)
		
		st.text("Data visualisation")

		plot_types = ("Line","Pie","Histogram","Bar","Scatter")
		

		# User choose type
		chart_type = st.selectbox("Choose your chart type", plot_types)

		with st.beta_container():
			st.subheader(f"Showing:  {chart_type}")
			#st.write("")

		

		def altair_plot(chart_type: str, df):
			""" return altair plots """

			if chart_type == "Scatter":
				fig = (alt.Chart(df,title="Tempreture patern")
				.mark_point()
				.encode(x="bill_depth_mm", y="bill_length_mm", color="species")
				.interactive() )
			elif chart_type == "Histogram":
				fig = (
				alt.Chart(df, title="Tempreture patern")
				.mark_bar()
				.encode(alt.X("bill_depth_mm", bin=True), y="count()")
				.interactive()
			)
			elif chart_type == "Bar":
				fig = (
				alt.Chart(df).mark_bar().encode(
				x='Potability:O',
				y= 'count(Potability):O',
				color='Potability:N'
				).interactive()
				)
			elif chart_type == "Boxplot":
				fig = (
				alt.Chart(df).mark_boxplot().encode(x="species:O", y="bill_depth_mm:Q")
				)
			elif chart_type == "Line":
				fig = (
				alt.Chart(df, title="Tempreture patern")
				.mark_line()
				.encode(x="Date:T", y="Daily_min_tem:Q")
				.interactive()
				)
			elif chart_type == "Pie":
				fig = (

				)
			return fig


		# create plots
		def show_plot(df):
			plot = altair_plot(chart_type, df)
			st.altair_chart(plot, use_container_width=True)

		
		

		if input:
			show_plot(df)
	
		#*********************** End plotting ******************************


		#********************** Begin Data preprocessing ********************

		#st.subheader("3. Data processing")

		#Area to select columns based on user input 
		#Link: https://stackoverflow.com/questions/66885387/how-to-plot-bar-chart-by-allowing-the-user-to-choose-the-columns-using-plotly-an
		#if input:
			#columns= list(df.columns)

			#selected_columns = st.multiselect("select column", options=columns)"""

		#*********************** End Data preprocessing ********************

		st.subheader("4. Saving dataset")
		if st.button("save data"):
			save_csv(df)

		

	if choices == 'Plots':
		st.subheader("Visualization")
if __name__ == '__main__':
	main()