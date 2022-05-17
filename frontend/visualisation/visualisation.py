import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from processing.preprocessing import deal_with_missing_values


# Bar plot with Altair 
def altair_plot(chart_type: str, df):
	""" return altair plots """

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
	


# Create plots
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

	return fig

# Plot dataset after handling missing values
def plot_notmissing(df):
	df = deal_with_missing_values(df)
	fig = plot(df)

	return fig
