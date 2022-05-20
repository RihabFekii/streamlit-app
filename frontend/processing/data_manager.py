import pandas as pd
import streamlit as st
from processing.preprocessing import deal_with_missing_values


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
		df_types=pd.DataFrame(df.dtypes, columns=['Data Type'])

	return df_types.astype(str) 

def save_dataset(df,file):
	new_df = deal_with_missing_values(df)
	# save dataset after preprocessing for training 
	name = file.name
	new_df.to_csv("/storage/preprocessed_" + name )

	return df