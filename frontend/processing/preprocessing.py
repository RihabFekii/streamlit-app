
# Plot dataset after handling missing values -> replaces zeros with the mean of the values in that column coreesponding to the same target class. 
def deal_with_missing_values(df):
	for col in df.columns:	
		missing_label_0 = df.query('Potability == 0')[col][df[col].isna()].index
		df.loc[missing_label_0,col] = df.query('Potability == 0')[col][df[col].notna()].mean()

		missing_label_1 = df.query('Potability == 1')[col][df[col].isna()].index
		df.loc[missing_label_1,col] = df.query('Potability == 1')[col][df[col].notna()].mean()
	return df