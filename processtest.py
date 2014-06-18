import pandas as pd 
import numpy as np 
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn import cross_validation
from sklearn import metrics 




if __name__ == "__main__":



	df_test = pd.read_csv("test.csv")
	df_test.index = df_test["EventId"]
	df_test = df_test.drop("EventId", axis=1)
	X = df_test



# transforming data

	columns_with_bad_data = []
	good_columns = []

	X2 = X.copy()
	X3 = X.copy() 
	# print X2.head() 
	
	for col in X.columns: 
		percent_bad = np.mean(X2[col] == -999.00)
		if percent_bad > .01:
			columns_with_bad_data.append(col)
			X2[col] = (X2[col] != -999.00) 
		else:
			good_columns.append(col) 

	# print X2.head() 

	# sanity check 
	print columns_with_bad_data

	X = scale(X)
	X2[good_columns] = scale(X2[good_columns])


	# now we're going to bin the problematic columns
	for col in columns_with_bad_data:

		col_max = np.max(X3[col])
		col_min = np.min(X3[col][X3[col] > -999.00])

		if col_min < 0:
			X3[col][X3[col] == -999.00] = 1.1 * col_min
		else:
			X3[col][X3[col] == -999.00] = -1.1 * col_min 

		X3[col] = X2[col]*pd.cut(X3[col], 10, labels=(0., 1., 2., 3., 4., 5., 6., 7., 8., 9.)) 


	X3[good_columns] = scale(X3[good_columns])
	print X3.head() 


	X3.to_csv("X_test.csv")








