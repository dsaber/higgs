import pandas as pd 
import numpy as np 
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn import cross_validation
from sklearn import metrics 




if __name__ == "__main__":

	# load the data in 
	df = pd.read_csv("training.csv")
	df.index = df["EventId"]
	df = df.drop("EventId", axis=1)
	df = df.drop("Weight", axis=1)



	# create our Y
	Y = df["Label"]
	Y[Y == "b"] = -1
	Y[Y == "s"] = 1 
	Y = Y.ravel() 
	Y = Y.astype(int) 

	Ylog = Y.copy()
	Ylog[Ylog == -1] = False 
	Ylog[Ylog == 1] = True 
	Ylog = Ylog.astype(int) 

	# create and scale our X
	X = df.drop("Label", axis=1)

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



	# Now try running two basic logistic regressions, one 
	# where we "binarize" the missing data away and one where we
	# simply leave it in

	print "---------------"
	print "LOGISTIC REGRESSION: " 

	log1 = LogisticRegression()
	log1.fit(X, Ylog)
	log1_score = np.mean(cross_validation.cross_val_score(log1, X, Ylog, cv=3, scoring="accuracy"))
	print "Leaving data as is: " + str(log1_score)

	log2 = LogisticRegression()
	log2.fit(X2, Ylog)
	log2_score = np.mean(cross_validation.cross_val_score(log2, X2, Ylog, cv=3, scoring="accuracy"))
	print "Transforming: " + str(log2_score)

	log3 = LogisticRegression() 
	log3.fit(X2[good_columns], Ylog) 
	log3_score = np.mean(cross_validation.cross_val_score(log3, X2[good_columns], Ylog, cv=3, scoring="accuracy"))
	print "Dropping columns with bad data: " + str(log3_score)

	log4 = LogisticRegression()
	log4.fit(X3, Ylog)
	log4_score = np.mean(cross_validation.cross_val_score(log4, X3, Ylog, cv=3, scoring="accuracy"))
	print "Binning columns: " + str(log4_score) 


	# print "---------------"
	# print "SVM: "

	# svm1 = svm.SVC(class_weight="auto")
	# svm1.fit(X, Y)
	# svm1_score = np.mean(cross_validation.cross_val_score(svm1, X, Y, cv=3, scoring="accuracy")) 
	# print "Leaving data as is: " + str(svm1_score)

	# svm2 = svm.SVC(class_weight="auto")
	# svm2.fit(X2, Y)
	# svm2_score = np.mean(cross_validation.cross_val_score(svm2, X2, Y, cv=3, scoring="accuracy"))
	# print "Transforming: " + str(svm2_score)

	X3.to_csv("X_train.csv")
	pd.DataFrame(Ylog).to_csv("Ylog.csv")
	pd.DataFrame(Ylog).to_csv("Y.csv")










