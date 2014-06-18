import pandas as pd 
import numpy as np 
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm 
from sklearn import cross_validation
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":

	X = pd.read_csv("X_train.csv")
	# print Y_hat.head() 
	X.index = X["EventId"]
	X = X.drop("EventId", axis=1)

	Y = pd.read_csv("Y.csv")
	Y = Y["0"]
	Y = Y.ravel() 

	Ylog = pd.read_csv("Ylog.csv")
	Ylog = Ylog["0"]
	Ylog = Ylog.ravel() 

	print X.shape 

	svm1 = GradientBoostingClassifier(verbose=3) 
	svm1.fit(X, Y)
	#svm1_score = np.mean(cross_validation.cross_val_score(svm1, X, Y, cv=3, scoring="accuracy", n_jobs=-1, verbose=2))
	#print svm1_score



	#######################################

	print "NOW FOR THE TEST DATA..." 

	X_test = pd.read_csv("X_test.csv")
	Y_hat = X_test["EventId"].copy() 
	Y_hat = pd.DataFrame(Y_hat, columns=["EventId"])
	X_test.index = X_test["EventId"]
	X_test = X_test.drop("EventId", axis=1)


	Y_hat["RankOrder"] = range(1, Y_hat.shape[0] + 1)


	Y_hat["Class"] = svm1.predict(X_test)
	Y_hat["Class"][Y_hat["Class"] == 1] = "s"
	Y_hat["Class"][Y_hat["Class"] == 0] = "b"

	Y_hat.index = Y_hat["EventId"]
	Y_hat = Y_hat.drop("EventId", axis=1)

	Y_hat.to_csv("predictions2.csv", index=True)

