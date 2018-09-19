#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###

#clf = SVC(kernel="linear")
clf = SVC(kernel="rbf", C=10000)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
prd = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print "Accuracy:", accuracy_score(labels_test, prd)

#print "10th: %r, 26th: %r, 50th: %r" % (prd[10], prd[26], prd[50])

print "No. of predicted to be in the 'Chris'(1): %r" % sum(prd)

#########################################################


