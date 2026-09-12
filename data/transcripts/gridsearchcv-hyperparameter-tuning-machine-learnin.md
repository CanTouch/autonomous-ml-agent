---
title: "GridSearchCV | Hyperparameter Tuning | Machine Learning with Scikit-Learn Python"
url: "https://www.youtube.com/watch?v=TvB_3jVIHhg"
date: "2021-08-13"
video_id: "TvB_3jVIHhg"
uploader: "Normalized Nerd"
language: "en"
---

# GridSearchCV | Hyperparameter Tuning | Machine Learning with Scikit-Learn Python

hello people from the future welcome to normalized nerd in this video i'm gonna talk about the grid search and how you can implement it using scikit-learn if you are new here then please subscribe to this channel because i make videos about machine learning and data science regularly so without any further ado let's get started okay first of all we need to import the two most important libraries for machine learning that is numpy and pandas and then we will be using the pandas

read csv function to read the data set and store it into a pandas data frame now the data that i'm using here is publicly available on kaggle and i will share the link in the video description so make sure to check that then i'm just printing the first few rows from our data frame now from the column names you can probably guess that this data set is actually about the strength of concrete the last column strength is our target

variable so given a set of features related to some concrete mixture we need to predict its strength and we can see that we have some features like amount of cement used blast furnace lag fly ash used water used etc obviously you can use all sorts of featured engineering and feature extraction methods on this data set to find the best set of features but in this video i'm not gonna do that the main focus of this video is to implement

grid search so i'm just skipping the featured extraction part so now i'm just separating the feature matrix and the target vector into two variables x and y and don't forget to reshape the y vector into a 2d list because the scikit-learn models take only 2d lists and now i'm just going to print the shapes of x and y just to be sure that we haven't messed up before jumping into the grid search we need to first split our data set into

train set and test set please remember that grid search is performed only on the training set during the training it does not and should not see the test set because you don't want to tune your hyper parameters based on the test set for the splitting we are using the train test split method and the test size is 20 now only one last step is left before jumping into the grid search and that is to define a base model

here i'm gonna use the xg boost regressor why because most of the times you will find yourself tuning the hyper parameters of xgboost model so here i am defining a xd boost regressor because our problem is a regression problem right and in the parameter i am only passing a random state so that you can reproduce the results and please notice i am passing no hyper parameters because we are gonna tune them okay now that we have got a

base model we are ready to implement the grid search the first step of grid search is to make a search space so what is the search space well it's nothing but the set of values of the hyper parameters that you want to search in order to find the best set of values for your hyper parameters so to do this we need to make a dictionary and here i am naming it as search space and the keys of the dictionary will be

the names of the hyper parameters so for example in estimators denote the number of trees you want to have in the xg boost model and here i am passing three values 100 200 and 500 that means in the grid search process we are going to pick each value for the number of trees and train a model and then we are going to find which value works the best the next hyper parameter is max depth which simply denotes the depth of the

tree then we have gamma and the learning rate so our dictionary is ready so from this dictionary can you find out how many models we are gonna train during our grid search process yes 72 because we have got three choices for the first one three choices for the second two choices for the third and fourth choices for the last so if you multiply all this you will get 72 so now we need to import the grid search cv

class from the model selection module of sklearn then i'm just creating an object of this grid search now we need to pass all the information regarding our grid search as parameters so the first parameter is obviously the model then we need to pass the search space dictionary then we need to pass some scoring methods here i am using the r2 and rmse scoring matrix now notice that here the rmse is actually negative root mean squared error and to know all the available

scoring matrix you can just print the sklearn.matrix.scorers.keys now the scoring matrix is actually very important because we need some number to compare all the 72 models right and scikit-learn will just use these scoring matrix to score each model now comes refit now here i'm passing r square that means the gs object will return a model that is best with respect to the r squared matrix then we have got the cross validation and here i am using

five-fold cross validation now if you don't know about the k-fold cross validation then i will highly suggest you to watch the video that i did on k4 cross validation after that verbose to tell how much information we want to print okay so the grid search object is ready and now we need to train this on our training set now if you read them you will understand how different set of hyper parameters affect the scoring matrix okay so the execution is done and now

i'm gonna print the best model and the way to do this is to use the best estimated method so here's the model with the best set of hyper parameters and if you want to get only the best set of hyper parameters then you can use the best params method so here's the optimal set of five parameters and you can see the best score of the model by using the best score method and the metric will be what you have used in the

refit method so our best model has achieved an r squared value of 0.92 which is pretty good now if you really want to analyze your model in detail then i will highly suggest you to make a csv file of your grid search result so i'm just creating a file from the cv results and i'm sorting them based on the r square score now let me show you how the csv file looks like so in this file you will find

all sorts of information corresponding to the grid search that we have just performed now i wanna show you something interesting that will actually help you in making a better and efficient model for example if you look at the parameter in estimators that is the number of trees used in the xgboost model so the top model uses 500 trees and the second best model uses 200 trees so there's a difference of 300 trees and that's a lot of computation power there

now if you see the mean r squared value that we got from the fivefold cross validation the best model achieves an r squared value of 0.9228 and the second best model achieves 0.92218 now does it really make sense to use extra 300 trees for such a small improvement in r squared value i don't think so so that's how you can use this csv file to come up with the more efficient model so that was all for this video guys i

really hope that you enjoyed learning about grid search and if you found this video interesting then please share this video and please don't forget to subscribe my channel stay safe and thanks for watching [Music] you
