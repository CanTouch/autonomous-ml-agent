---
title: "StatQuest: Random Forests in R"
url: "https://www.youtube.com/watch?v=6EXPYzbfLCE"
date: "2018-02-26"
video_id: "6EXPYzbfLCE"
uploader: "StatQuest with Josh Starmer"
language: "en"
---

# StatQuest: Random Forests in R

you don't need a ukulele to do statistics but it makes it more fun hello I'm Josh stormer and welcome to stat quest today we're going to talk about how to build use and evaluate random forests in our this stat quest builds on two stat quests that I've already created that demonstrate the theory behind random forests so if you're not familiar with it check them out just so you know you can download all the code that I describe in

this tutorial using the link in the description below the first thing we do is load in ggplot2 so we can draw fancy graphs and when I do that our prints out a little message it's no big deal then we load Cal plot which just improves some of ggplot2 s default settings it overwrites the GG save function and that's fine with me so no worries here the last library we need to load is random forests duh so we can

make random forests it also prints out some stuff in red but it's no big deal we can move on from here for this example we're going to get a real data set from the UCI machine learning repository specifically we want the heart disease data set so we make a variable called URL and set it to the location of the data we want for our random forest and this is how we read the data set into our from the URL the head function

shows us the first six rows of data unfortunately none of the columns are labeled wha-wha so we named the columns after the names that were listed on the UCI website the UCI website actually lists a whole lot of information about this data so it's worth checking out if you haven't done that already hooray now when we look at the first six rows with the head function things look a lot better however the stur function which describes the structure of the data

tells us that some of the columns are messed up sex is supposed to be a factor where zero represents female and one represents male CP aka chest pain is also supposed to be a factor where levels 1 through 3 represent different types of pain and 4 represents no chest pain see a and Thal are correctly called factors but one of the levels is question mark when we need it to be an n/a so we've got some cleaning up to do

the first thing we do is change the question marks to n A's then just to make the data easier on the eyes we convert the zeros in sex to F for female and the ones to M for male lastly we convert the column into a factor then we convert a bunch of other columns into factors since that's what they're supposed to be see the UCI website or the sample code on the stat quest blog for more details

since the CA column originally had a question mark in it rather than in a are thinks it's a column of strings we correct that assumption by telling our it's a column of integers and then we convert it to a factor then we do the exact same thing for Thal the last thing we need to do to the data is make HD aka heart disease a factor that is easy on the eyes here I'm using a fancy trick with

if-else to convert the zeros to healthy and the ones to unhealthy we could have done a similar trick for sex but I wanted to show you both ways to convert numbers to words once we're done fixing up the data we can check that we've made the appropriate changes with the stir function sex is now a factor with levels F and M and everything else looks good too hooray we're done with the boring part now we can have some fun

since we are going to be randomly sampling things let's set the seed for the random number generator so that we can reproduce our results now we impute values for the NA s in the dataset with our F impute the first argument to our F impute is HD tilde dot and that means we want the HD aka heart disease column to be predicted by the data in all of the other columns here's where we specify which data set to use in this case

there's only one data set and it's called data here's where we specify how many random forests our F impute should build to estimate the missing values in theory four to six iterations is enough just for fun I set this parameter editor equal to 20 but it didn't improve the estimates lastly we save the results the data set with imputed values instead of n A's as data dot imputed after each iteration our F impute prints out the out of bag

OOB error rate this should get smaller if the estimates are improving since it doesn't we can conclude that our estimates are as good as they're going to get with this method here's where we actually build a proper random forest using the random forest function just like when we imputed values for the NA s we want to predict HD aka heart disease using all of the other columns in the data set however this time we specify data imputed as the data set we

also want random forests to return the proximity matrix we'll use this to cluster the samples at the end of the stat quest lastly we save the random forest and associated data like the proximity matrix as model to get a summary of the random forest and how well it performed we can just type model on the command prompt and then hit enter here's what gets printed to the screen the first thing is the original call to random forest next we

see that the random forest was built to classify samples if we had used the random forest to predict weight or height it would say regression and if we had omitted the thing the random forest was supposed to predict entirely it would say unsupervised then it tells us how many trees are in the random forest the default value is 500 later we will check to see if 500 trees is enough for optimal classification then it tells us

how many variables or columns of data were considered at each internal node classification trees have a default setting of the square root of the number of variables regression trees have a default setting of the number of variables divided by 3 since we don't know if 3 is the best value we'll fiddle with this parameter later on here's the out-of-bag oob error estimate this means that 83.5% of the oob samples were correctly classified by the random forest lastly we have a confusion matrix

there were 141 healthy patients that were correctly labeled healthy hooray there were 27 unhealthy patients that were incorrectly classified as healthy boo there were 23 healthy patients that were incorrectly classified unhealthy ooh lastly there were 112 unhealthy patients that were correctly classified unhealthy hooray to see if 500 is enough for optimal classification we can plot the error rates here we create a data frame that formats the error rate information so that ggplot2 will be happy this is kind of complicated so let

me walk you through it for the most part this is all based on a matrix within model called dot rate this is what the earth dot rate matrix looks like there's one column for the out-of-bag error rate one column for the healthy error rate ie how frequently healthy patients are misclassified and one column for the unhealthy error rate ie how frequently unhealthy patients are misclassified each row reflects the error rates at different stages of creating the random

forest the first row contains the error rates after making the first tree the second row contains the error rates after making the first two trees the last row contains the error rates after making all 500 trees so what we're doing here is making a data frame that looks like this there's one column for the number of trees there's one column for the type of error and one column for the actual error value and here's the call

to ggplot BAM the blue line shows the error rate when classifying unhealthy patients the green line shows the overall out-of-bag error rate the red line shows the error rate when classifying healthy patients in general we see the error rates decrease when our random forest has more trees if we added more trees with the error rate go down further to test this hypothesis we make a random forest with 1,000 trees the out-of-bag error rate is the same as

before and the confusion matrix shows that we didn't do a better job classifying patients and we can plot the error rates just like before double bam and we see that the error rates stabilize right after 500 trees so adding more trees didn't help but we would not have known this unless we used more trees now we need to make sure we are considering the optimal number of variables at each internal node in the tree we start by making an empty vector

that can hold ten values and then we create a loop that tests different numbers of variables at each step each time we go through the loop I increases by one it starts at one and ends after ten in this line we are building a random forest using AI to determine the number of variables to try at each step specifically we are setting m try equals I and I equals values between 1 and 10 this is where we store the out-of-bag

error rate after we build each random forest that uses a different value for M try this is a bit of complex code here's what's going on tamp top model contains a matrix called air dot rate just like model did before and we want to access the value in the last row and in the first column ie the out-of-bag error rate when all 1000 trees have been made now we can print out the out-of-bag error rates for different values for m

try the third value corresponding to m try equals 3 which is the default in this case has the lowest out-of-bag error rate so the default value was optimal but we wouldn't have known that unless we tried other values lastly we want to use the random forests to draw an MDS plot with samples this will show us how they are related to each other if you don't know what an MDS plot is don't freak out just check out the stat quest

on it we start by using the dist function to make a distance matrix from 1 minus the proximity matrix then we run CMD scale on the distance matrix CMD scale stands for classical multi-dimensional scaling then we calculate the percentage of variation in the distance matrix that the x and y axes account for again see the other stat quest for details then we format the data for ggplot and then we draw the graph with ggplot triple bam unhealthy samples are on the

left side healthy samples are on the right side I wonder if patient 253 was misdiagnosed and actually has heart disease note the x-axis accounts for 47% of the variation in the distance matrix the y-axis only accounts for 14% of the variation in the distance matrix that means that the big differences are along the x-axis lastly if we got a new patient and didn't know if they had heart disease and they clustered down here we'd be pretty

confident that they had heart disease [Music] hooray we've made it to the end of another exciting stat quest if you like this stat quest and want to see more of them please subscribe and if you have any suggestions for future stat quests well put them in the comments below until next time quest on
