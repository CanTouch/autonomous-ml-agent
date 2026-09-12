---
title: "Train Test Split vs K Fold vs Stratified K fold Cross Validation"
url: "https://www.youtube.com/watch?v=fKz-SgScM3Q"
date: "2019-08-08"
video_id: "fKz-SgScM3Q"
uploader: "Krish Naik"
---

# Train Test Split vs K Fold vs Stratified K fold Cross Validation

hello all my name is Krishna and welcome to my youtube channel in my previous video I have already discussed about various types of cross-validation and in this particular video I am going to show you how we can bc implement k-fold cross-validation and stratified k-fold cross-validation and will also try to see the difference between them again in the theoretical part which was in my previous video I have already explained about the theory now what I am basically

going to do is that if you remember in one of my playlists I had actually implemented something called as hyper parameter optimization for XG boost using the randomizer CV I am going to take the same example and this particular video you can basically search in the YouTube channel itself and you'll be able to see this I'm just going to continue this I am going to you know use AZ boost and after that you know for predicting the accuracy for

checking the accuracy I am going to basically use k-fold cross-validation don't stratified k-fold cross-validation just to give you a concept this is basically a churn modeling data set we have Bank custom stone modeling we need to basically predict whether the customer will leave the bank in the future or not and based on that banks can basically you know suggest some products some good products some good offers to the customer so that they do not leave the bank so this particular

churn modeling we have basically created by applying eggs eBoost now here you can see we have actually applied hyper parameter optimization and this particular video is already uploaded the main thing that we will be focusing on is that how to create your cross validation that is careful CV and stratified k-fold cross-validation now first of all we're all cross validation can be very very useful okay so guys remember suppose you want to verify between two different models or two

different algorithms or three different algorithms or many different algorithms which is performing well you can basically and that too after the hyper parameter optimization you can use this particular model and you can basically see that which model basically performs well with the help of cross validation how you'll do it I'll just give you an example suppose first of all let me just show you and what will happen if I do just train test split yeah so here in this

particular example you'll be seeing that mmm let me just see where I have written printed split okay now here in train test split as you know that we will be creating some training data set and for test data set and then we use the model for training that particular data set with the with the help of training data set and with the test data set will try to check the accuracy and that is what we are doing

over here we are basically implementing SQL on load model selection import train test split and Here I am basically performing printf split on my data set that is xny okay and here you can see that I am basically specifying 30% as mantis sex with some random stream now let me just execute this in front of you now this has executed the Train test split has basically happened what I will do is that whichever classifier I used

over here basically I have basically applied easily classifier and this is after hyper parameter optimization I've got this particular parameter somebody using it and once I use it I'll just do fit on my clean and white rain and then I'll do predict and then after that I'll go and see my confusion matrix and accuracies for now here you can see that my confusion matrix is basically giving this all values my accuracy score is somewhere around 86 posit now let me

just change this random state to zero now when I change this random state to zero and I'll execute all these things and now see my previous security was somewhere around 86 percent as soon as I execute it you can see that the case is changing okay so it has become 0.866 yes I know that it is a very small number the decrease is very less I just imagine this particular data set has 10,000 records suppose I have million of

records but ending this random state it must happen in such a way that your accuracy differs a lot now in this case I have got 85 since I have used random state has 200 now with the help of this particular random state you will be observing that every time your training and your test data set is randomly selected okay so in order to prevent this you know in order to prevent this since your accuracy is always changing

based on your training and test data set the actual IQ accuracy that you need to calculate must basically using cross-validation my cross-validation first technique that we are going to basically uses k-fold cross-validation in k for cross-validation all you have to do is that input from Crossrail school so basically you have to import this library called as cross del square and it is present inside a scalar or not model solution so as soon as you input all you have to give is the first

parameter which is your model that is a classifier model so I am going to give the same classifier model whichever I have created over here then I'm going to say dot fit sorry I'm just going to assign my second parameter II that is my X which is my independent features y is basically my dependent features now here you just imagine as I'm not giving training or test data Here I am basically giving sk x and y and this

cross-validation value i am giving you tests and that basically means I will perform ten different experiments you know in in my training and in my independent and dependent feature so it will look something like this suppose this is my ten thousand records for the first experiment you'll be observing that first part will be your test dataset now you want to know how many date how many data will be there in the test dataset you it is basically

dependent on this particular value that is ten a total number of records I have is basically 10k okay ten thousand so I will just write it as 10k and if I just divide by ten right that basically means if I divide it by ten this basically answer will do 1k now 1k records should be there in your test data set at every experiment okay so for the first experiment I will get over here the first initial thousand records will be a

test the remaining nine thousand will be your train okay then in the second experiment what will happen is that the next thousand records the next thousand records will basically be your test and remaining this all records will be your dream now similarly this will happen for ten different experiments till we go till the last iteration that is still here well at last I duration will be seeing that this all data will be testing the meaning all will be three

now this cross validation now each and every experiment will give you an accuracy because I'll be training my data set on the train data and my equation will be checked for as data okay so this will be going on for ten different experiments so finally if you go and see this will create ten different accuracy scores okay so in order to once I execute it you can see that once I execute this so this has got

executed suppose it will take some time because it is basically doing ten different iterations okay now if I go and see my score this is basically my score for ten different experiments so this is one two three four five six seven eight nine obviously in when I used to change the random state in train and test so this accuracy difference was obviously getting created now how do you say a stakeholder whenever you are solving a use case that what should be

this particular critic of the model so what you can do is that he can just type score dot mean this is one way and by this you say that an average accuracy of it is exported your model is ways that you should basically see your lower accuracy now over here the lower lowest accuracy is basically your 0.85 - so you can say that your model has the lowest accuracy of 0.85 - and the maximum accuracy that it can basically give is

somewhere around 0.87 880 7.8 is how you say your stakeholders in a real world scenario whenever you are working in any company for any products for any useless that is how you basically sleep and take k4 cross-validation was pretty much simple you know so it was just going based on the number of experiments now the next thing is that we'll discuss and always remember guys if you have more number of Records this cross-validation range can be giving you

a different different kind of results which you can average it know so we can have reject and you can provide that particular case it is a stakeholder of that particular model now the next technique is something called a stratified cake k-fold cross-validation if you have seen my video about the theoretical part of stratified k-fold cross-validation I had told you that this will also be working like a full cross-validation only will be specifying here some number of experiments okay

will be specifying some number of experiments in this and in each and every experiment training and test data will be captured in such a way that for each and every class you know for each and every class there will be at least some instances it will not be a imbalanced imbalanced data set that will be picked up now suppose if I tell you you're basically examples suppose I have thousand records in my dataset and in this thousand records I have 600 years

output labels and 400 no no labels right so every time the training and test data will be picked up it will be having a good combinations of yes or no and that is it is going to do it by your own by by its own and in this particular example I'll be showing you in the coding how it is done it so let us go ahead and try to see so first of all I will import this from a scale on tour

model selection stratified Cape full I am going to create an accuracy list because this for each and every experiment have to calculate the accuracy right and always remember if we have a proper combination of both yes or no that that model will be a very good model base for you for your plan for your new test data it will basically give you a high bias so that basically will give you very good details it's so over here in stratified Kafer let me

just take this also as an underscore splits is equal to 10 this basically saying that 10 different experiments will be going on and then what as soon as I use this object SKF dot split and here I give my x and y it will return me two values one will be your training index and one will be your test index and the combination will be in such a way that amount of all good proportion lot of yes or no labels will be

classified and will be taken over there so first time I got my trained index and test index this trained index I'm going to use basically in my ex my ex independent features and I will say that ok this will become my extreme x1 train so here I have written x1 trained so that my train test which I had done on the above I should not replace that extreme value so I created a different variables now similarly X dot I lock if

I use this text index I will be getting x1 paste and similarly I should get y1 brain and y1 test so this too will be your train and Y X train and why train this will be your X test my test ok and simple it is almost simple very easy now as you know that I have to use my classifier model for training on my X train and my train right and for testing al basically used my x1

test and for calculating my creases for I will be using this y one test and that is what I am doing I'm doing that I'm saying that classified dot fit x1 underscore pain I'm a y1 underscore plane and here I've basically written classified or predict x1 underscore test now see I'm testing for my test data once I touched I get my prediction and finally by using accuracy score I am comparing prediction and y1 underscore test very simple and for each and every

score I'm appending in this accuracy list and finally let me let me just stop print it for you any different experiments will go on so this has happened if I go on if I go down completely guys I will basically get the accuracy will get if I go and see my accuracy you can see that this is my crazy right this is my crazy ten different experience so I have 10 different accuracy I do accuracy okay I mean

oops has no attribute mean why it's okay accuracy is a list so what I'll do is that I will just convert this to numb path import number as NP I'll say that NP dot array and let me just provide this particular variable is my accuracy and then I'll say dot mean now and I hope this may work okay so here you can see that list does not have a meaning function okay so I basically converted this into an array and then find out the

mean now I'm getting 86% eighty six point four six you can see if you have a good combination you'll almost get both of them are same almost same so usually you should prefer using stratified k-fold whenever you have a huge imbalance data set again imbalance if you have a imbalanced data said first of all you need to fix that image data set and that is basically fixed you know through your feature selection through a feature engineering process I already

have uploaded a lot tons of videos on that I hope you like this particular video to understand the difference between k-fold cross-validation stratified k-fold cross-validation this was all about this particular video guys I hope you know you are liking these videos make sure you subscribe the channel if you have not already subscribed please share with all your friends see you in the next video have a great day ahead god bless you
