---
title: "StatQuest: Decision Trees, Part 2 - Feature Selection and Missing Data"
url: "https://www.youtube.com/watch?v=wpNl-JwwplA"
date: "2018-01-29"
video_id: "wpNl-JwwplA"
uploader: "StatQuest with Josh Starmer"
language: "en"
---

# StatQuest: Decision Trees, Part 2 - Feature Selection and Missing Data

when you've got too much data don't freak [Music] out when you've got missing data don't freak out you've got stat Quest hello I'm Josh starmer and welcome to stack Quest today we're going to be talking about decision trees part two feature selection and missing data this is just a short and sweet stack quest to touch on a few topic ICS we didn't get to in the original stat Quest on decision trees in the first stat Quest on

decision trees we started with a table of data and built a decision tree that gave us a sense of How likely a patient might have heart disease if they have other symptoms we first asked if a patient had good blood circulation if so we then asked if they had blocked arteries and if so we then asked if they had chest pain if so there's a good chance that they have heart disease since 17 people with similar answers did and only three

people with similar answers did not if they don't have chest pain there's a good chance that they do not have heart disease however remember that if someone had good circulation and did not have blocked arteries we did not ask about chest pain because there was less impurity in our result s if we didn't in other words we calculated the impurity after separating the patients using chest pain then we calculated impurity without separating and since the impurity was

lower when we didn't separate we made it a leaf node now imagine if chest pain never gave us a reduction in impurity score if this were the case we would never use chest pain to separate the patients and chest pain would not be part of our tree now even though we have data for chest pain it is not part of our tree anymore all that's left are good circulation and blocked arteries this is a type of automatic

feature selection however we could also have created a threshold such that the impurity reduction has to be large enough to make a big difference as a result we end up with simpler trees that are not overfit oh no some jargon just snuck up on us overfit means our tree does well with the original data the data we used to make the tree but doesn't do well with any other data set decision trees have the downside of often being

overfit requiring each split to make a large reduction in impurity helps a tree from being overfit so in a nutshell that's what feature selection is all about now let's talk about missing data in the first video on decision trees we calculated impurity for blocked arteries and we skipped this patient since we didn't know if they had blocked arteries or not but it doesn't have to be that way we could pick the most common option if overall yes occurred more

times than no we could put yes here alternatively we could find another column that has the highest correlation with blocked arteries and use that as a guide in this case chest pain and blocked arteries are often very similar the first patient has no in both categories the second patient has yes in both categories the third patient has no in both categories and so for the fourth patient since chest pain is yes we'll make blocked arteries yes as

well now imagine we had weight data instead of blocked artery data we could replace this missing value with the mean or the median alternatively we could find another column that has the highest correlation with weight in this case height is highly correlated with weight and then we can do a linear regression on the two columns and use the least squares line to predict the value for weight so you can see that if we're missing some data there are a lot of

ways to guess at what it might be bam hooray we've made it to the end of another exciting stack Quest if you like this stack Quest and want to see more like it please subscribe and if you have any ideas for future stack quests well put them in the comments below until next time queston
