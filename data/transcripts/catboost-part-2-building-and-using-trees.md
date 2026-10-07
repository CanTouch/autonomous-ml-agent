---
title: "CatBoost Part 2: Building and Using Trees"
url: "https://www.youtube.com/watch?v=3Bg2XRFOTzg"
date: "2023-03-06"
video_id: "3Bg2XRFOTzg"
uploader: "StatQuest with Josh Starmer"
language: "en"
---

# CatBoost Part 2: Building and Using Trees

Going to build a tree, one row at a time! StatQuest! Hello, I'm Josh Starmer and welcome to StatQuest. Today we're going to talk about CatBoost Part 2, Building and Using Trees. If you want to do it in the cloud, then do it with Lightning. It'll be easier and you'll thank me later. Bam!

This StatQuest is also brought to you by the letters 'A', 'B', and 'C'. 'A' always, 'B' be, 'C' curious. Always be curious. Note: This StatQuest assumes that you have already seen CatBoost Part 1, Ordered Target Encoding. If not, check out the Quest. This StatQuest also assumes that you are familiar with the cosine similarity. If not, check out the Quest. Now imagine we had this super simple data set that only uses 'Favorite Color' to predict 'Height'. We'll use this to show how CatBoost creates

trees and show how these trees are combined to make predictions. Bam. Each time CatBoost creates a tree, the first thing it does is randomize the rows of the training data set and then it applies Ordered Target Encoding to all of the discrete columns with more than two options. Note: If there are only two options than they are simply replaced with 1's and 0's. So, in this case, where we have three options, that means we apply Ordered Target Encoding to 'Favorite Color'. However, unlike in the StatQuest, on

Ordered Target Encoding, where the target, 'Loves Troll 2' was also categorical, this time the target is a continuous variable. So CatBoost puts the continuous values for 'Height' into discrete bins. In this case, since we have so little data will create two bins that are roughly the same size, and the two smallest values for 'Height' will be assigned to the first bin, 0, and the three larger values will be assigned to the second bin, 1. And now we can use the bin numbers instead of 'Height' to do

Ordered Target Encoding just like we did in CatBoost Part 1. Remember, the key thing that makes Ordered Target Encoding different from regular Target Encoding is that it pretends it is receiving the data sequentially, one row at a time. And the reason CatBoost uses Ordered Target Encoding is to avoid Leakage. Leakage can refer to a lot of things, but in this case, it refers to a row's target value having an effect on the same row's encoding. Note: If we had more data, we could end up with more

than two bins and we would end up using a slightly different equation to calculate the encoding, which, if you are just dying to know the difference, is documented on the CatBoost website. Anyway, now that we have applied Ordered Target Encoding to 'Favorite Color', we can ignore the bin numbers because we won't need them until we build the next tree. Now, CatBoost initializes the model predictions to 0 for all rows, and calculates the initial residuals, the differences between the observed

heights and the predicted heights. For example, the residual for the first row is the observed height, 1.81, minus the prediction, 0, which equals 1.81. Likewise, CatBoost calculates all of the other residuals. Bam. Now that we have residuals, we can start to build a tree. First, we need to find the best threshold for 'Favorite Color' for the root of the tree. So we start by sorting the values for 'Favorite Color' from low to high and that helps us identify two potential thresholds to try.

0.04, the value halfway between 0.025 and 0.05, and 0.29, the value halfway between 0.05 and 0.525. Note: If we had a lot more data, we'd probably have a lot more thresholds, however, CatBoost limits the maximum number of thresholds it tests by putting values close to each other in the same bin. Anyway, the first threshold we want to test is 0.04. Note: To keep this example simple, we're just going to build a stump instead of a larger tree. However, the concepts are the same for both types of trees. First,

CatBoost initializes the output of the leaves to be 0. Now we run the first row down the tree and put its residual, 1.81, in the leaf on the right because the value for 'Favorite Color', 0.05, is greater than the threshold, 0.04. Then we keep track of the output for that row, which is currently 0. Then we update the output of the leaf to be the average of the residuals in it, which in this case is 1.81. Now we run

the second row down the tree and put its residual, 1.56, in the leaf on the right and we keep track of the output, which is now 1.81, and then update the output value for the leaf to be the average of the two residuals in the leaf, 1.69. The third row goes to the leaf on the left and we keep track of the output value, which is currently 0. Then we update the output value for the left leaf to be the average

of the residuals in it, 1.77. The fourth row goes to the right and its output value is currently 1.69 before we update it to the average of the values in the leaf, 1.56. The fifth and final row goes to the right and its output value is currently 1.56 before we update it to the average of the values in the leaf, 1.58. Now we need to quantify how well this threshold, 'Favorite Color' less than 0.04, makes predictions.

CatBoost quantifies how good the predictions for each threshold are by calculating the cosine similarity between the 'Leaf Output' column and the residuals. We start by labeling the column with the residuals 'A' and the column with the leaf output values 'B'. Now computing the cosine similarity for this threshold, 'Favorite Color' less than 0.04, is just a matter of plugging the numbers into the equation. And the cosine similarity between the residuals and the leaf output values is 0.71. Thus

the score for this threshold, 'Favorite Color' less than 0.04, is 0.71. Now we need to calculate the cosine similarity for the second threshold, 'Favorite Color' less than 0.29. So we reset the output values for the leaves to be 0 and then run the data down the tree, keeping track of, and then updating, the leaf output values as we go. Beep boop. Then we calculate the cosine similarity between the residuals and the leaf output values. And the cosine similarity between the residuals and the

leaf output values is 0.79. Bam. Now, let's go back to the first tree we built, where we tested 'Favorite Color' less than 0.04, and compare its cosine similarity, 0.71, to the cosine similarity for the second tree, 0.79. Because the cosine similarity is higher for the second tree when the threshold is 'Favorite Color' less 0.29, we select the second threshold for the root of our new tree. And that is how CatBoost determines which threshold to use when building trees. And if we wanted to build

a larger tree, then we evaluate each split the same way, by comparing the cosine similarity of each possible threshold. Bam! Note: Because we started out with so little data in this example, just five rows, we used all of it to calculate the cosine similarity, and that means that we included these two rows that have leaf output equal to 0 in the calculation. Now remember, leaf output equals 0 is what we used to initialize the tree when we started to run

data down it. In other words, leaf output equals zero is not based on any data. It's just an initialization value, and it doesn't make a whole lot of sense to include leaf output values that are not based on data in the cosine similarity calculation. So, in practice, when you have a lot of data, CatBoost simply ignores the first bunch of rows when calculating the cosine similarity. Small bam. Now that we've built the first tree, we need to build the next tree. So we go back to the data and blank out

the residuals and update the predictions by adding the leaf output values, scaled by a learning rate, to them. In this case, we'll set the learning rate to 0.1. So, for the first row, the new prediction is the most recent prediction, 0, plus the learning rate, 0.1, times the leaf output, 0. So we get 0 as the new prediction.

Now we blank the current prediction and leaf output in the table and update the prediction with the new prediction, 0. Now we do the same thing to update all of the other predictions. Dooba dooba dooba dooba dooba dooba dooba do. Note: Compared to the known height values, the predictions made by this tree are not very good, but they are better than what we started out with when we predicted everyone head height equal to 0.

So we're making progress, we just need to make more. Now, before we move on, I want to make sure we understand one of CatBoost's main ideas. The main idea of how CatBoost calculates the output values for trees is that, just like for Ordered Target Encoding, it treats the data as if it was receiving it sequentially. This means that the residual in a row is not part of the calculation of the leaf output for the same row, and

thus the residual in a row is not, in any way, part of the calculation of the prediction for that same row. And this avoids Leakage, just like it does for Ordered Target Encoding. Bam! So, now that we've updated the predictions, we can update the residuals by subtracting the predicted values from the observed. Then we replace the values for 'Favorite Color' with their original color names, 'Green', 'Blue', and 'Red'.

Then, just like we did earlier, we randomize the rows. And now, do you remember the bin numbers we assigned each row? Well, we'll use them again to apply Ordered Target Encoding to 'Favorite Color'. Now we can hide the 'Bin Number' column since we won't need it to build a tree, and then sort the 'Favorite Color' column to identify thresholds for the tree. And, in this case, there's only one threshold, 0.29, so there is only one tree to build. Then we run the data down the tree, keeping track of, and then

updating, the leaf output values as we go. Bam! Now we blank out the residuals and update the predictions, and again, the predictions are not awesome, but they are better than what we had before. So we're still heading in the right direction. So, in practice, we would just keep making trees, possibly hundreds of trees, and updating the predictions until they are good. Double bam! However, in the interest of time, let's imagine that we only have the two trees we've created so far, and we get new

data and want to predict 'Height'. The first thing we need to do is encode 'Blue' as a number. So we go back to the original training data set with the bin numbers and we use all of the rows that have the same 'Favorite Color', 'Blue', to calculate the target encoding. Now we just run the data down the trees and add up the output values from the leaves times the learning rate, 0.1, to get a prediction, 0.3. And

yes, this is a terrible prediction, but it's with only two trees and we understand the process. So bam! Now that we know how to build tiny trees and make predictions, imagine we had more data and more variables and we wanted to build a larger tree. When CatBoost builds larger trees, it builds oblivious or symmetric decision trees. A symmetric decision tree uses the exact same threshold for each node in the same level. In this example tree, that means that both nodes on the second level use the same threshold, 'Age'

less than 12. CatBoost uses symmetric decision trees for two reasons. The first reason may sound a little crazy, it makes the trees worse at making predictions. However, remember, the whole idea of Gradient Boosting is to combine a bunch of weak learners to make decisions, and symmetric decision trees are just a weaker type of learner.

Bam. The second reason is because symmetric decision trees can make predictions faster than normal trees. To see why this is, let's consider a normal decision tree with two different thresholds on the second level. Now, if we wanted to run this data down the three to predict their 'Height', then we would start by seeing if 'Favorite Color' is less than 0.87. And if 'Favorite Color' is less than 0.87, then we would ask about 'Age'. Otherwise, we ask about 'Movie'. Keeping track of what question

we are asking takes time. In contrast, when each level asks the exact same questions, we don't have to keep track of which ones to ask, because regardless of which way we go, we always ask the same questions. In technical terms, regardless of what path we need to take through the tree. Symmetric trees are relatively fast because they can ask all of the questions in a single vector operation. Bam. In summary, the main things that make CatBoost different from other Gradient Boosting methods are, in

order to avoid Leakage, CatBoost treats the data as if it were arriving sequentially, one row at a time, when doing Target Encoding and when calculating the output values from the trees. Beep, boop, beep. And when CatBoost creates large trees, it creates symmetrical trees, because they're weaker and their output values can be computed relatively quickly. Triple Bam! And now it's time for some shameless self-promotion.

If you want to review statistics and machine learning offline, check out the StatQuest PDF study guides and my book, The StatQuest Illustrated Guide to Machine Learning, at statquest.org. There's something for everyone! Hooray! We've made it to the end of another exciting StatQuest. If you liked this StatQuest and want to see more, please subscribe. And if you want to support StatQuest, consider contributing to my patreon campaign, becoming a channel member, buying one or two of my original

songs or a t-shirt or a hoodie or just donate. The links are in the description below. All right, until next time quest on!
