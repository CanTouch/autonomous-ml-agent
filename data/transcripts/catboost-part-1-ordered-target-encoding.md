---
title: "CatBoost Part 1: Ordered Target Encoding"
url: "https://www.youtube.com/watch?v=KXOTSkPL2X4"
date: "2023-02-27"
video_id: "KXOTSkPL2X4"
uploader: "StatQuest with Josh Starmer"
language: "en"
---

# CatBoost Part 1: Ordered Target Encoding

ordered Target in coding going to do it for cat boost bam stat Quest hello I'm Josh starmer and welcome to stat Quest today we're going to talk about cat boost part one order Target encoding if you got a big huge cat boost model and you run it in the cloud you better use lightning bam this stack Quist is also brought to you by the letters a b and c a always b b c curious always be curious

note this stack Quest assumes you are already familiar with Target encoding if not check out the quest also note cat boost is a machine learning algorithm that at a fundamental level is very similar to gradient boost and XG boost so if you're not already familiar with those methods you might want to check out the quests in the stat Quest on one hun label and Target encoding we had this data and we wanted to use favorite color and height to

predict if someone loves Troll 2 which is a really terrible movie then we applied one hot label and Target en coding to favorite color and talked about the pros and cons of using each method the problem with basic Target encoding is that each Row's Target value the thing we want to predict is used to modify the same row value in favorite color and doing this sort of thing is a data science no no that we call leakage

leakage results in models that work great with training data but not so well with testing data so we ultimately described kfold targeting coding which splits the data into K groups to reduce leakage now if you read the original cat boost manuscript RP they point out that if we only have a single category for example if everyone loved the color blue and we apply leave one out targeting coding to the data then all of the rows with favorite

color equal to 0.33 correspond to people who love Troll 2 and all the rows with favorite color equal to 0.5 correspond to the people who do not love Troll 2 and that means any tree that starts out by splitting on favorite color less than 0.42 will classify each person in the training data set perfectly and that means we have leakage now to be honest I think it's a little silly to include a variable that only has a single category in the first

place and even if we did it is standard practice to convert features with only one or two options like Love's troll two to zeros and ones rather than use Target encoding so in this example we would just convert blue to zero rather than use Target encoding so this example of leakage seems a little silly to me because it should not happen however the creators of cat boost didn't think it was silly so they came up with a whole

new way to encode categorical features bam in fact cat boost is short for Cate orical boosting because of how Central dealing with categorical variables is to this method so let's go back to the original data set and talk about how cat boost encodes categorical features the way cat boost avoids leakage when encoding categorical variables starts with treating each row of data as if it were being fed into the algorithm sequentially for example cat boost treats the first row with blue as if it

is all the data it has received so far and that means that cat boost ignores all the other rows when Target encoding the first occurrence of blue another thing different about cat boost is the equation the big difference is that instead of using an overall mean it uses a defined prior or guess that in the examples I saw was set to 0.05 the cat boost equation also simplifies the denominator by just adding one to the number of rows rather

than a weight now given this equation cat boost plugs in values derived from all of the other data that came before the current row and that means that since we are starting with the first row and no data came before it then the option count the number of people we have seen before who at this point love blue in Troll 2 is zero and we plug in zero for n because there are no previous rows that have

blue as the favorite color and when we do the math we get 0.05 so we plug in 0.05 for blue in the first row now we work on the second row because none of the preceding rows also have red as the favorite color we set option count to zero again and again n equals 0 so we replace red in the second row with 0.05 likewise we replace Green in the third row with 0.05 however in the fourth row things

finally change now because we've seen blue before in the first row we use it to calculate the option count so in this case because the one person who liked blue before also liked Troll 2 the option count is equal to 1 and n equals 1 so we plug in 0.525 for the second time we see blue likewise the second time we see green we plug in 0.525 however the third time we see green we use the two previous times when

calculating the option count and the option count is two because both of the two previous people that liked Green also liked Troll 2 and because we've already seen two people who like green n equals 2 and that means we replace the third occurrence of green with 0.683 lastly we replace the third occurrence of blue with 0.35 because only one of the two previous times we saw blue also liked Troll 2 and thus this is how cat boost

performs Target encoding and because the order of the data makes a difference in the encoding this method is called ordered Target encoding bam lastly once you're done creating your cat boost model the entire data set is used to Target and code the new data that you want to classify note as I said earlier I'm not certain the motivation for this method is really justifiable however the important thing is that regardless of the justification cat boost works and it works well and

that's an important lesson about machine learning machine learning is all about results and doing whatever it takes to get them bam now it's time for some Shameless self-promotion if you want to review statistics and machine learning offline check out the stat Quest PDF study guides and my book the stat Quest Illustrated guide to machine learning at stack quest.org there's something for everyone hooray we've made it to the end of another exciting stack Quest if you like this stack Quest and want to see

more please subscribe and if you want to support stack Quest consider contributing to my patreon campaign becoming a channel member buying one or two of my original songs or a t-shirt or a hoodie or just donate the links are in the description below all right until next time Quest on
