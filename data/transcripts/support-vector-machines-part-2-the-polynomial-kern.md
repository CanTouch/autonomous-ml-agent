---
title: "Support Vector Machines Part 2: The Polynomial Kernel (Part 2 of 3)"
url: "https://www.youtube.com/watch?v=Toet3EiSFcM"
date: "2019-11-04"
video_id: "Toet3EiSFcM"
uploader: "StatQuest with Josh Starmer"
language: "en"
---

# Support Vector Machines Part 2: The Polynomial Kernel (Part 2 of 3)

I once knew a kernel, its name was Fred. This StatQuest isn't about that kernel. StatQuest. Hello, I'm Josh Starmer and welcome to StatQuest. Today, we're going to talk about support vector machines part two, the polynomial kernel. Specifically, we're going to talk about the polynomial kernel's parameters and how the polynomial kernel calculates high-dimensional relationships.

Note, this StatQuest assumes that you are already familiar with support vector machines. If not, check out the quest. The link is in the description below. In the StatQuest on support vector machines, we had a training data set based on drug dosages measured in a bunch of patients. The red dots represented patients that were not cured.

And the green dots represented patients that were cured. In other words, the drug doesn't work if the dosage is too small or too large. It only works when the dosage is just right. Because this training data set had so much overlap, we were unable to find a satisfying support vector classifier to separate the patients that were cured from the patients that were not cured.

However, when we gave each point a Y axis coordinate by squaring the original dosage measurements, we could draw a line that separated the two categories of patients. So, we used a support vector machine with a polynomial kernel to compute the relationships between the observations in a higher dimension.

And then found a good support vector classifier based on the high-dimensional relationships. The polynomial kernel that I used looks like this. A and B refer to two different observations in the data set. R determines the coefficient of the polynomial. And like I mentioned in the earlier StatQuest, D sets the degree of the polynomial.

In my example, I set R = 1/2 and D = 2. Since we are squaring the term, we can expand it to be the product of two terms. Now we just do the multiplication. And combine these two terms. And just because it will make things look better later, let's flip the order of these two terms.

Finally, this polynomial is equal to this dot product. A dot product sounds fancy. But all it is is the first terms multiplied together plus the second terms multiplied together plus the third terms multiplied together. The dot product gives us the high-dimensional coordinates for the data. The first terms are the X axis coordinates.

And the second terms are the Y axis coordinates. The third terms are Z axis coordinates. But since they are the same for both points, we can ignore them. Thus, we have X and Y axis coordinates for the data in the higher dimension. Bam! Alternatively, we could have set R = 1 and D = 2.

Now when we do the math, we get this polynomial and this dot product. We can verify that the dot product is correct by multiplying each term together and then add everything up and the result should be equal to the polynomial. Using this dot product, the new x-axis coordinates are the square root of two times the original dosage values.

So, we move the points on the x-axis over by a factor of the square root of two. The new y-axis coordinates are the same as before, the original dosage values squared. And just like before, we can ignore the z-axis coordinate since it is a constant value. Now, just like before, we can use the high dimensional relationships to find a support vector classifier.

Double bam! Now, brace yourself. Things are about to get a little crazy. Going back to the polynomial kernel with r equals 1/2 and d equals two, it turns out that all we need to do to calculate the high dimensional relationships is calculate the dot products between each pair of points.

And since this kernel is equal to this dot product, all we need to do is plug values into the kernel to get the high dimensional relationships. For example, if we wanted to know the high dimensional relationships between these two observations, then we plug the dosages into the kernel, do the math, and 16,002.25 is one of the two dimensional relationships that we need to solve for the support vector classifier, even though we didn't actually transform the data to two dimensions.

Triple bam! Unfortunately, why we only need to compute the dot product is out of the scope of this StatQuest. Womp womp. To review, the polynomial kernel computes relationships between pairs of observations. A and B refer to the two observations that we want to calculate the high dimensional relationships for.

R determines the polynomial's coefficient. And D determines the degree of the polynomial. Note, R and D are determined using cross-validation. Once we decide on values for R and D, we just plug in the observations and do the math to get the high dimensional relationships. Hooray! We've made it to the end of another exciting StatQuest. If you like this StatQuest and want to see more, please subscribe. And if you want to support StatQuest, consider contributing to my Patreon campaign, becoming a

channel member, buying one or two of my original songs, or a t-shirt, or a hoodie, or just donate. The links are in the description below. All right, until next time, quest on.
