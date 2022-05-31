# Climate-Change-Sentiment-Predictor

This project aims to develop a program that, given a tweet relating to climate change, can predict whether the user's sentiment is sympathetic regarding climate change or not. Using data from a dataset of tweets with a manually labeled sentiment to train from, we aim to help future analysis regarding climate change through this program that looks into human sentiments through tweets. By providing a program that automatically labels sentiments, this project can easily provide future research with numerous and timely data to use.

This project is developed by:
- Alvarado, Enrique Luis
- Bassig, Lance Raphael
- Roxas, Dennis Christian
- Surara, Ron Christian

## Definition of Terms

1. **Sympathetic**: _Adjective describing someone who talks and acts for a particular topic/sentiment: in this study, someone who is aware of, and/or acts in a way that makes others be positively aware of, climate change._
2. **"Positive"**: _Label of climate change tweets that attempt to bring positive awareness to climate change and their causes through information or persuation. In essence, this label is attached to tweets aligned with the sentiment that humans are the cause of climate change and that it is an urgent issue. This label also includes news on climate change._
    > Indiscriminately bang nilabel ang true and fake news under this label?
3. **"Negative"**: _Label of climate change tweets that negatively bring awareness to climate change and their causes, e.g. deniers of climate change or stating that the issue is out of our control thus dissuading action._
    > Ininclude ba ang fake news sa label na ito?
4. **"Neutral"**: _Label of climate change tweets that simply express awareness of the issue but not attempt to inform, persuade nor dissuade people from it are labeled "neutral"._

## Problem Statement

This project aims to lay the groundwork towards answering the following question: "How sympathetic can we expect the general populace to be with regards to climate change?"

In particular, the research group seeks to do the following:
1. Perform Exploratory Data Analysis (EDA) on the composition and word occurences of the data set from which the machine will learn from,
2. Choose the best Logistic Regression model to use,
3. Utilize the learned machine to label climate change related tweets from May 2016 to May 2017, and from May 2021 to May 2022.
4. Perform EDA on the composition and word occurences of the resulting labeled data, and
5. Determine if there is a significant change in sentiment on climate change over the past half decade.

The null hypothesis is determined to be that "there is no significant change in the twitter sentiment on climate change from 2016-2017 and that of 2021-2022."

The alternative hypothesis is determined to be that "there is a significant change in the twitter sentiment on climate change from 2016-2017 and that of 2021-2022."

<!--
In order to make the first steps towards fully answering the main question, we first need a way to review the data needed to construct a satisfying answer. As of May 2022, 500,000,000 tweets are posted every day (Internet Live Stats, 2022), and even if less that 1 percent of that is related to climate change, a small group of humans will take a while reviewing hundreds of thousands of data, so we need to have a machine that assists us in our endeavor.
-->

## Methodology

The machine that we have developd was be made to learn how to classify climate-change related tweets to the following categories: "positive", "negative", and "neutral". (See the definition of terms for what each of the labels mean)

Each tweet in the learning data that was used by the machine was individually labeled by the research group according to their stance and sentiment on climate change; the Climate Sentiment on Twitter (Guzman, 2020) dataset was reviewed for this purpose. This dataset is a raw database of 396 tweets from January 2020 to December 2020. The data from this dataset is just large and recent enough to be utilized for this project.

The data was preprocessed by adding a new `Sympathy?` column that indicates whether a tweet is sympathetic to climate change or not; then the group manually added the appropriate value for each tweet ("Yes" if the tweet is sympathetic to climate change, "No" otherwise). Instances of null rows were removed, and `TfidfVectorizer` was then used to convert text data to numeric data. The tweet content will be cleaned of its URLs, hashtags, mentions, emojis, smileys, and stop words (e.g. "a", "the", "this" , "amp", which is the HTML code of the ampersand symbol, `&amp`); this is so that the machine can better process and properly learn the necessary information from the dataset. Exploratory data analyses on the sentiment composition and word occurences were also performed on this dataset to have an idea of what the machine learns from the tweets.

The Logistic Regression model, particularly the Limited-Memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS) solver, will be used in this project. According to a prior analysis (Kim, 2022) on a similar dataset (Qian, 2019), this model has been observed to perform best particularly in distinguishing tweets on climate change to appropriate sentiment labels. Due to the relatively low volume of the dataset, the model can achieve a maximum of only ~73% accuracy.

After the machine has been developed, tweets related to climate change was scalped from twitter and fed to the machine. The resulting labeled data was then explored in order to answer the main problem faced by this project.

Specifically, 1059 tweets related to climate change between May 2016 to May 2017 and 1002 tweets related to climate change between May 2021 to May 2022 were then scalped from twitter for the machine to label. Exploratory data analysis is performed on the resulting labeled data to have an idea on what the sentiments of the tweets are like.

Finally, a chi-squared test is performed on the sentiment counts of the results of the two datasets labeled by the machine to determine whether to reject the null hypothesis or not, with the `alpha = 1 - model accuracy`.

## Results and Discussion



## Conclusion

## References

(2022) Twitter Usage Statistics. _Internet Live Stats._ Archived 2022, May 7, 5:55 PM GMT: https://web.archive.org/web/20220507054131/https://www.internetlivestats.com/twitter-statistics/

Guzman, J. (2020). 2020 Climate Sentiment on Twitter. Kaggle. Retrieved from https://www.kaggle.com/datasets/joseguzman/climate-sentiment-in-twitter?resource=download

Qian, E. (2019, November 13). Twitter Climate Change Sentiment Dataset. Kaggle. Retrieved from https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset

Kim, R. (2022, April 5). Twitter Climate Change Analysis. Kaggle. Retrieved from https://www.kaggle.com/code/roellekim/twitter-climate-change-sentiment-analysis

## Unreviewed

Patronella, A. M. (2021). Covering Climate Change: A Sentiment Analysis of Major Newspaper Articles from 2010 - 2020. _Inquiries Journal, 13(9)_. Archived 2022, May 11, 11:43 AM GMT: https://web.archive.org/web/20220511114350/http://www.inquiriesjournal.com/articles/1910/covering-climate-change-a-sentiment-analysis-of-major-newspaper-articles-from-2010--2020