###################################################
# TASK 1: Calculate the Average Rating Based on Current Comments and Compare it with the Existing Average Rating.
###################################################

# In the shared data set, users rated a product and made comments.
# Our aim in this task is to evaluate the given points by weighting them according to date.
# It is necessary to compare the initial average score with the date-weighted score to be obtained.

import datetime as dt
import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.width', 1000)

###################################################
# Step 1: Read the Data Set and Calculate the Average Score of the Product.
###################################################
df_ = pd.read_csv("W4/measurement_problems/Rating Product&SortingReviewsinAmazon/amazon_review.csv")
df = df_.copy()
df.head()
df.isnull().sum()
df.dtypes
df.shape
df.describe().T
df["overall"].unique()

df["overall"].mean()

###################################################
# Step 2: Calculate the Weighted Score Average by Date.
###################################################


df.head()
df.info()


def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df)
time_based_weighted_average(df, 30, 26, 22, 22)
df["overall"].mean()

###################################################
# Task 2: Determine 20 Reviews to Display on the Product Detail Page for the Product.
###################################################


###################################################
# Step 1. Generate the variable helpful_no
###################################################

# Note:
# total_vote is the total number of up-downs given to a comment.
# up means helpful.
# There is no helpful_no variable in the data set, it must be generated from existing variables.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head(30)

###################################################
# Step 2. Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores and Add them to the Data
###################################################

# Up-Down Diff Score = (up ratings) − (down ratings)
def score_up_down_diff(up, down):
    return up - down
# Score = Average rating = (up ratings) / (all ratings)
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)
# Wilson Lower Bound Score
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score

     - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is considered as the WLB score.
     - The score to be calculated is used for product ranking.
     - Note:
     If the scores are between 1-5, 1-3 is marked as negative and 4-5 is marked as positive and can be adapted to Bernoulli.
     This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)

df.head(30)

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################
df.sort_values("wilson_lower_bound", ascending=False)
