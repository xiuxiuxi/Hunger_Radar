import pandas as pd
import nltk
import re
import os
import time
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split


PROJECT_DIR = os.path.dirname(os.path.abspath("__file__"))

JSON_PATH =  os.path.join(PROJECT_DIR, "data/yelp_academic_dataset_review.json")
LABEL_DATA_PATH = os.path.join(PROJECT_DIR, "data/labeled_review.csv")

BUSINESS_PATH = os.path.join(PROJECT_DIR, "data/results/MA_Res_business_id.csv")
CLEANED_REVIEW_PATH = os.path.join(PROJECT_DIR, "data/results/review_sentiment_full.csv")
TRAINING_DATA_PATH = os.path.join(PROJECT_DIR, "data/results/clean_labeled_review.csv")
OUTSAMPLE_DATA_PATH = os.path.join(PROJECT_DIR, "data/results/trained_labeled_review.csv")
POLARITY_PATH = os.path.join(PROJECT_DIR, "data/results/business_polarity.csv")

############################################################################
# Data Importing


def clean_labeled_data():

    df_label = pd.read_csv(LABEL_DATA_PATH, error_bad_lines=False, engine="python")
    df_label = df_label[['reviewID', 'reviewerID', 'reviewContent', 'usefulCount', 'rating', 'flagged', 'restaurantID']]
    df_label = df_label.loc[df_label.flagged.isin(['Y', 'N']), :]
    df_label.dropna(inplace=True)
    df_label.rename(columns={
        'reviewID': 'review_id',
        'reviewerID': 'user_id',
        'reviewContent': 'text',
        'usefulCount': 'useful',
        'rating': 'stars',
        'restaurantID': 'business_id'
    }, inplace=True)
    df_label.to_csv(TRAINING_DATA_PATH, sep=',', index=False)


def import_data():
    # JSON files downloaded from Yelp Open DataSet. REPLACE IT WITH YOUR OWN DATA FILE PATH
    review_json_path = JSON_PATH
    # cleaned business data for restaurants in MA
    ma_res_id_path = BUSINESS_PATH
    # A JsonReader with chunk size of 1000
    df_review = pd.read_json(review_json_path, lines=True, chunksize=5000)
    business_filtered = pd.read_csv(ma_res_id_path)

    return df_review, business_filtered

# END
############################################################################


############################################################################
# NLP utils func

# There are two methods could be used:
# Method 1: Turn long sentences into tokens. With tokens, could be used with any pre-trained Bag-of-Words dictionary
#           or even word-embedded model.
#           For sentiment analysis, could count positive words and negative words to obtain a user-defined score.
#
# Method 2: Keep the long sentences, and let the package TextBlob to do everything behind the scene. Simple but may not
#           as effective as method 1


# ---------------------Method 1 related -------------------------------------

REMOVE_PHRASES = ['']


def decontracted(phrase: str) -> str:
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocessing(text: str, threshold=None) -> List[str]:
    # step 1: lowercase
    # step 2: de-contracted in case there is 'll, 's left
    # step 3: remove self-defined meaningless phrases
    # step 4: tokenize the inputs (review text: a long string) into a list of words
    # step 5: remove stop words
    # step 6: remove all non-alphanumeric characters and numbers
    # step 7: lemmatizer

    text = text.lower()
    text = decontracted(text)
    stop_words = set(stopwords.words('english') + ['restaurant', 'restaurants', 'food', 'foods'])

    # for word in REMOVE_PHRASES:
    #    text = text.replace(word, '')

    lemmatizer = WordNetLemmatizer()
    tokens = [word for word in nltk.word_tokenize(text)]
    filtered_tokens = []
    tokens = [i for i in tokens if i not in stop_words]
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [lemmatizer.lemmatize(ft) for ft in filtered_tokens]

    # optional: if the length of stems less than a certain number, consider this review meaningless
    if threshold is not None:
        return stems if len(stems) >= 3 else None
    else:
        return stems


def text_cleaning(text: str) -> str:
    """
    :param text: raw reviews
    :return: a long string
    Make text lowercase, remove text in square brackets,remove links,remove special characters
    and remove words containing numbers.
    """

    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# ---------------------Method 2 related -------------------------------------
def get_sentiment_score_textblob(review):
    s = TextBlob(review)
    return s.sentiment.polarity

# END
############################################################################


############################################################################
# NLP analysis

# Would use method 2 from previous section first, then after finishing processing all reviews data, calc correlation
# with yelp star, if the value is low, may need to consider method 1 as well.

# take the first 1000 lines as a test sample

def filter_and_score_reviews():
    df_review, business_filtered = import_data()
    reviews_filtered = pd.DataFrame()
    start_time = time.time()

    for chunk in df_review:
        chunk = chunk.drop(['cool', 'date', 'funny'], axis=1)
        chunk = pd.merge(chunk, business_filtered, how='inner', on='business_id')
        chunk = chunk.dropna(subset=['text'])
        if chunk.empty:
            pass
        else:
            chunk['polarity'] = chunk.text.apply(get_sentiment_score_textblob)*2+3
            reviews_filtered = reviews_filtered.append(chunk)
    print("Reviews filtering & Score calculating finished. Total time taken: {} s".format(time.time()-start_time))
    reviews_filtered.to_csv(CLEANED_REVIEW_PATH, sep=',', index=False)
    # reviews_filtered[['stars', 'polarity']].corr(method='pearson', min_periods=1)


def feature_extraction(df, labeled=True):
    """
    :param labeled:
    :param df: review data
    :return: features extracted from review data for classification
    """
    if labeled:
        df['flagged'] = df['flagged'] == 'Y'
        df['flagged'] = df['flagged'].astype(int)

    # Review Centric
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['word_density'] = df['char_count'] / (df['word_count']+1)
    df['punctuation_count'] = df['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    df['title_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    df['upper_case_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    # Reviewer Centric
    df['user_id_no_of_review'] = df.groupby('user_id')['user_id'].transform('size')
    df['user_id_ave_rating'] = df.groupby('user_id')['stars'].transform('mean')
    df['user_id_ave_no_words'] = df.groupby('user_id')['word_count'].transform('mean')
    df['user_id_avg_of_useful'] = df.groupby('user_id')['useful'].transform('mean')

    # Business (restaurant) Centric
    df['busi_no_of_review'] = df.groupby('business_id')['business_id'].transform('size')
    df['busi_id_ave_rating'] = df.groupby('business_id')['stars'].transform('mean')
    df['busi_id_ave_no_words'] = df.groupby('business_id')['word_count'].transform('mean')

    features = ['char_count', 'word_count', 'word_density', 'punctuation_count', 'title_word_count',
                'upper_case_word_count', 'user_id_no_of_review', 'user_id_ave_rating', 'user_id_ave_no_words',
                'user_id_avg_of_useful', 'busi_no_of_review', 'busi_id_ave_rating', 'busi_id_ave_no_words']

    # Rescaling the features for faster convergence
    df[features] = df[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    if labeled:
        return df[['review_id'] + features], df['flagged']
    else:
        return df[['review_id'] + features]


def plotROC(mtd, FP_rate, TP_rate, roc_auc):
    plt.title(mtd +' :ROC curve')
    plt.plot(FP_rate, TP_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.xlim([-0.1, 1.05])
    plt.ylim([-0.1, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def hyperParameterTuning(rf_clf, x_train, y_train):
    param_grid_rf = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16]}
    gscv_rfc = GridSearchCV(estimator=rf_clf, param_grid=param_grid_rf, cv=5, n_jobs=-1)
    gscv_rfc.fit(x_train, y_train)
    # -------------------------------
    return gscv_rfc


def train_classifier():
    training_data = pd.read_csv(TRAINING_DATA_PATH)
    X, y = feature_extraction(training_data)
    xTrain, xTest, yTrain, yTest = train_test_split(X.iloc[:, 1:], y, test_size=0.3, random_state=2021)
    clf = RandomForestClassifier()
    gscv_rfc = hyperParameterTuning(clf, xTrain, yTrain)
    gscv_rfc.fit(xTrain, yTrain)
    yPred = gscv_rfc.predict(xTest)
    y_pred_prob = gscv_rfc.predict_proba(xTest)[:, 1]
    accuracy = metrics.accuracy_score(yTest, yPred)
    precision = metrics.precision_score(yTest, yPred)
    recall = metrics.recall_score(yTest, yPred)
    f1score = metrics.f1_score(yTest, yPred)
    FP_rate, TP_rate, thresholds = metrics.roc_curve(yTest, y_pred_prob)
    roc_auc = metrics.auc(FP_rate, TP_rate)
    plotROC('Random Forest', FP_rate, TP_rate, roc_auc)
    return gscv_rfc


def fit_outsample():
    outsample_data = pd.read_csv(CLEANED_REVIEW_PATH)
    X = feature_extraction(outsample_data, labeled=False)
    gscv_rfc = train_classifier()
    outsample = pd.DataFrame(X[['review_id']], columns=['review_id'])
    outsample['label'] = gscv_rfc.predict(X.iloc[:, 1:])
    outsample = pd.merge(outsample, outsample_data[['business_id', 'review_id', 'polarity']], on='review_id')
    del outsample_data
    del X
    outsample.to_csv(OUTSAMPLE_DATA_PATH, sep=',', index=False)


def weighted_polarity(b, df):
    cnt_fake = sum(df.label)
    cnt_auth = df.shape[0] - cnt_fake
    fake_rate = cnt_fake / (cnt_fake + cnt_auth)
    if cnt_auth == 0:
        # if all reviews are predicted as fake, deduct one point, if less than 1, round to 1
        polarity = df.polarity.mean()
        if polarity <= 2:
            polarity = 1
        else:
            polarity = polarity - 1
    if cnt_fake == 0:
        # if all reviews are predicted as authentic,
        polarity = df.polarity.mean()
    else:
        grouped_polarity = df.groupby('label')['polarity'].mean()
        # lower the weight of fake reviews to 10% of its original weight
        polarity = grouped_polarity[1] * 0.1*fake_rate + grouped_polarity[0]*(1-0.1*fake_rate)
    result = pd.DataFrame(columns=['business_id', 'polarity'])
    result.loc[0] = [b, polarity]
    return result


def score_business():
    outsample = pd.read_csv(OUTSAMPLE_DATA_PATH)
    business_list = outsample.business_id.unique()
    df_polarity = pd.DataFrame(columns=['business_id', 'polarity'])
    for b in business_list:
        df = outsample.loc[outsample.business_id == b, :]
        df_polarity = df_polarity.append(weighted_polarity(b, df))

    df_polarity.to_csv(POLARITY_PATH, index=False)


if __name__ == "__main__":
    clean_labeled_data()
    filter_and_score_reviews()
    fit_outsample()
    score_business()




