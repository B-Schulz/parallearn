import sys
sys.path.append('/home/bjorn/repos/')

from parallearn.feature_extraction.text import CountVectorizer

# from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(
    analyzer='word',
    stop_words=None,
    max_df=1.0,
    min_df=0,
)

base_sents = ['dies dies ist ein satz satz satz',
         'dies ist noch noch ein satz',
         'ist dies noch ein satz',
         'noch ein ein ganz anderer ein satz']

timing_sents = 100000 * base_sents

# transformed_base_sents = cv.fit_transform(base_sents)

# print(transformed_base_sents.todense())

_ = cv.fit_transform(timing_sents)
