import sys
sys.path.append('/home/bjorn/repos/')

from parallearn.feature_extraction.text import CountVectorizer, timeit, datetime

cv_1 = CountVectorizer(
    analyzer='word',
    stop_words=None,
    max_df=1.0,
    min_df=0,
)

cv_2 = CountVectorizer(
    analyzer='word',
    stop_words=None,
    max_df=1.0,
    min_df=0,
)

base_sents = ['dies dies ist ein satz satz satz',
         'dies ist noch noch ein satz',
         'ist dies noch ein satz',
         'noch ein ein ganz anderer ein satz']

# timing_sents = 100000 * base_sents
# with open('test_text_cicero.txt', 'r') as f:
#     cicero_sents = f.readlines()[0].split('.')
#
# timing_sents = 1000 * cicero_sents
# print(len(timing_sents))
#
# start = datetime.now()
# _ = cv_1.fit_transform_parallel(timing_sents, None, n_jobs=2)
# _ = timeit('fit_transform Parallel', start)
# print('len vocab: {}'.format(len(cv_1.vocabulary_)))
#
# print('\n\n')
# start = datetime.now()
# _ = cv_2.fit_transform(timing_sents, None)
# _ = timeit('fit_transform Sing', start)
# print('len vocab: {}'.format(len(cv_2.vocabulary_)))
print(cv_1.fit_transform_parallel(base_sents, None, n_jobs=4).todense())
