import sys
sys.path.append('/home/bjorn/repos/')

from parallearn.feature_extraction.text import CountVectorizer
from parallearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime


def timeit(msg, t):
    new_t = datetime.now()
    print('{} -- {}'.format(new_t-t, msg))
    return new_t


cv_1 = TfidfVectorizer(
    analyzer='word',
    stop_words=None,
    max_df=1.0,
    min_df=0,
)

cv_2 = TfidfVectorizer(
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
with open('test_text_cicero.txt', 'r') as f:
    cicero_sents = f.readlines()[0].split('.')

timing_sents = 10000 * cicero_sents
print(len(timing_sents))

start = datetime.now()
cv_1.fit(timing_sents, n_jobs=3)
t = timeit('fit Parallel', start)
emb1 = cv_1.transform(timing_sents, n_jobs=3)
_ = timeit('transform Parallel', t)
# emb1 = cv_1.fit_transform(timing_sents, n_jobs=3)
# _ = timeit('fit_transform Parallel', start)
# print('len vocab: {}'.format(len(cv_1.vocabulary_)))


print('\n')
start = datetime.now()
cv_2.fit(timing_sents)
t = timeit('fit Sing', start)
emb2 = cv_2.transform(timing_sents)
_ = timeit('transform Sing', t)
# emb2 = cv_2.fit_transform(timing_sents)
# _ = timeit('fit_transform Sing', start)
# print('len vocab: {}'.format(len(cv_2.vocabulary_)))
# print(cv_1.fit_transform(base_sents, None, n_jobs=4).todense())


print('Vocabs equal? {}'.format(cv_2.vocabulary_ == cv_1.vocabulary_))
print('Embeddings equal? {}'.format((emb1!=emb2).nnz==0))

