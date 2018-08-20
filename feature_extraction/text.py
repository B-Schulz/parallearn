import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import six, numbers, np, defaultdict, _make_int_array, sp, check_is_fitted
import math
from multiprocessing import Pool


class CountVectorizer(sklearn.feature_extraction.text.CountVectorizer):
    def _vocab_worker(self, worker_number, raw_documents, fixed_vocab):
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = _make_int_array()
        values = _make_int_array()
        indptr.append(0)

        for doc in raw_documents:
            feature_counter = {}
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        return {'worker_number': worker_number,
                'fixed_vocab': fixed_vocab,
                'vocabulary': dict(vocabulary),
                'j_indices': j_indices,
                'values': values,
                'indptr': indptr}

    def _count_vocab(self, raw_documents, fixed_vocab, n_jobs=1):
        with Pool(processes=n_jobs) as p:
            results = []
            job_size = math.ceil(len(raw_documents)/n_jobs)

            for i in range(n_jobs):
                results.append(p.apply_async(self._vocab_worker,
                                             args=(i, raw_documents[i*job_size:(i+1)*job_size], fixed_vocab)))
            p.close()
            p.join()

            vocabulary, j_indices, values, indptr = self._join_results(*[result.get() for result in results])

            j_indices = np.asarray(j_indices, dtype=np.intc)
            indptr = np.frombuffer(indptr, dtype=np.intc)
            values = np.frombuffer(values, dtype=np.intc)


            X = sp.csr_matrix((values, j_indices, indptr),
                              shape=(len(indptr) - 1, len(vocabulary)),
                              dtype=self.dtype)
            X.sort_indices()
            return vocabulary, X

    @staticmethod
    def _join_results(*results):
        fixed_vocab = results[0]['fixed_vocab']

        vocabulary = results[0]['vocabulary']
        j_indices = results[0]['j_indices']
        values = results[0]['values']
        indptr = results[0]['indptr']

        for result in results[1:]:

            # if subprocesses buit their own vocabularies,
            # mapping between them must be calculated and applied

            if not fixed_vocab:
                mapping = {}
                for k, v in result['vocabulary'].items():
                    if k not in vocabulary:
                        idx = len(vocabulary)
                        vocabulary[k] = idx
                    mapping[v] = vocabulary[k]
                j_indices.extend([mapping[x] for x in result['j_indices']])
            else:
                j_indices.extend(result['j_indices'])

            values.extend(result['values'])
            try:
                last_val = indptr[-1]
            except IndexError:
                last_val = 0

            indptr.extend([ptr+last_val for ptr in result['indptr'][1:]])
        return vocabulary, j_indices, values, indptr

    def transform(self, raw_documents, n_jobs=1):
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()

        self._check_vocabulary()

        # use the same matrix-building strategy as fit_transform
        _, X = self._count_vocab(raw_documents, fixed_vocab=True, n_jobs=n_jobs)

        if self.binary:
            X.data.fill(1)
        return X

    def fit(self, raw_documents, y=None, n_jobs=1):
        self.fit_transform(raw_documents, n_jobs=n_jobs)
        return self

    def fit_transform(self, raw_documents, y=None, n_jobs=1):
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, X = self._count_vocab(raw_documents,
                                          self.fixed_vocabulary_,
                                          n_jobs=n_jobs)
        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary_:
            X = self._sort_features(X, vocabulary)

            n_doc = X.shape[0]
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * n_doc)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            X, self.stop_words_ = self._limit_features(X, vocabulary,
                                                       max_doc_count,
                                                       min_doc_count,
                                                       max_features)
            self.vocabulary_ = vocabulary

        return X


class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer, CountVectorizer):

    def fit(self, raw_documents, y=None, n_jobs=1):
        X = CountVectorizer.fit_transform(self, raw_documents, n_jobs=n_jobs)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None, n_jobs=1):
        X = CountVectorizer.fit_transform(self, raw_documents, n_jobs=n_jobs)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents, copy=True, n_jobs=1):
        check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')

        X = CountVectorizer.transform(self, raw_documents, n_jobs=n_jobs)
        return self._tfidf.transform(X, copy=False)
