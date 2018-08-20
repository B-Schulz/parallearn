from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer_non_parallel

from sklearn.feature_extraction.text import six, numbers, np, defaultdict, _make_int_array, sp
from datetime import datetime
import math
from multiprocessing import Pool

def timeit(msg, t):
    new_t = datetime.now()
    print('{} -- {}'.format(new_t-t, msg))
    return new_t


class CountVectorizer(CountVectorizer_non_parallel):
    def _vocab_worker(self, worker_number, raw_documents):
        global VOCABULARY

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

                feature_idx = vocabulary[feature]

                if feature_idx not in feature_counter:
                    feature_counter[feature_idx] = 1
                else:
                    feature_counter[feature_idx] += 1
        #
            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        return {'worker_number': worker_number,
                'vocabulary': dict(vocabulary),
                'j_indices': j_indices,
                'values': values,
                'indptr': indptr}

            # print('worker {} worked on doc {}'.format(worker_number, ix))

    def _count_vocab_parallel(self, raw_documents, fixed_vocab, n_jobs=1):
        with Pool(processes=n_jobs) as p:
            # p.apply_async(self._queue_documents, args=(raw_documents,))
            results = []
            job_size = math.ceil(len(raw_documents)/n_jobs)

            for i in range(n_jobs):
                results.append(p.apply_async(self._vocab_worker,
                                             args=(i, raw_documents[i*job_size:(i+1)*job_size])))
            p.close()
            p.join()

            vocabulary, j_indices, values, indptr = self._join_results(*[result.get() for result in results])

            j_indices = np.asarray(j_indices, dtype=np.intc)
            indptr = np.frombuffer(indptr, dtype=np.intc)
            values = np.frombuffer(values, dtype=np.intc)
            # vocabulary = dict(VOCABULARY)

            X = sp.csr_matrix((values, j_indices, indptr),
                              shape=(len(indptr) - 1, len(vocabulary)),
                              dtype=self.dtype)
            X.sort_indices()
            return vocabulary, X
            #
            # # print('worker_number: {}'.format(r['worker_number']))
            # print('vocabulary: {}'.format(VOCABULARY))
            # print('j_indices(#{}): {}'.format(len(j_indices), j_indices))
            # print('values(#{}): {}'.format(len(values), values))
            # print('indptr(#{}): {}'.format(len(indptr), indptr))

    def _join_results(self, *results):
        vocabulary = results[0]['vocabulary']
        j_indices = results[0]['j_indices']
        values = results[0]['values']
        indptr = results[0]['indptr']

        for result in results[1:]:
            mapping = {}
            for k, v in result['vocabulary'].items():
                if k not in vocabulary:
                    idx = len(vocabulary)
                    vocabulary[k] = idx
                mapping[v] = vocabulary[k]

            j_indices.extend([mapping[x] for x in result['j_indices']])
            values.extend(result['values'])
            try:
                last_val = indptr[-1]
            except IndexError:
                last_val = 0

            indptr.extend([ptr+last_val for ptr in result['indptr'][1:]])
        return vocabulary, j_indices, values, indptr

    def fit_transform_parallel(self, raw_documents, y=None, n_jobs=1):
        """Learn the vocabulary dictionary and return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        start = datetime.now()

        t = timeit('Begin...', start)

        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features
        t = timeit('Validate Vocab', t)

        vocabulary, X = self._count_vocab_parallel(raw_documents,
                                                    self.fixed_vocabulary_,
                                                   n_jobs=n_jobs)
        t = timeit('Count Vocab', t)

        if self.binary:
            X.data.fill(1)
        t = timeit('Check self.binary', t)

        if not self.fixed_vocabulary_:
            X = self._sort_features(X, vocabulary)
            t = timeit('sort features', t)

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
            t = timeit('Doc Counts', t)
            X, self.stop_words_ = self._limit_features(X, vocabulary,
                                                       max_doc_count,
                                                       min_doc_count,
                                                       max_features)
            _ = timeit('limit features', t)
            self.vocabulary_ = vocabulary
        _ = timeit('Overall timing', start)

        return X












    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        start = datetime.now()

        t = timeit('Begin...', start)

        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features
        t = timeit('Validate Vocab', t)

        vocabulary, X = self._count_vocab(raw_documents,
                                          self.fixed_vocabulary_)
        t = timeit('Count Vocab', t)

        if self.binary:
            X.data.fill(1)
        t = timeit('Check self.binary', t)

        if not self.fixed_vocabulary_:
            X = self._sort_features(X, vocabulary)
            t = timeit('sort features', t)

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
            t = timeit('Doc Counts', t)
            X, self.stop_words_ = self._limit_features(X, vocabulary,
                                                       max_doc_count,
                                                       min_doc_count,
                                                       max_features)
            _ = timeit('limit features', t)
            self.vocabulary_ = vocabulary
        _ = timeit('Overall timing', start)

        return X
