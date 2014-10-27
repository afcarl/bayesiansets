import numpy
import scipy.sparse
import warnings


class BayesianSets():
    """

    Bayesian sets for (sparse) binary data.

    :param c: Hyperparameter for the hyperprior (default c=2).

    Notes:
        There are two different variables 'c' in the paper: here distinction is
        made using c (hyperparameter) and c_value (variable).
        Uses the same hyperpriors as [Ghahramni, 2005].

    References:
        Z. Ghahramani, K.A. Heller (2005). Bayesian Sets.

    Example:
        bs = BayesianSets()
        bs.fit(np.array([[1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]))
        ix = bs.predict([[1, 1, 0])
    """
    #TODO: Use sklearn multilabel to parse labels?

    def __init__(self, c=2):
        self.c = c
        self.c_value = None
        self.X = None
        self.mean_X = None
        self.alpha = None
        self.beta = None
        self.q = None
        self.log_scores_ = None
        self.ordered_index_ = None

    def fit(self, X):
        """
        Initialize BayesianSets with data X so it's prepared for querying.

        :param X: Array-like, shape = [n_items, n_sets] with values 0, 1.
        :return: object self.
        """
        if not scipy.sparse.isspmatrix_csr(X):
            self.X = scipy.sparse.csr_matrix(X)
        else:
            self.X = X
        self.X = self._check_params()
        self.mean_X = self.X.mean(axis=0)
        self.alpha = self.c * self.mean_X
        self.beta = self.c * (1 - self.mean_X)
        return self

    def predict(self, query):
        """
        Rank items according to probability of being in query D_C.

        :param query: Boolean vector with items in the query D_C.
        :return: Item indices in the data set sorted using their scores.
        """
        #TODO: add error message if fit not called yet
        self._compute_vectors(query)
        self._compute_log_scores_()
        self._get_ordered_index()
        return self.ordered_index_

    def _check_params(self):
        """
        Check and clean matrix X (remove items present in all or no sets).

        :return: X
        """
        unique_non_zero_values = numpy.unique(self.X.data)
        if not (len(unique_non_zero_values) <= 1 and
                unique_non_zero_values == 1):
            raise ValueError('X has values that are not 0 or 1.')
        item_is_in_a_set = self.X.sum(axis=0) != 0
        item_is_in_a_set_ix = numpy.where(item_is_in_a_set)[1]
        if not numpy.all(item_is_in_a_set):
            warnings.warn("Removing items present in no sets.")
            self.X = self.X[:, item_is_in_a_set_ix.tolist()[0]]
        item_is_not_in_all_sets = self.X.sum(axis=0) != self.X.shape[0]
        item_is_not_in_all_sets_ix = numpy.where(item_is_not_in_all_sets)[1]
        if not numpy.all(item_is_not_in_all_sets):
            warnings.warn("Removing items present in all sets.")
            self.X = self.X[:, item_is_not_in_all_sets_ix.tolist()[0]]
        return self.X

    def _compute_vectors(self, item_index):
        """
        Compute vectors c and q.

        :param item_index:
        :return: None
        """
        n_indices = len(item_index)
        sum_xi = self.X[item_index, :].sum(axis=0)
        alpha_tilde = self.alpha + sum_xi
        beta_tilde = self.beta + n_indices - sum_xi
        sum_alpha_beta = self.alpha + self.beta
        self.c_value = (numpy.log(sum_alpha_beta)
                        - numpy.log(sum_alpha_beta + n_indices)
                        + numpy.log(beta_tilde) - self.beta).sum()
        self.q = ((numpy.log(alpha_tilde) - numpy.log(self.alpha)
                  - numpy.log(beta_tilde) + numpy.log(self.beta))).transpose()

    def _compute_log_scores_(self):
        """
        Compute log scores.

        :return: None
        """
        self.log_scores_ = numpy.asarray(self.c_value +
                                         self.X.dot(self.q)).flatten()

    def _get_ordered_index(self):
        """
        Order indices using the log scores.

        :return: None
        """
        #TODO: don't order scores?
        self.ordered_index_ = numpy.argsort(self.log_scores_)[::-1]
        self.log_scores_.sort()
        self.log_scores_ = self.log_scores_[::-1]