from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from os.path import join
import numpy as np
import fasttext
import os
from functools import wraps
from fasttext import FastText

saved_args = None


class FastTextClassifier(BaseEstimator):

    def __saver_decorator(f):
        @wraps(f)
        def wrapper(*args, **kwds):
            global saved_args
            saved_args = kwds
            return f(*args, **kwds)

        return wrapper

    @__saver_decorator
    def __init__(
            self,
            path_to_tempfiles='.',
            reset_data_files=True,
            remove_summary=True,
            verbose: int = 0,
            model: str = None, lr: float = None, dim: int = None, ws: int = None, epoch: int = None,
            minCount: int = None, minCountLabel: int = None, minn: int = None, maxn: int = None, neg: int = None,
            wordNgrams: int = None, loss="softmax", bucket: int = None, thread: int = None, lrUpdateRate: int = None,
            t: float = None, label: str = None, pretrainedVectors: str = None):
        """
        fastText model initialization.

        :param path_to_tempfiles: folder for temp files (txt with data for training).
        :param reset_data_files: replace old temp files with same name during fitting.
        :param remove_summary: delete temp files after training.
        :param verbose: print additional info (also affect fastText verpose parameter).

        :param OTHERS: fastText model parameters.
        """
        # Saving user keyword arguments using global variable.
        global saved_args
        self.kwargs = saved_args

        # Additional parameters.
        self.path_to_tempfiles = path_to_tempfiles
        self.reset_data_files = reset_data_files
        self.remove_summary = remove_summary
        self.verbose = verbose

        # Future FastText model.
        self.model_object = None

    def __log_info(self, message):
        if self.verbose is not None:
            if self.verbose > 0:
                print(message)

    def __create_summary(self, x_data, y_data=None, path_to_tempfiles: str = None, reset_data_files: bool = None):
        """
        Create temp text file from x&y data for training fastText model.

        :param x_data: text examples (list of strings).
        :param y_data: class labels (list of strings).
        :param path_to_tempfiles: folder for temp files (overwrite init value).
        :param reset_data_files: replace old temp files with same name during fitting (overwrite init value).
        """
        reset_data_files = reset_data_files or self.reset_data_files
        folder_path = path_to_tempfiles or self.path_to_tempfiles
        path_to_summary = join(folder_path, "fastText_summary.txt")

        if os.path.isfile(path_to_summary) and not reset_data_files:
            pass
        else:
            os.makedirs(os.path.abspath(folder_path), exist_ok=True)

            # If unsupervised:
            if y_data is None:
                with open(path_to_summary, 'wt', encoding='utf-8-sig') as f:
                    f.writelines([s + '\n' for s in x_data])
            # If supervised:
            else:
                if len(x_data) == len(y_data):
                    self.__log_info('Creating new summary file (train+test)')
                    text = []
                    for el in range(len(x_data)):
                        text.extend([self.label + str(y_data[el]) + " " + x_data[el]])

                    with open(path_to_summary, 'wt', encoding='utf-8-sig') as f:
                        f.writelines([s + '\n' for s in text])
                    self.__log_info('Summary file successfully created.')
                else:
                    raise Exception("Data and target are of different lengths.")
        return path_to_summary

    def __set_fastext_common_default_values(self):
        self.model = "skipgram"
        self.dim = 100
        self.ws = 5
        self.epoch = 5
        self.minCount = 1
        self.minCountLabel = 0
        self.neg = 5
        self.wordNgrams = 1
        self.bucket = 2000000
        self.thread = 12
        self.lrUpdateRate = 100
        self.t = 0.0001
        self.label = "__label__"
        self.verbose = 2
        self.pretrainedVectors = ""

    def __set_fastext_supervised_default_values(self):
        self.lr = 0.1
        self.minn = 0
        self.maxn = 0
        self.loss = "softmax"

    def __set_fastext_unsupervised_default_values(self):
        self.lr = 0.05
        self.minn = 3
        self.maxn = 6
        self.loss = "ns"

    def __set_user_values(self):
        for k, v in self.kwargs.items():
            if v is not None:
                setattr(self, k, v)

    def fit(self, x_data, y_data=None, path_to_tempfiles: str = None, reset_data_files: bool = None):
        """
        Training fastText model.

        :param x_data: text examples (list of strings).
        :param y_data: class labels (list of strings).
        :param path_to_tempfiles: folder for temp files (overwrite init value).
        :param reset_data_files: replace old temp files with same name during fitting (overwrite init value).
        """
        self.__log_info('Number of sentences: {0}'.format(len(x_data)))
        self.__set_fastext_common_default_values()
        path_to_summary = self.__create_summary(x_data, y_data, path_to_tempfiles, reset_data_files)

        if y_data is None:
            self.__log_info('Train unsupervised model.')
            self.__set_fastext_unsupervised_default_values()
            self.__set_user_values()

            self.model_object = FastText.train_unsupervised(
                input=path_to_summary,
                model=self.model,
                lr=self.lr, dim=self.dim, ws=self.ws, epoch=self.epoch, minCount=self.minCount,
                minCountLabel=self.minCountLabel, minn=self.minn, maxn=self.maxn, neg=self.neg,
                wordNgrams=self.wordNgrams, loss=self.loss, bucket=self.bucket, thread=self.thread,
                lrUpdateRate=self.lrUpdateRate, t=self.t, label=self.label, verbose=self.verbose,
                pretrainedVectors=self.pretrainedVectors)
        else:
            self.__log_info('Train supervised model.')
            self.__set_fastext_supervised_default_values()
            self.__set_user_values()

            self.model_object = FastText.train_supervised(
                input=path_to_summary,
                lr=self.lr, dim=self.dim, ws=self.ws, epoch=self.epoch, minCount=self.minCount,
                minCountLabel=self.minCountLabel, minn=self.minn, maxn=self.maxn, neg=self.neg,
                wordNgrams=self.wordNgrams, loss=self.loss, bucket=self.bucket, thread=self.thread,
                lrUpdateRate=self.lrUpdateRate, t=self.t, label=self.label, verbose=self.verbose,
                pretrainedVectors=self.pretrainedVectors)

        self.__log_info("Model trained with user parameters:\n{0}.".format(
            {k: v for (k, v) in self.kwargs.items() if v is not None}))
        if self.remove_summary:
            os.remove(path_to_summary)
        return self

    def save(self, path):
        self.model_object.save_model(path)

    def load(self, path):
        self.model_object = fasttext.load_model(path)

    def predict(self, x_data, k_nearest: int = 1):
        """
        Predict values for x_data input values using trained model.

        :param x_data: text examples (list of strings).
        :param k_nearest: how many closest classes to show.
            if k_nearest == 1, returns list of int(labels),
            else returns list with labels and probabilities.
        """
        if self.model_object is None:
            raise Exception("Model is not trained.")

        predicted = [self.model_object.predict(m, k_nearest) for m in x_data]
        if k_nearest != 1:
            prob = [[el.replace(self.label, '') for el in tup[0]] for tup in predicted]
            result = [[prob[el], predicted[el][1]] for el in range(0, len(x_data))]
        else:
            result = [el[0][0].replace(self.label, '') for el in predicted]

        return result

    def score(self, x_data, y_data):
        """
        Accuracy score of model.
        """
        score = accuracy_score(self.predict(x_data), y_data)
        return score

    def transform(self, x_data):
        if self.model_object is None:
            raise Exception("Model is not trained.")

        if isinstance(x_data, list):
            x_vector = np.array([self.model_object.get_sentence_vector(x_str) for x_str in x_data])
        elif isinstance(x_data, str):
            x_vector = np.array(self.model_object.get_sentence_vector(x_data))
        else:
            raise ValueError('Input should be string or list of string.')
        return x_vector

    def vectorize_string(self, string: str):
        if self.model_object is None:
            raise Exception("Model is not trained.")

        if " " in string:
            return self.model_object.get_sentence_vector(string)
        else:
            return self.model_object.get_word_vector(string)


#############################################################################################################
if __name__ == "__main__":
    labelLiteral = '__label__'

    # some test data:
    example_x_data = [
        "aaaaa",
        "bbbbbb",
        "cccccc",
        "dddddd",
        "eeeeee",
        "aa aa aa",
        "bb bb bb",
        "cc cc cc",
        "dd d dd d",
        "ee ee eeee e"]
    example_y_data = ["0", "0", "0", "0", "0", "1", "1", "1", "1", "1"]

    # Use fastText

    ft = FastTextClassifier(path_to_tempfiles="fasttext_data", reset_data_files=True,
                            lr=0.8, dim=7, ws=5, epoch=25, minCount=1, minCountLabel=0,
                            minn=3, maxn=6, neg=5, wordNgrams=2, loss="softmax", bucket=1999999, thread=12,
                            lrUpdateRate=100, t=1e-4, label=labelLiteral, verbose=2, pretrainedVectors="")

    ft.fit(example_x_data, example_y_data)
    res = ft.score(example_x_data[0:2], example_y_data[0:2])
    print(res)

    # Using KFold method
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True)
    splitNumber = 0
    res = []
    for train_index, test_index in kf.split(example_x_data):
        splitNumber += 1
        print("split number:", splitNumber)
        ft.fit([example_x_data[el] for el in train_index], [example_y_data[el] for el in train_index])
        res.extend([ft.score([example_x_data[el] for el in test_index], [example_y_data[el] for el in test_index])])
        # res = estimate(y_test, predicted, label2name, print_incorect=False, print_table=True, X=X_test)
        # mat = pretty_confusion_matrix(y_test, predicted, label2name, showLabelText=True)
    print("Mean of {0} runs: {1:6.3f}".format(kf.get_n_splits(), np.mean(res)))

    # Using Stratified KFold
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5)
    splitNumber = 0
    res = []
    for train_index, test_index in skf.split(example_x_data, example_y_data):
        splitNumber += 1
        print("split number:", splitNumber)
        ft.fit([example_x_data[el] for el in train_index], [example_y_data[el] for el in train_index])
        res.extend([ft.score([example_x_data[el] for el in test_index], [example_y_data[el] for el in test_index])])
    print("Mean of {0} runs: {1:6.3f}".format(kf.get_n_splits(), np.mean(res)))

    # Using gridSearch
    # Recreate fasttext fot using gridSearch
    ft = FastTextClassifier(path_to_tempfiles="fasttext_data_grid_search", reset_data_files=True,
                            lr=0.8, dim=7, ws=5, epoch=25, minCount=1, minCountLabel=0,
                            minn=3, maxn=6, neg=5, wordNgrams=2, loss="softmax", bucket=3300000, thread=12,
                            lrUpdateRate=100, t=1e-4, label=labelLiteral, verbose=2, pretrainedVectors="")

    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV

    kf = KFold(n_splits=3, shuffle=True)
    parameters = {'lr': [0.6, 0.8], 'dim': [5, 7]}
    clf = GridSearchCV(ft, parameters, cv=kf.split(example_x_data))
    clf.fit(example_x_data, example_y_data)

    # Printing results:
    for el in range(len(clf.cv_results_['mean_test_score'])):
        print("Mean: {0:6.3f}, std: {1:6.3f}, fit time: {2:6.3f}, params: {3}".format(
            clf.cv_results_['mean_test_score'][el],
            clf.cv_results_['std_test_score'][el],
            clf.cv_results_['mean_fit_time'][el],
            clf.cv_results_['params'][el]))

    # fastText supervised example:
    # ft.fit(X_data, y_data)
    # ft.score(X_data, y_data)
    # predicted = ft.predict(['что-то здесь не работает'])
    # print(predicted)

    # %%
    # fastText unsupervised example:
    # ft.fit(X_data[0:500])
    # predicted = ft.model.get_word_vector('работать')
    # print(predicted)
