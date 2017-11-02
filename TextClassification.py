from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
import nltk
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

model_classifier_name = 'classifier.pkl'
model_vectorizer_name = 'vectorizer.pkl'
model_tfidf_transformer_name = 'tfidf_transformer.pkl'
model_label_names_name = 'label_names.pkl'

class TextCls():

    def __init__(self, label_names = None, X_train= None, y_train= None):
        if label_names is not None:
            self.k = len(label_names)
        self.label_names = label_names
        self.X_train, self.y_train = X_train, y_train

    def fit(self):
        stopwords = set([word.upper() for word in nltk.corpus.stopwords.words('portuguese')])
        stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '-'])
        self.vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords)

        X_train_vect = self.vectorizer.fit_transform(self.X_train)
        self.tfidf_transformer = TfidfTransformer()
        X_train_trans = self.tfidf_transformer.fit_transform(X_train_vect)

        #self.classifier = KNeighborsClassifier(n_neighbors=self.k)
        #self.classifier = MultinomialNB()
        linearSVC = LinearSVC()
        self.classifier = CalibratedClassifierCV(linearSVC)

        self.classifier.fit(X_train_trans, self.y_train)

    def predict(self, X_test):
        X_test_vect = self.vectorizer.transform(X_test)
        X_test_trans = self.tfidf_transformer.transform(X_test_vect)
        y_pred = self.classifier.predict(X_test_trans)
        return y_pred

    def predict_single(self, doc):
        X_test_vect = self.vectorizer.transform([doc])
        X_test_trans = self.tfidf_transformer.transform(X_test_vect)
        y_pred = zip(self.classifier.classes_, self.classifier.predict_proba(X_test_trans)[0])
        y_pred = sorted([(self.label_names[ind], score) for ind, score in y_pred], key=lambda x: -x[1])
        return y_pred

    def report(self, x_test, y_test, y_pred):
        print(classification_report(y_test, y_pred, target_names=self.label_names, digits=4))

        cnf_matrix = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cnf_matrix)
        total = 0
        same = 0
        y_test_array = y_test.tolist()
        for i in range(len(y_test_array)):
            if y_test_array[i] == y_pred[i]:
                same += 1
            total += 1
        print(total, same)

        print(precision_recall_fscore_support(y_test, y_pred, average='micro'))

    def predict_proba(self, X):
        pass

    def save(self):
        joblib.dump(self.classifier, model_classifier_name)
        joblib.dump(self.vectorizer, model_vectorizer_name)
        joblib.dump(self.tfidf_transformer, model_tfidf_transformer_name)
        joblib.dump(self.label_names, model_label_names_name)

    def load(self):
        self.classifier = joblib.load(model_classifier_name)
        self.vectorizer = joblib.load(model_vectorizer_name)
        self.tfidf_transformer = joblib.load(model_tfidf_transformer_name)
        self.label_names = joblib.load(model_label_names_name)

    @staticmethod
    def exists_saved_model():
        file_exists = os.path.exists(model_classifier_name) and os.path.exists(model_vectorizer_name) and os.path.exists(model_tfidf_transformer_name) and os.path.exists(model_label_names_name)
        return file_exists

    def plot_confusion_matrix(self, cm,
                              normalize=False,
                              title='Confusion matrix',

                              cmap=plt.cm.Blues):
        classes = self.label_names
        np.set_printoptions(precision=2)
        plt.figure()

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

def main():

    train_samples = {
        'article': [
            'this is an amazing article',
            'this article is great read'
        ],
        'artwork': [
            'this is a great painting art work',
            'this is a random oil painting'
        ]
    }

    label_names = sorted(train_samples.keys())

    X_train, y_train = [], []
    for label_name, docs in train_samples.items():
        label_index = label_names.index(label_name)
        for doc in docs:
            X_train.append(doc)
            y_train.append(label_index)

    model = TextCls(label_names, X_train, y_train)

    model.fit()

    X_test = [
        'this article is great',
        'cool painting'
    ]
    y_test = [
        label_names.index('article'),
        label_names.index('artwork')
    ]

    for doc in X_test:
        top_preds = model.predict_single(doc)[:2]
        print(doc)
        for label, score in top_preds:
            print('\t{}\t{}'.format(label, score))

    y_pred = model.predict(X_test)
    model.report(X_test, y_test, y_pred)


if __name__ == '__main__':
    main()