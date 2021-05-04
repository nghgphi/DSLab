from numpy.lib.function_base import _calculate_shapes
from scipy.sparse import data
from scipy.sparse.construct import random
from utils import *
import numpy as np
def clustering_with_KMeans():
    data, labels = load_data(data_path= r'data\tf_idf.txt')

    from sklearn.cluster import KMeans
    from scipy.sparse import csc_matrix

    X = csc_matrix(data)

    kmeans = KMeans(
        n_clusters= 20,
        init= 'random',
        n_init= 5,
        tol= 1e-3,
        random_state= 2021
    ).fit(X)
    labels = kmeans.labels_
    from sklearn.metrics.cluster import completeness_score
    print("Completeness score: {}".format(completeness_score(labels, labels)))
clustering_with_KMeans()
# def compute_accuracy(predicted_y, expected_y):
#     matches = np.equal(predicted_y, expected_y)
#     accuracy = np.sum(matches.astype(float)) / expected_y.size
#     return accuracy
# # def classifying_with_linear_SVMs():
# #     train_x, train_y = load_data(data_path= r'data\train_tf_idf.txt')
# #     from sklearn.svm import LinearSVC
# #     classifier = LinearSVC(
# #         C= 10,
# #         tol= 0.001,
# #         verbose= True
# #     )
# #     classifier.fit(train_x, train_y)
# #     test_x, test_y = load_data(data_path= r'data\test_tf_idf.txt')
# #     predicted_y = classifier.predict(test_x)
# #     accuracy = compute_accuracy(predicted_y, test_y)
# #     print('Accuracy: ', accuracy)

# # classifying_with_linear_SVMs()

# def classifying_with_kernel_SVMs():
#     train_x, train_y = load_data(data_path= r'data\train_tf_idf.txt')
#     train_x = np.array(train_x)
#     train_y = np.array(train_y)
#     from sklearn.svm import LinearSVC, SVC
#     classifier = SVC(
#         C= 10,
#         tol= 0.001,
#         kernel= 'rbf',
#         gamma= 0.1,
#         verbose= True
#     )
#     classifier.fit(train_x, train_y)
#     test_x, test_y = load_data(data_path= r'data\test_tf_idf.txt')
#     predicted_y = classifier.predict(test_x)
#     accuracy = compute_accuracy(predicted_y, test_y)
#     print('Accuracy: ', accuracy)

# classifying_with_kernel_SVMs()