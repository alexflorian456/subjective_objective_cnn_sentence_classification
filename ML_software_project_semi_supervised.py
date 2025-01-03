import data_helpers
import numpy as np
import math
import tensorflow as tf
from matplotlib import pyplot as plt

tf.flags.DEFINE_string("positive_data_file", "./data/subj-obj/all_obj.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/subj-obj/all_subj.txt", "Data source for the negative data.")

vocab_file = "./corola.100.50.vec"

FLAGS = tf.flags.FLAGS

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def create_vocabulary_dicts(vocab_file):
    vocab_embed_dict = {}
    with open(vocab_file, 'r') as f:
        f.readline() # skip header
        word_id = 0
        while True:
            line = f.readline()
            if not line:
                break
            word = line.strip().split()[0]
            embed = line.strip().split()[1:]
            vocab_embed_dict[word] = [float(element) for element in embed]
            word_id += 1
    return vocab_embed_dict

def sentences_to_indices_matrix_and_embed_tensor(sentences, vocab_embed_dict, max_document_length, generate_embed_tensor=True):
    sentence_array = np.zeros((len(sentences), max_document_length), dtype=np.int32)

    word_id_dict = {}
    word_id = 1

    for sentence_idx, sentence in enumerate(sentences):
        for word_idx, word in enumerate(sentence.split(' ')):
            if word[len(word)-1] == '.':
                word = word.split('.')[0]
            
            word_array_value = 0
            if word in vocab_embed_dict.keys():
                if word in word_id_dict.keys():
                    word_array_value = word_id_dict[word]
                else:
                    word_id_dict[word] = word_id
                    word_array_value = word_id
                    word_id += 1
            
            sentence_array[sentence_idx, word_idx] = word_array_value

    if generate_embed_tensor:
        embed_tensor = np.zeros((word_id, max([len(embed) for embed in vocab_embed_dict.values()])), dtype=np.float32)
        print(embed_tensor.shape)

        iter = 1
        for word, word_id in word_id_dict.items():
            embed_tensor[word_id, :] = vocab_embed_dict[word]
            if iter%1000 == 0:
                print(f"Generating embed tensor: {int(iter/len(word_id_dict.keys())*100)}% Done")
            iter +=1
    else:
        embed_tensor = None
    
    return sentence_array, embed_tensor

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within class scatter matrix:
        # SW = sum((X_c - mean_X_c)^2 )

        # Between class scatter:
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            # (4, 1) * (1, 4) = (4,4) -> reshape
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for iter in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot(iter)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot(iter)

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [
            euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)
        ]
        return sum(distances) == 0

    def plot(self, step):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T[:2, :]
            ax.scatter(*point)

        for point in self.centroids:
            point = point[:2]
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.savefig(f"kmeans_steps/{step}.png")

print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
print("Max doc length:", max_document_length)
vocab_embed_dict = create_vocabulary_dicts(vocab_file)
x, embed_tensor = sentences_to_indices_matrix_and_embed_tensor(x_text, vocab_embed_dict, max_document_length)

sentence_embeddings_tensor = np.zeros((x.shape[0], embed_tensor.shape[1]), dtype=np.float32)

for sentence_index, sentence in enumerate(x):
    embedding_tensor_sum = np.zeros(embed_tensor.shape[1], dtype=np.float32)
    non_null_words = 0
    for word_index in sentence:
        if word_index == 0:
            continue
        non_null_words += 1
        embedding_tensor_sum += embed_tensor[word_index]
    
    sentence_embeddings_tensor[sentence_index] = embedding_tensor_sum / non_null_words
    if math.isnan(np.linalg.norm(sentence_embeddings_tensor[sentence_index])):
        print("Nan embedding: ", sentence, x_text[sentence_index])

lda_n_components = 10
lda = LDA(lda_n_components)
lda.fit(sentence_embeddings_tensor, y[:, 1])
sentences_transformed = lda.transform(sentence_embeddings_tensor)
print(sentences_transformed.shape)
plt.scatter(sentences_transformed[y[:, 1] == 1.0, 0], sentences_transformed[y[:, 1] == 1.0, 1], alpha=0.5, color='blue')
plt.scatter(sentences_transformed[y[:, 1] == 0.0, 0], sentences_transformed[y[:, 1] == 0.0, 1], alpha=0.5, color='red')
plt.savefig("lda_test.png")
plt.close()

k = KMeans(K=2, max_iters=150, plot_steps=True)
y_pred = k.predict(sentences_transformed)
