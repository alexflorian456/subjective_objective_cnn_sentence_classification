import data_helpers
import numpy as np
import math
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import t

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

        # Centroids and clusters
        self.centroids = None
        self.clusters = None

    def fit(self, X):
        """
        Fits the KMeans model to the data X.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize centroids randomly
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

            # Check if clusters have converged
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot(iter)

    def predict(self, X):
        """
        Predicts the closest cluster for each sample in X.
        """
        predictions = []
        for sample in X:
            centroid_idx = self._closest_centroid(sample, self.centroids)
            predictions.append(centroid_idx)
        return np.array(predictions)

    def _create_clusters(self, centroids):
        # Assign samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # Distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # Assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # Distances between each old and new centroid
        distances = [
            euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)
        ]
        return sum(distances) == 0

    def plot(self, step):
        """
        Saves a plot of the current clusters and centroids.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            points = self.X[index].T[:2]
            ax.scatter(*points)

        for point in self.centroids:
            ax.scatter(*point[:2], marker="x", color="black", linewidth=2)

        plt.savefig(f"kmeans_steps/{step}.png")
        plt.close()


def binary_classification_metrics(predicted_labels, ground_truth_labels):
    """
    Compute Accuracy, Precision, Recall, and F1-Score for both classes
    and report their macro-average.

    Parameters:
    - predicted_labels: Array-like, predicted labels (0 or 1).
    - ground_truth_labels: Array-like, ground truth labels (0 or 1).

    Returns:
    - metrics: A dictionary containing accuracy, macro precision, macro recall, and macro F1-score.
    """
    # Convert inputs to numpy arrays
    predicted_labels = np.array(predicted_labels)
    ground_truth_labels = np.array(ground_truth_labels)

    # Metrics for class 0
    TP_0 = np.sum((predicted_labels == 0) & (ground_truth_labels == 0))
    FP_0 = np.sum((predicted_labels == 0) & (ground_truth_labels == 1))
    FN_0 = np.sum((predicted_labels == 1) & (ground_truth_labels == 0))

    precision_0 = TP_0 / (TP_0 + FP_0) if (TP_0 + FP_0) > 0 else 0.0
    recall_0 = TP_0 / (TP_0 + FN_0) if (TP_0 + FN_0) > 0 else 0.0
    f1_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0

    # Metrics for class 1
    TP_1 = np.sum((predicted_labels == 1) & (ground_truth_labels == 1))
    FP_1 = np.sum((predicted_labels == 1) & (ground_truth_labels == 0))
    FN_1 = np.sum((predicted_labels == 0) & (ground_truth_labels == 1))

    precision_1 = TP_1 / (TP_1 + FP_1) if (TP_1 + FP_1) > 0 else 0.0
    recall_1 = TP_1 / (TP_1 + FN_1) if (TP_1 + FN_1) > 0 else 0.0
    f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0

    # Macro-average metrics
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    # Accuracy
    accuracy = np.sum(predicted_labels == ground_truth_labels) / len(ground_truth_labels)

    # Return results as a dictionary
    return {
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1-Score": macro_f1
    }

def clustering_metrics(X, labels, true_labels=None):
    """
    Computes internal and external clustering metrics.

    Parameters:
    - X: np.ndarray, the data points (n_samples, n_features).
    - labels: np.ndarray, the predicted cluster labels (n_samples,).
    - true_labels: np.ndarray or None, the ground truth labels (optional).

    Returns:
    - metrics: dict, dictionary with the computed metrics.
    """

    # Internal metrics
    def silhouette_score(X, labels):
        """Computes the silhouette score for internal evaluation."""
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            return 0  # Silhouette score is undefined for one cluster

        distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        silhouette_values = []

        for i in range(len(X)):
            cluster = labels[i]
            own_cluster = distances[i][labels == cluster]
            other_clusters = [
                distances[i][labels == other].mean()
                for other in unique_labels if other != cluster
            ]
            a = own_cluster.mean() if len(own_cluster) > 1 else 0
            b = min(other_clusters) if other_clusters else 0
            silhouette_values.append((b - a) / max(a, b))
        
        return np.mean(silhouette_values)

    def compute_davies_bouldin(X, labels):
        """Computes Davies-Bouldin Index."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        if n_clusters <= 1:
            return float('inf')  # Undefined for 1 or no clusters

        centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
        cluster_variances = [
            np.mean(np.linalg.norm(X[labels == k] - centroid, axis=1))
            for k, centroid in enumerate(centroids)
        ]
        db_values = []
        for i in range(n_clusters):
            max_ratio = 0
            for j in range(n_clusters):
                if i != j:
                    ratio = (cluster_variances[i] + cluster_variances[j]) / np.linalg.norm(centroids[i] - centroids[j])
                    max_ratio = max(max_ratio, ratio)
            db_values.append(max_ratio)
        return np.mean(db_values)

    # External metrics (if ground truth labels are provided)
    if true_labels is not None:
        def adjusted_rand_index(labels, true_labels):
            """Computes the Adjusted Rand Index manually."""
            n = len(labels)
            contingency = np.zeros((2, 2))
            for i in range(n):
                contingency[labels[i], true_labels[i]] += 1

            sum_rows = contingency.sum(axis=1)
            sum_cols = contingency.sum(axis=0)
            sum_comb = (contingency * (contingency - 1)).sum()

            expected_index = sum(sum_rows * (sum_rows - 1)) * sum(sum_cols * (sum_cols - 1)) / (n * (n - 1))
            max_index = (sum(sum_rows * (sum_rows - 1)) + sum(sum_cols * (sum_cols - 1))) / 2
            return (sum_comb - expected_index) / (max_index - expected_index)

        def normalized_mutual_info(labels, true_labels):
            """Computes the Normalized Mutual Information manually."""
            def entropy(labels):
                probs = np.bincount(labels) / len(labels)
                return -np.sum(probs * np.log2(probs + 1e-10))  # Avoid log(0)

            joint = np.zeros((2, 2))
            for i in range(len(labels)):
                joint[labels[i], true_labels[i]] += 1
            joint /= joint.sum()

            h_labels = entropy(labels)
            h_true = entropy(true_labels)
            h_joint = -np.sum(joint * np.log2(joint + 1e-10))  # Joint entropy

            mutual_info = h_labels + h_true - h_joint
            return mutual_info / max(h_labels, h_true)

        ari = adjusted_rand_index(labels, true_labels)
        nmi = normalized_mutual_info(labels, true_labels)
    else:
        ari = nmi = None

    # Collect metrics
    metrics = {
        "Silhouette Score": silhouette_score(X, labels),
        "Davies-Bouldin Index": compute_davies_bouldin(X, labels),
    }

    if true_labels is not None:
        metrics.update({
            "Adjusted Rand Index (ARI)": ari,
            "Normalized Mutual Information (NMI)": nmi,
        })

    return metrics

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

print(sentence_embeddings_tensor.shape)

shuffle_perm = np.random.permutation(y.shape[0])
X_shuffled = sentence_embeddings_tensor[shuffle_perm]
y_shuffled = y[shuffle_perm]

# No LDA
# Configuration
k = KMeans(K=2, max_iters=150, plot_steps=False)
fold_count = 5
fold_size = y.shape[0] // fold_count
confidence_level = 0.95

# Initialize accumulators for binary classification and clustering metrics
no_lda_accuracies = []
no_lda_precisions = []
no_lda_recalls = []
no_lda_f1_scores = []
silhouette_scores = []
davies_bouldin_indexes = []
adjusted_rand_indices = []
normalized_mutual_informations = []

for fold_start in range(0, fold_size * fold_count, fold_size):
    # Define the range for the current test fold
    fold_end = fold_start + fold_size
    
    # Split the data into training and test sets for this fold
    X_test = X_shuffled[fold_start:fold_end, :]
    y_test = y_shuffled[fold_start:fold_end]
    
    X_train = np.vstack((X_shuffled[:fold_start, :], X_shuffled[fold_end:, :]))
    y_train = np.vstack((y_shuffled[:fold_start, :], y_shuffled[fold_end:, :]))

    # Train KMeans
    k.fit(X_train)
    
    # Predict clusters
    y_pred_no_lda = k.predict(X_test)
    
    # Compute metrics (Binary classification metrics)
    metrics_no_lda = binary_classification_metrics(y_pred_no_lda, y_test[:, 0])
    metrics_no_lda_inv = binary_classification_metrics(1 - y_pred_no_lda, y_test[:, 0])
    metrics_no_lda = metrics_no_lda if metrics_no_lda['Accuracy'] > metrics_no_lda_inv['Accuracy'] else metrics_no_lda_inv
    
    # Append binary classification metrics for this fold
    no_lda_accuracies.append(metrics_no_lda['Accuracy'])
    no_lda_precisions.append(metrics_no_lda['Macro Precision'])
    no_lda_recalls.append(metrics_no_lda['Macro Recall'])
    no_lda_f1_scores.append(metrics_no_lda['Macro F1-Score'])
    
    # Compute clustering metrics (Internal and External)
    clustering_metrics_fold = clustering_metrics(X_test, y_pred_no_lda, true_labels=y_test[:, 0])  # Assuming y_test contains true labels
    
    # Unpack clustering metrics and store in separate accumulators
    silhouette_scores.append(clustering_metrics_fold["Silhouette Score"])
    davies_bouldin_indexes.append(clustering_metrics_fold["Davies-Bouldin Index"])
    adjusted_rand_indices.append(clustering_metrics_fold["Adjusted Rand Index (ARI)"])
    normalized_mutual_informations.append(clustering_metrics_fold["Normalized Mutual Information (NMI)"])

# Convert metrics to numpy arrays for statistical analysis
no_lda_accuracies = np.array(no_lda_accuracies)
no_lda_precisions = np.array(no_lda_precisions)
no_lda_recalls = np.array(no_lda_recalls)
no_lda_f1_scores = np.array(no_lda_f1_scores)
silhouette_scores = np.array(silhouette_scores)
davies_bouldin_indexes = np.array(davies_bouldin_indexes)
adjusted_rand_indices = np.array(adjusted_rand_indices)
normalized_mutual_informations = np.array(normalized_mutual_informations)

# Compute means for binary classification metrics
accuracy_mean = np.mean(no_lda_accuracies)
precision_mean = np.mean(no_lda_precisions)
recall_mean = np.mean(no_lda_recalls)
f1_mean = np.mean(no_lda_f1_scores)

# Compute standard deviations for binary classification metrics
accuracy_std = np.std(no_lda_accuracies, ddof=1)
precision_std = np.std(no_lda_precisions, ddof=1)
recall_std = np.std(no_lda_recalls, ddof=1)
f1_std = np.std(no_lda_f1_scores, ddof=1)

# Compute confidence intervals (95%) for binary classification metrics
t_value = t.ppf((1 + confidence_level) / 2, df=fold_count - 1)
accuracy_ci = (accuracy_mean - t_value * accuracy_std / np.sqrt(fold_count), 
               accuracy_mean + t_value * accuracy_std / np.sqrt(fold_count))
precision_ci = (precision_mean - t_value * precision_std / np.sqrt(fold_count), 
                precision_mean + t_value * precision_std / np.sqrt(fold_count))
recall_ci = (recall_mean - t_value * recall_std / np.sqrt(fold_count), 
             recall_mean + t_value * recall_std / np.sqrt(fold_count))
f1_ci = (f1_mean - t_value * f1_std / np.sqrt(fold_count), 
         f1_mean + t_value * f1_std / np.sqrt(fold_count))

# Print binary classification results
print("Cross val No-LDA metrics:")
print(f"Accuracy: {accuracy_mean:.6f} ± {accuracy_std:.6f}, CI: {accuracy_ci}")
print(f"Macro Precision: {precision_mean:.6f} ± {precision_std:.6f}, CI: {precision_ci}")
print(f"Macro Recall: {recall_mean:.6f} ± {recall_std:.6f}, CI: {recall_ci}")
print(f"Macro F1: {f1_mean:.6f} ± {f1_std:.6f}, CI: {f1_ci}")

# Now compute statistics for clustering metrics
# Silhouette Score
silhouette_mean = np.mean(silhouette_scores)
silhouette_std = np.std(silhouette_scores, ddof=1)
silhouette_ci = (silhouette_mean - t_value * silhouette_std / np.sqrt(fold_count), 
                 silhouette_mean + t_value * silhouette_std / np.sqrt(fold_count))

# Davies-Bouldin Index
davies_bouldin_mean = np.mean(davies_bouldin_indexes)
davies_bouldin_std = np.std(davies_bouldin_indexes, ddof=1)
davies_bouldin_ci = (davies_bouldin_mean - t_value * davies_bouldin_std / np.sqrt(fold_count), 
                     davies_bouldin_mean + t_value * davies_bouldin_std / np.sqrt(fold_count))

# Adjusted Rand Index (ARI)
ari_mean = np.mean(adjusted_rand_indices)
ari_std = np.std(adjusted_rand_indices, ddof=1)
ari_ci = (ari_mean - t_value * ari_std / np.sqrt(fold_count), 
          ari_mean + t_value * ari_std / np.sqrt(fold_count))

# Normalized Mutual Information (NMI)
nmi_mean = np.mean(normalized_mutual_informations)
nmi_std = np.std(normalized_mutual_informations, ddof=1)
nmi_ci = (nmi_mean - t_value * nmi_std / np.sqrt(fold_count), 
          nmi_mean + t_value * nmi_std / np.sqrt(fold_count))

# Print clustering metrics results
print("\nClustering metrics:")
print(f"Silhouette Score: {silhouette_mean:.6f} ± {silhouette_std:.6f}, CI: {silhouette_ci}")
print(f"Davies-Bouldin Index: {davies_bouldin_mean:.6f} ± {davies_bouldin_std:.6f}, CI: {davies_bouldin_ci}")
print(f"Adjusted Rand Index (ARI): {ari_mean:.6f} ± {ari_std:.6f}, CI: {ari_ci}")
print(f"Normalized Mutual Information (NMI): {nmi_mean:.6f} ± {nmi_std:.6f}, CI: {nmi_ci}")


# LDA with different numbers of components
for lda_n_components in range(1, 101):
    print(f"{lda_n_components} LDA components:")
    # lda = LDA(lda_n_components)
    # lda.fit(sentence_embeddings_tensor, y[:, 1])
    # sentences_transformed = lda.transform(sentence_embeddings_tensor)
    # print(sentences_transformed.shape)

    # y_pred = k.predict(sentences_transformed)

    # metrics = binary_classification_metrics(y_pred, y[:, 1])
    # metrics_inv = binary_classification_metrics(1 - y_pred, y[:, 1])

    # print(metrics if metrics['Accuracy'] > metrics_inv['Accuracy'] else metrics_inv)
    # Initialize accumulators for binary classification and clustering metrics
    lda_accuracies = []
    lda_precisions = []
    lda_recalls = []
    lda_f1_scores = []
    silhouette_scores = []
    davies_bouldin_indexes = []
    adjusted_rand_indices = []
    normalized_mutual_informations = []

    for fold_start in range(0, fold_size * fold_count, fold_size):
        # Define the range for the current test fold
        fold_end = fold_start + fold_size
        
        # Split the data into training and test sets for this fold
        X_test = X_shuffled[fold_start:fold_end, :]
        y_test = y_shuffled[fold_start:fold_end]
        
        X_train = np.vstack((X_shuffled[:fold_start, :], X_shuffled[fold_end:, :]))
        y_train = np.vstack((y_shuffled[:fold_start, :], y_shuffled[fold_end:, :]))

        # Train KMeans
        lda = LDA(lda_n_components)
        lda.fit(X_train, y_train[:, 0])
        X_train_lda = lda.transform(X_train)
        X_test_lda = lda.transform(X_test)
        k.fit(X_train_lda)
        
        # Predict clusters
        y_pred_lda = k.predict(X_test_lda)
        
        # Compute metrics (Binary classification metrics)
        metrics_lda = binary_classification_metrics(y_pred_lda, y_test[:, 0])
        metrics_lda_inv = binary_classification_metrics(1 - y_pred_lda, y_test[:, 0])
        metrics_lda = metrics_lda if metrics_lda['Accuracy'] > metrics_lda_inv['Accuracy'] else metrics_lda_inv
        
        # Append binary classification metrics for this fold
        lda_accuracies.append(metrics_lda['Accuracy'])
        lda_precisions.append(metrics_lda['Macro Precision'])
        lda_recalls.append(metrics_lda['Macro Recall'])
        lda_f1_scores.append(metrics_lda['Macro F1-Score'])
        
        # Compute clustering metrics (Internal and External)
        clustering_metrics_fold = clustering_metrics(X_test_lda, y_pred_lda, true_labels=y_test[:, 0])  # Assuming y_test contains true labels
        
        # Unpack clustering metrics and store in separate accumulators
        silhouette_scores.append(clustering_metrics_fold["Silhouette Score"])
        davies_bouldin_indexes.append(clustering_metrics_fold["Davies-Bouldin Index"])
        adjusted_rand_indices.append(clustering_metrics_fold["Adjusted Rand Index (ARI)"])
        normalized_mutual_informations.append(clustering_metrics_fold["Normalized Mutual Information (NMI)"])

    # Convert metrics to numpy arrays for statistical analysis
    lda_accuracies = np.array(lda_accuracies)
    lda_precisions = np.array(lda_precisions)
    lda_recalls = np.array(lda_recalls)
    lda_f1_scores = np.array(lda_f1_scores)
    silhouette_scores = np.array(silhouette_scores)
    davies_bouldin_indexes = np.array(davies_bouldin_indexes)
    adjusted_rand_indices = np.array(adjusted_rand_indices)
    normalized_mutual_informations = np.array(normalized_mutual_informations)

    # Compute means for binary classification metrics
    accuracy_mean = np.mean(lda_accuracies)
    precision_mean = np.mean(lda_precisions)
    recall_mean = np.mean(lda_recalls)
    f1_mean = np.mean(lda_f1_scores)

    # Compute standard deviations for binary classification metrics
    accuracy_std = np.std(lda_accuracies, ddof=1)
    precision_std = np.std(lda_precisions, ddof=1)
    recall_std = np.std(lda_recalls, ddof=1)
    f1_std = np.std(lda_f1_scores, ddof=1)

    # Compute confidence intervals (95%) for binary classification metrics
    t_value = t.ppf((1 + confidence_level) / 2, df=fold_count - 1)
    accuracy_ci = (accuracy_mean - t_value * accuracy_std / np.sqrt(fold_count), 
                accuracy_mean + t_value * accuracy_std / np.sqrt(fold_count))
    precision_ci = (precision_mean - t_value * precision_std / np.sqrt(fold_count), 
                    precision_mean + t_value * precision_std / np.sqrt(fold_count))
    recall_ci = (recall_mean - t_value * recall_std / np.sqrt(fold_count), 
                recall_mean + t_value * recall_std / np.sqrt(fold_count))
    f1_ci = (f1_mean - t_value * f1_std / np.sqrt(fold_count), 
            f1_mean + t_value * f1_std / np.sqrt(fold_count))

    # Print binary classification results
    print("Cross val No-LDA metrics:")
    print(f"Accuracy: {accuracy_mean:.6f} ± {accuracy_std:.6f}, CI: {accuracy_ci}")
    print(f"Macro Precision: {precision_mean:.6f} ± {precision_std:.6f}, CI: {precision_ci}")
    print(f"Macro Recall: {recall_mean:.6f} ± {recall_std:.6f}, CI: {recall_ci}")
    print(f"Macro F1: {f1_mean:.6f} ± {f1_std:.6f}, CI: {f1_ci}")

    # Now compute statistics for clustering metrics
    # Silhouette Score
    silhouette_mean = np.mean(silhouette_scores)
    silhouette_std = np.std(silhouette_scores, ddof=1)
    silhouette_ci = (silhouette_mean - t_value * silhouette_std / np.sqrt(fold_count), 
                    silhouette_mean + t_value * silhouette_std / np.sqrt(fold_count))

    # Davies-Bouldin Index
    davies_bouldin_mean = np.mean(davies_bouldin_indexes)
    davies_bouldin_std = np.std(davies_bouldin_indexes, ddof=1)
    davies_bouldin_ci = (davies_bouldin_mean - t_value * davies_bouldin_std / np.sqrt(fold_count), 
                        davies_bouldin_mean + t_value * davies_bouldin_std / np.sqrt(fold_count))

    # Adjusted Rand Index (ARI)
    ari_mean = np.mean(adjusted_rand_indices)
    ari_std = np.std(adjusted_rand_indices, ddof=1)
    ari_ci = (ari_mean - t_value * ari_std / np.sqrt(fold_count), 
            ari_mean + t_value * ari_std / np.sqrt(fold_count))

    # Normalized Mutual Information (NMI)
    nmi_mean = np.mean(normalized_mutual_informations)
    nmi_std = np.std(normalized_mutual_informations, ddof=1)
    nmi_ci = (nmi_mean - t_value * nmi_std / np.sqrt(fold_count), 
            nmi_mean + t_value * nmi_std / np.sqrt(fold_count))

    # Print clustering metrics results
    print("\nClustering metrics:")
    print(f"Silhouette Score: {silhouette_mean:.6f} ± {silhouette_std:.6f}, CI: {silhouette_ci}")
    print(f"Davies-Bouldin Index: {davies_bouldin_mean:.6f} ± {davies_bouldin_std:.6f}, CI: {davies_bouldin_ci}")
    print(f"Adjusted Rand Index (ARI): {ari_mean:.6f} ± {ari_std:.6f}, CI: {ari_ci}")
    print(f"Normalized Mutual Information (NMI): {nmi_mean:.6f} ± {nmi_std:.6f}, CI: {nmi_ci}")
