import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
import data_helpers
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

fold_count = 5
fold_size = y.shape[0] // 5

no_lda_accuracy_sum = 0
no_lda_precision_sum = 0
no_lda_recall_sum = 0
no_lda_f1_sum = 0

for fold_start in range(0, fold_size * fold_count, fold_size):

    k = KMeans(n_clusters=2, random_state=42)
    # No LDA
    # Define the range for the current test fold
    fold_end = fold_start + fold_size
    
    # Split the data into training and test sets for this fold
    X_test = X_shuffled[fold_start:fold_end, :]
    y_test = y_shuffled[fold_start:fold_end]
    
    X_train = np.vstack((X_shuffled[:fold_start, :], X_shuffled[fold_end:, :]))
    y_train = np.vstack((y_shuffled[:fold_start, :], y_shuffled[fold_end:, :]))

    k.fit(X_train)
    y_pred_no_lda = k.predict(X_test)

    metrics_no_lda = binary_classification_metrics(y_pred_no_lda, y_test[:, 0])
    metrics_no_lda_inv = binary_classification_metrics(1- y_pred_no_lda, y_test[:, 0])
    metrics_no_lda = metrics_no_lda if metrics_no_lda['Accuracy'] > metrics_no_lda_inv['Accuracy'] else metrics_no_lda_inv

    no_lda_accuracy_sum += metrics_no_lda['Accuracy']
    no_lda_precision_sum += metrics_no_lda['Macro Precision']
    no_lda_recall_sum += metrics_no_lda['Macro Recall']
    no_lda_f1_sum += metrics_no_lda['Macro F1-Score']

# Compute averages across folds
no_lda_accuracy_avg = no_lda_accuracy_sum / fold_count
no_lda_precision_avg = no_lda_precision_sum / fold_count
no_lda_recall_avg = no_lda_recall_sum / fold_count
no_lda_f1_avg = no_lda_f1_sum / fold_count

# Print results
print("Cross val No-LDA metrics:")
print(f"Accuracy: {no_lda_accuracy_avg:.6f}")
print(f"Macro Precision: {no_lda_precision_avg:.6f}")
print(f"Macro Recall: {no_lda_recall_avg:.6f}")
print(f"Macro F1: {no_lda_f1_avg:.6f}")

lda_accuracy_sum = 0
lda_precision_sum = 0
lda_recall_sum = 0
lda_f1_sum = 0

for fold_start in range(0, fold_size * fold_count, fold_size):

    k = KMeans(n_clusters=2, random_state=42)
    # No LDA
    # Define the range for the current test fold
    fold_end = fold_start + fold_size
    
    # Split the data into training and test sets for this fold
    X_test = X_shuffled[fold_start:fold_end, :]
    y_test = y_shuffled[fold_start:fold_end]
    
    X_train = np.vstack((X_shuffled[:fold_start, :], X_shuffled[fold_end:, :]))
    y_train = np.vstack((y_shuffled[:fold_start, :], y_shuffled[fold_end:, :]))

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_train, y_train[:, 0])
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)

    k.fit(X_train_lda)
    y_pred_lda = k.predict(X_test_lda)

    metrics_lda = binary_classification_metrics(y_pred_lda, y_test[:, 0])
    metrics_lda_inv = binary_classification_metrics(1- y_pred_lda, y_test[:, 0])
    metrics_lda = metrics_lda if metrics_lda['Accuracy'] > metrics_lda_inv['Accuracy'] else metrics_lda_inv

    lda_accuracy_sum += metrics_lda['Accuracy']
    lda_precision_sum += metrics_lda['Macro Precision']
    lda_recall_sum += metrics_lda['Macro Recall']
    lda_f1_sum += metrics_lda['Macro F1-Score']

# Compute averages across folds
lda_accuracy_avg = lda_accuracy_sum / fold_count
lda_precision_avg = lda_precision_sum / fold_count
lda_recall_avg = lda_recall_sum / fold_count
lda_f1_avg = lda_f1_sum / fold_count

# Print results
print()
print("Cross val LDA metrics:")
print(f"Accuracy: {lda_accuracy_avg:.6f}")
print(f"Macro Precision: {lda_precision_avg:.6f}")
print(f"Macro Recall: {lda_recall_avg:.6f}")
print(f"Macro F1: {lda_f1_avg:.6f}")

# lda = LinearDiscriminantAnalysis(n_components=1)
# sentences_transformed = lda.fit_transform(sentence_embeddings_tensor, y[:, 1])
# print(sentences_transformed.shape)

# y_pred = k.fit_predict(sentences_transformed)

# metrics = binary_classification_metrics(y_pred, y[:, 1])
# metrics_inv = binary_classification_metrics(1 - y_pred, y[:, 1])

# print(metrics if metrics['Accuracy'] > metrics_inv['Accuracy'] else metrics_inv)

# # Step 1: Apply LDA (dimensionality reduction)
# lda = LinearDiscriminantAnalysis(n_components=min(len(set(y)) - 1, X.shape[1]))
# X_lda = lda.fit_transform(X, y)

# # Step 2: Apply KMeans clustering on LDA-transformed data
# kmeans = KMeans(n_clusters=2, random_state=42)
# kmeans_labels = kmeans.fit_predict(X_lda)


