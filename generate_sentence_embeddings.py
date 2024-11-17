import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import data_helpers
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

tf.flags.DEFINE_string("positive_data_file", "./data/subj-obj/all_obj.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/subj-obj/all_subj.txt", "Data source for the negative data.")

vocab_file = "./corola.100.50.vec"

FLAGS = tf.flags.FLAGS

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

def sentences_to_indices_matrix_and_embed_tensor(sentences, vocab_embed_dict, max_document_length, generate_embed_tensor):
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

def sentence_embeddings(generate_embed_tensor=True):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    print("Max doc length:", max_document_length)
    vocab_embed_dict = create_vocabulary_dicts(vocab_file)
    x, embed_tensor = sentences_to_indices_matrix_and_embed_tensor(x_text, vocab_embed_dict, max_document_length, generate_embed_tensor)

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

    print(sentence_embeddings_tensor.shape[1])

    # Option 1: Without normalization
    pca_unnormalized = PCA(n_components=2)
    reduced_unnormalized = pca_unnormalized.fit_transform(sentence_embeddings_tensor)

    # Option 2: With normalization
    normalized_embeddings = sentence_embeddings_tensor / np.linalg.norm(sentence_embeddings_tensor, axis=1, keepdims=True)
    pca_normalized = PCA(n_components=2)
    reduced_normalized = pca_normalized.fit_transform(normalized_embeddings)


    # Plot the sentences as points
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_unnormalized[y[:, 1] == 1.0, 0], reduced_unnormalized[y[:, 1] == 1.0, 1], alpha=0.5, color='blue')
    plt.scatter(reduced_unnormalized[y[:, 1] == 0.0, 0], reduced_unnormalized[y[:, 1] == 0.0, 1], alpha=0.5, color='red')
    eigenvectors=pca_unnormalized.components_
    for i in range(eigenvectors.shape[0]):
        plt.quiver(0, 0, eigenvectors[i, 0], eigenvectors[i, 1], angles='xy', scale_units='xy', scale=0.01, color='yellow', label=f"PC{i+1}" if i == 0 else "")
    
    plt.title(f"Sentence Embeddings in PCA Space.")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig("pca_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(pca_unnormalized.components_, cmap='coolwarm')
    plt.title('PCA Loading Matrix')
    plt.savefig("loading_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Assuming `labels` is the target and `embeddings` is the feature matrix
    X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings_tensor, y[:, 1], test_size=0.3, random_state=42)

    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    # Get feature importance
    feature_importances = rf.feature_importances_

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.title("Feature Importance from Random Forest")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    # plt.show()
    plt.savefig("importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Reduce dimensionality using t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(sentence_embeddings_tensor)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[y[:, 1] == 1, 0], tsne_result[y[:, 1] == 1, 1], color='blue', label='objective', alpha=0.5)
    plt.scatter(tsne_result[y[:, 1] == 0, 0], tsne_result[y[:, 1] == 0, 1], color='red', label='subjective', alpha=0.5)
    plt.title("t-SNE of Sentence Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.savefig("tsne.png", dpi=300, bbox_inches="tight")
    plt.close()

    correlation_matrix = np.corrcoef(sentence_embeddings_tensor, rowvar=False)
    df_corr = pd.DataFrame(correlation_matrix)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_corr, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.title("Correlation Matrix of Sentence Embeddings")
    plt.savefig("corr_plot.png", dpi=300, bbox_inches="tight")

    # Descriptive statistics for sentence embeddings
    embedding_stats = pd.DataFrame(reduced_unnormalized).describe()

    # Descriptive statistics for labels
    label_stats = pd.Series(y[:, 1]).describe()

    print("Embedding Statistics:\n", embedding_stats)
    print("\nLabel Statistics:\n", label_stats)

def main(argv=None):
    sentence_embeddings()

if __name__ == '__main__':
    tf.app.run()