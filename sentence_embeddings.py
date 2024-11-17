import tensorflow as tf
import data_helpers
import numpy as np

tf.flags.DEFINE_string("positive_data_file", "./data/subj-obj/all_obj.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/subj-obj/all_subj.txt", "Data source for the negative data.")

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
        embed_tensor = tf.Variable(tf.zeros([word_id, max([len(embed) for embed in vocab_embed_dict.values()])]))
        print(embed_tensor.shape)

        iter = 1
        for word, word_id in word_id_dict.items():
            embed_tensor[word_id].assign(tf.constant(vocab_embed_dict[word]))
            if iter%1000 == 0:
                print(f"Generating embed tensor: {int(iter/len(word_id_dict.keys())*100)}% Done")
            iter +=1
    else:
        embed_tensor = None
    
    print("array:", sentence_array)
    return sentence_array, embed_tensor

def preprocess(generate_embed_tensor=True):
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
    print(x)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    print(y_train)
    x_text = [x_text[i] for i in shuffle_indices]
    x_text = x_text[dev_sample_index:]
    return x_train, y_train, x_dev, y_dev, embed_tensor, x_text
