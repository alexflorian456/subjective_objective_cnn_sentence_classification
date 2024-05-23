#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import string

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/subj-obj/all_obj.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/subj-obj/all_subj.txt", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

vocab_file = "./corola.100.50.vec"

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

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

    # unfound_word_id = len(vocab_id_dict.keys()) + 1
    # unfound_words_dict = {}
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
            


            # if word not in vocab_id_dict.keys():
            #     if word not in unfound_words_dict.keys():
            #         unfound_words_dict[word] = unfound_word_id
            #         unfound_word_id += 1
            #         #word_array_value = unfound_word_id # TODO: decomment and find solution
            #     else:
            #         #word_array_value = unfound_words_dict[word]
            #         pass
            # else:
            #     word_array_value = vocab_id_dict[word]
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

# def create_embed_tensor(vocab_id_dict, vocab_embed_dict):
#     embed_tensor = tf.Variable(tf.zeros([max(vocab_id_dict.values())+1, max([len(embed) for embed in vocab_embed_dict.values()])]))
#     print("Embed tensor shape: ", embed_tensor.shape)
#     iter = 0
#     for word, word_id in vocab_id_dict.items():
#         # embed_tensor[word_id, :] = vocab_embed_dict[word]
#         # embed_tensor = tf.tensor_scatter_nd_update(embed_tensor, [[word_id]], [tf.constant(vocab_embed_dict[word])])
#         embed_tensor[word_id-1].assign(tf.constant(vocab_embed_dict[word]))
#         if iter%5000 == 0:
#             print(f"Generating embed tensor: {int(iter/len(vocab_id_dict.keys())*100)}% Done")
#         iter +=1
#     print("Done generating embed tensor")
#     # print(embed_tensor)
#     return embed_tensor

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

def train(x_train, y_train, x_dev, y_dev, embed_tensor):
    # Training
    # ==================================================

    # with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=int(embed_tensor.shape[0]),
                embedding_size=int(embed_tensor.shape[1]),
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                embed_tensor=embed_tensor)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, x_dev, y_dev, embed_tensor, _ = preprocess()
    train(x_train, y_train, x_dev, y_dev, embed_tensor)

if __name__ == '__main__':
    tf.app.run()
