import tensorflow as tf
import data_helper
from tensorflow.contrib import learn
import numpy as np
from rcnn import TextRCNN
import warnings
warnings.filterwarnings('ignore')

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string('data_file_path', './data/rt-polarity.csv', 'Data source')
tf.flags.DEFINE_string('feature_name', 'comment_text', 'The name of feature column')
tf.flags.DEFINE_string('label_name', 'label', 'The name of label column')
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_string("cell_type", "lstm", "Type of RNN cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)")
tf.flags.DEFINE_integer("word_embedding_dim", 256, "Dimensionality of word embedding (Default: 300)")
tf.flags.DEFINE_integer("context_embedding_dim", 256, "Dimensionality of context embedding")
tf.flags.DEFINE_integer("hidden_size", 128, "Size of hidden layer (Default: 512)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.9, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 20)")

FLAGS = tf.flags.FLAGS


def pre_process():
    # load data
    x_text, y = data_helper.load_data_and_labels(FLAGS.data_file_path, FLAGS.feature_name, FLAGS.label_name)
    # Build vocabulary and cut or extend sentence to fixed length
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # replace the word using the index of word in vocabulary
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # random shuffle data
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

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


if __name__ == '__main__':
    x_train, y_train, vocab_processor, x_dev, y_dev = pre_process()
    # train step
    rcnn = TextRCNN(sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    word_embedding_size=FLAGS.word_embedding_dim,
                    context_embedding_size=FLAGS.context_embedding_dim,
                    cell_type=FLAGS.cell_type,
                    hidden_size=FLAGS.hidden_size,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
    # optimizer
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(rcnn.loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        # train
        for epoch_i in range(FLAGS.num_epochs):
            for batch_i, (x_batch, y_batch) in enumerate(data_helper.get_batches(x_train, y_train, FLAGS.batch_size)):
                _, acc, loss = sess.run([optimizer, rcnn.accuracy, rcnn.loss],
                                        feed_dict={rcnn.input_text: x_batch, rcnn.input_y: y_batch,
                                                   rcnn.dropout_keep_prob: FLAGS.dropout_keep_prob})
                if batch_i % 10 == 0:
                    print('Epoch {}/{}, Batch {}/{}, loss: {}, accuracy: {}'.format(epoch_i, FLAGS.num_epochs, batch_i,
                                                                                    len(x_train) // FLAGS.batch_size,
                                                                                    loss, acc))
        # save model
        saver.save(sess, './model/model_20200311.ckpt')
        print('model stored!')

        # valid step
        saver.restore(sess, "./model/model_20200311.ckpt")
        print('valid accuracy: {}'.format(sess.run(rcnn.accuracy,
                                                   feed_dict={rcnn.input_text: x_dev, rcnn.input_y: y_dev,
                                                              rcnn.dropout_keep_prob: 1.})))
