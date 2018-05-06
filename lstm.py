import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import tensorflow as tf
from string import punctuation
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

train = pd.read_csv("/data/train.csv")
# test = pd.read_csv("test.csv")

# Keep only id, comment_text and toxic
rel_cols = ["id", "comment_text", "toxic"]
train = train.loc[:, rel_cols]

# Now pull out all words
comments = train["comment_text"]

# Function to clean up a single comment
def clean_comment(comment):
    newlines_removed = comment.replace('\n', " ")
    lowercased = newlines_removed.lower()
    cleaned = "".join([i for i in lowercased if i not in punctuation])
    return cleaned

cleaned_comments = comments.apply(func=clean_comment)

# Count all words
combined_text = cleaned_comments.str.cat(sep=" ")

words = combined_text.split(" ")

counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

comments_encoded = []
for comment in cleaned_comments:
    comments_encoded.append([vocab_to_int[word] for word in comment.split(" ")])

comment_length = 200 # This is the length we are going to standardize every comment to
num_comments = len(comments_encoded)

# Create our feature array
features = np.zeros((num_comments, comment_length), dtype=np.int64)
for i, comment in enumerate(comments_encoded):
    original_length = len(comment)
    if original_length > comment_length:
        comment = comment[:comment_length]
    startIndex = comment_length - min(original_length, comment_length)
    features[i, startIndex:comment_length] = comment

# Set up training/test datasets
x = features
y = train["toxic"]
ids = train["id"]
y = y.values.reshape(-1, 1)

split_frac = 0.8

split_index = int(split_frac * len(features))

train_x, val_x = x[:split_index], x[split_index:]
train_y, val_y = y[:split_index], y[split_index:]
train_ids, val_ids = ids[:split_index], ids[split_index:]

split_frac = 0.5
split_index = int(split_frac * len(val_x))

val_x, test_x = val_x[:split_index], val_x[split_index:]
val_y, test_y = val_y[:split_index], val_y[split_index:]
val_ids, test_ids = val_ids[:split_index], val_ids[split_index:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
print("label set: \t\t{}".format(train_y.shape), 
      "\nValidation label set: \t{}".format(val_y.shape),
      "\nTest label set: \t\t{}".format(test_y.shape))

# Start building the graph
lstm_size = 256
lstm_layers = 3
batch_size = 1000
embed_size = 300
learning_rate = 0.01

n_words = len(vocab_to_int) + 1 # Add 1 for 0 added to vocabulary

# Create the graph object
tf.reset_default_graph()
with tf.name_scope('inputs'):
    inputs = tf.placeholder(tf.int64, shape=(None, comment_length), name="inputs")
    labels = tf.placeholder(tf.int64, shape=(None, 1), name="labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob") # Use dropout

with tf.name_scope("embeddings"):
    embeddings = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embeddings, inputs)

# Create LSTM Cells
def lstm_cell():
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

with tf.name_scope("lstm_layers"):
    # Make it a deep network
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
    
    # Initial state
    init_state = cell.zero_state(batch_size, tf.float32)

with tf.name_scope("rnn_forward"):
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=init_state)

with tf.name_scope("predictions"):
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    tf.summary.histogram("predictions", predictions)

with tf.name_scope("loss"):
    loss = tf.losses.mean_squared_error(labels, predictions)
    tf.summary.scalar("loss", loss)
    
with tf.name_scope("train"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

merged = tf.summary.merge_all()

with tf.name_scope("validation"):
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int64), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Batching data
def get_batches(x, y, batch_size=batch_size):
    n_batches = len(x) // batch_size # Note that this removes some excess data
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]

    for i in range(n_batches):
        start, end = i * batch_size, (i+1) * batch_size
        yield x[start:end], y[start:end]

# Training
n_epochs = 20
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./logs/tb/train', sess.graph)
    test_writer = tf.summary.FileWriter('./logs/tb/test', sess.graph)
    
    epoch_losses = []
    epoch_labels = range(1, n_epochs+1)

    iteration = 1
    for epoch in range(n_epochs):
        epoch_loss = []
        state = sess.run(init_state)
        for i, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed_dict = {
                inputs: x,
                labels: y,
                keep_prob: 0.5,
                init_state: state
            }
            fetches = [merged, loss, final_state, train_op]
            summary, loss_val, state, _ = sess.run(fetches, feed_dict=feed_dict)
            epoch_loss.append(loss_val)
            
            train_writer.add_summary(summary, iteration)
            if iteration % 5 == 0:
                print("Epoch: {}/{}".format(epoch + 1, n_epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss_val))
            if iteration % 25 == 0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed_dict = {
                        inputs: x,
                        labels: y,
                        keep_prob: 1,
                        init_state: val_state
                    }
                    summary, batch_acc, val_state = sess.run([merged, accuracy, 
                                                              final_state], feed_dict=feed_dict)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration += 1
            test_writer.add_summary(summary, iteration)
        epoch_losses.append(np.mean(epoch_loss))
    
    # Run validation set
    test_acc = []
    preds = np.array([])
    valid_y = np.array([])
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for x, y in get_batches(test_x, test_y, batch_size):

        reshaped_y = np.array(y).reshape(-1,)
        valid_y = np.append(valid_y, reshaped_y)

        feed_dict = {
            inputs: x,
            labels: y,
            keep_prob: 1,
            init_state: test_state
        }
        summary, batch_acc, val_state, pred_vals = sess.run([merged, accuracy, 
                                                    final_state, predictions], feed_dict=feed_dict)
        
        reshaped_preds = np.array(pred_vals).reshape(-1,)
        preds = np.append(preds, reshaped_preds)

        test_acc.append(batch_acc)
    print("Test acc: {:.3f}".format(np.mean(test_acc)))

    preds = np.array([1 if i > 0.5 else 0 for i in preds])

    # Print confusion matrix
    cm = confusion_matrix(valid_y, preds)
    print(cm)
    valid_ids = test_ids[:len(valid_y)]
    # Save all wrong examples
    test_results = pd.DataFrame({
        "y": valid_y,
        "prediction": preds,
        "ids": valid_ids
    })

    test_results.to_csv("/output/final.csv")

    # Plot loss
    plt.figure()
    plt.plot(epoch_labels, epoch_losses)
    plt.savefig("/output/losses.png", dpi=75)

    saver.save(sess, "/output/model.ckpt")
