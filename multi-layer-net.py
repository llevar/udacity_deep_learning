from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import math

pickle_file = '/Users/siakhnin/Documents/workspace/udacity_deep_learning/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
#  print('Training set', train_dataset.shape, train_labels.shape)
#  print('Validation set', valid_dataset.shape, valid_labels.shape)
#  print('Test set', test_dataset.shape, test_labels.shape)
  
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
#print('Training set', train_dataset.shape, train_labels.shape)
#print('Validation set', valid_dataset.shape, valid_labels.shape)
#print('Test set', test_dataset.shape, test_labels.shape)

max_steps = 1000
learning_rate = 0.001
dropout = 0.9
data_dir = '/tmp/data'
summaries_dir = '/tmp/mnist_logs'
batch_size = 128
num_features = image_size * image_size

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var, name):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read, and
    adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights, layer_name + '/weights')
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases, layer_name + '/biases')
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
      activations = act(preactivate, 'activation')
      tf.histogram_summary(layer_name + '/activations', activations)
      return tf.nn.dropout(activations, keep_prob), weights


graph = tf.Graph()
with graph.as_default():

    with tf.name_scope("input"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="x-input")
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.image_summary('input', image_shaped_input, 10)
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="y-input")
        keep_prob = tf.placeholder(dtype=tf.float32, name="dropout-keep-prob")
        tf.scalar_summary('dropout_keep_probability', keep_prob)
       
        

    
    
    layer1, weights_1 = nn_layer(x, num_features, 1024, 'layer1')
    layer2, weights_2 = nn_layer(layer1, 1024, num_labels, 'layer2')
    y = tf.nn.softmax(layer2, 'predictions')
    #layer1, weights_1 = nn_layer(x, num_features, 1024, 'layer1')
    #layer2, weights_2 = nn_layer(layer1, 1024, 300, 'layer2')
    #layer3, weights_3 = nn_layer(layer2, 300, 50, 'layer3')
    #layer4, weights_4 = nn_layer(layer3, 50, num_labels, 'layer4')
    #y = tf.nn.softmax(layer4, 'predictions')
    with tf.name_scope('train'):
        with tf.name_scope('cross_entropy'):
            diff = y_ * tf.log(y)
            with tf.name_scope('total'):
                cross_entropy = -tf.reduce_sum(diff)
            with tf.name_scope('normalized'):
                normalized_cross_entropy = -tf.reduce_mean(diff)
            tf.scalar_summary('cross entropy', normalized_cross_entropy)
        
        reg_rate = tf.placeholder(tf.float32, name="regularization-rate")
        regularizer = reg_rate * (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_1))
        #regularizer = reg_rate * (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4))
        loss_func = cross_entropy + regularizer
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.001, global_step, max_steps, 0.96)
        #learning_rate = learning_rate * math.pow(0.96, float(current_step)/1000)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_func, global_step=global_step)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

with  tf.Session(graph=graph) as my_session:
    
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(summaries_dir + '/train', my_session.graph)
    test_writer = tf.train.SummaryWriter(summaries_dir + '/test')


    tf.initialize_all_variables().run()
    
    train_accuracy = []
    train_epochs = []
    validation_accuracy = []
    validation_epochs = []


    for step in range(max_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        if step % 500 == 0:
            feed_dict = {x: valid_dataset, y_: valid_labels, keep_prob: 1}
            summary, my_accuracy = my_session.run([merged, accuracy], feed_dict=feed_dict)
            validation_accuracy.append(my_accuracy)
            validation_epochs.append(step)
            test_writer.add_summary(summary, step)
            print('Validation accuracy at step %s: %s' % (step, my_accuracy))
            
        else:
            feed_dict = {x: batch_data, y_: batch_labels, keep_prob: dropout, reg_rate: 0.001}
            summary, my_accuracy, _ = my_session.run([merged, accuracy, train_step], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            train_accuracy.append(my_accuracy)
            train_epochs.append(step)
    
    fig,ax = plt.subplots(1,1)
    train_time_subsample = np.array(train_epochs[::200])
    train_data_subsample = np.array(train_accuracy[::200])
    x_smooth = np.linspace(train_time_subsample[0], train_time_subsample[-1], 300)
    y_smooth = spline(train_time_subsample,train_data_subsample,x_smooth)
    
    ax.plot(x_smooth,y_smooth, 'g-', label="Training Accuracy")
    ax.plot(validation_epochs,validation_accuracy, 'ro')
    ax.plot(validation_epochs,validation_accuracy, 'b-', label="Validation Accuracy")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, max_steps * 1.05)
    ax.set_ylim(0, 1.1)
   
    
    plt.draw()
       
    feed_dict = {x: test_dataset, y_: test_labels, keep_prob: 1, reg_rate: 0.001}
    summary, my_accuracy = my_session.run([merged, accuracy], feed_dict=feed_dict)
    
    ax.plot(max_steps, my_accuracy, 'go', markersize=5, label="Test Accuracy")
    plt.annotate("Test Accuracy", xy = (max_steps, my_accuracy), xytext=(-20,20), 
                 textcoords='offset points', ha='right', va='bottom',
                 bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                 arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=4)
    plt.draw()
    
    test_writer.add_summary(summary)
    print("Test accuracy: %s" % my_accuracy)
    plt.show()