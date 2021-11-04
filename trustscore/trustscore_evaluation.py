# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.metrics import precision_recall_curve
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def run_logistic(X_train, y_train, X_test, y_test, get_training=False):
  model = LogisticRegression()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  all_confidence = model.predict_proba(X_test)
  confidences = all_confidence[range(len(y_pred)), y_pred]
  if not get_training:
    return y_pred, confidences
  y_pred_training = model.predict(X_train)
  all_confidence_training = model.predict_proba(X_train)
  confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                y_pred_training]
  return y_pred, confidences, y_pred_training, confidence_training


def run_linear_svc(X_train, y_train, X_test, y_test, get_training=False):
  model = LinearSVC()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  all_confidence = model.decision_function(X_test)
  confidences = all_confidence[range(len(y_pred)), y_pred]
  if not get_training:
    return y_pred, confidences
  y_pred_training = model.predict(X_train)
  all_confidence_training = model.decision_function(X_train)
  confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                y_pred_training]
  return y_pred, confidences, y_pred_training, confidence_training


def run_random_forest(X_train, y_train, X_test, y_test, get_training=False):
  model = RandomForestClassifier()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  all_confidence = model.predict_proba(X_test)
  confidences = all_confidence[range(len(y_pred)), y_pred]
  if not get_training:
    return y_pred, confidences
  y_pred_training = model.predict(X_train)
  all_confidence_training = model.predict_proba(X_train)
  confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                y_pred_training]
  return y_pred, confidences, y_pred_training, confidence_training


def run_simple_NN(X,
                  y,
                  X_test,
                  y_test,
                  num_iter=10000,
                  hidden_units=100,
                  learning_rate=0.05,
                  batch_size=100,
                  display_steps=1000,
                  n_layers=1,
                  get_training=False):
  """Run a NN with a single layer on some data.

  Returns the predicted values as well as the confidences.
  """
  n_labels = np.max(y) + 1
  n_features = X.shape[1]

  x = tf.placeholder(tf.float32, [None, n_features])
  y_ = tf.placeholder(tf.float32, [None, n_labels])

  def simple_NN(input_placeholder, n_layers):

    W_in = weight_variable([n_features, hidden_units])
    b_in = bias_variable([hidden_units])
    W_mid = [
        weight_variable([hidden_units, hidden_units])
        for i in range(n_layers - 1)
    ]
    b_mid = [bias_variable([hidden_units]) for i in range(n_layers - 1)]
    W_out = weight_variable([hidden_units, n_labels])
    b_out = bias_variable([n_labels])

    layers = [tf.nn.relu(tf.matmul(input_placeholder, W_in) + b_in)]
    for i in range(n_layers - 1):
      layer = tf.nn.relu(tf.matmul(layers[-1], W_mid[i]) + b_mid[i])
      layers.append(layer)

    logits = tf.matmul(layers[-1], W_out) + b_out
    return logits

  NN_logits = simple_NN(x, n_layers)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=NN_logits))
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(NN_logits, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def one_hot(ns):
    return np.eye(n_labels)[ns]

  y_onehot = one_hot(y)
  y_test_onehot = one_hot(y_test)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iter):
      ns = np.random.randint(0, len(X), size=batch_size)
      if (i + 1) % display_steps == 0:
        train_accuracy = accuracy.eval(feed_dict={x: X, y_: y_onehot})
        test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test_onehot})

        print("step %d, training accuracy %g, test accuracy %g" %
              (i + 1, train_accuracy, test_accuracy))
      train_step.run(feed_dict={x: X[ns, :], y_: y_onehot[ns, :]})

    testing_logits = NN_logits.eval(feed_dict={x: X_test})
    testing_prediction = tf.argmax(NN_logits, 1).eval(feed_dict={x: X_test})
    NN_softmax = tf.nn.softmax(NN_logits).eval(feed_dict={x: X_test})
    testing_confidence_raw = tf.reduce_max(NN_softmax,
                                           1).eval(feed_dict={x: X_test})

    if not get_training:
      return testing_prediction, testing_confidence_raw
    training_prediction = tf.argmax(NN_logits, 1).eval(feed_dict={x: X})
    NN_softmax = tf.nn.softmax(NN_logits).eval(feed_dict={x: X})
    training_confidence_raw = tf.reduce_max(NN_softmax,
                                            1).eval(feed_dict={x: X})
    return testing_prediction, testing_confidence_raw, training_prediction, training_confidence_raw


def plot_precision_curve(
    extra_plot_title,
    percentile_levels,
    signal_names,
    final_TPs,
    final_stderrs,
    final_misclassification,
    model_name="Model",
    colors=["blue", "darkorange", "brown", "red", "purple"],
    legend_loc=None,
    figure_size=None,
    ylim=None):
  if figure_size is not None:
    plt.figure(figsize=figure_size)
  title = "Precision Curve" if extra_plot_title == "" else extra_plot_title
  plt.title(title, fontsize=20)
  colors = colors + list(cm.rainbow(np.linspace(0, 1, len(final_TPs))))

  plt.xlabel("Percentile level", fontsize=18)
  plt.ylabel("Precision", fontsize=18)
  for i, signal_name in enumerate(signal_names):
    ls = "--" if ("Model" in signal_name) else "-"
    plt.plot(
        percentile_levels, final_TPs[i], ls, c=colors[i], label=signal_name)

    plt.fill_between(
        percentile_levels,
        final_TPs[i] - final_stderrs[i],
        final_TPs[i] + final_stderrs[i],
        color=colors[i],
        alpha=0.1)

  if legend_loc is None:
    if 0. in percentile_levels:
      plt.legend(loc="lower right", fontsize=14)
    else:
      plt.legend(loc="upper left", fontsize=14)
  else:
    if legend_loc == "outside":
      plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=14)
    else:
      plt.legend(loc=legend_loc, fontsize=14)
  if ylim is not None:
    plt.ylim(*ylim)
  model_acc = 100 * (1 - final_misclassification)
  plt.axvline(x=model_acc, linestyle="dotted", color="black")
  plt.show()


def run_precision_recall_experiment_general(X,
                                            y,
                                            n_repeats,
                                            percentile_levels,
                                            trainer,
                                            test_size=0.5,
                                            extra_plot_title="",
                                            signals=[],
                                            signal_names=[],
                                            predict_when_correct=False,
                                            skip_print=False):

  def get_stderr(L):
    return np.std(L) / np.sqrt(len(L))

  all_signal_names = ["Model Confidence"] + signal_names
  all_TPs = [[[] for p in percentile_levels] for signal in all_signal_names]
  misclassifications = []
  sign = 1 if predict_when_correct else -1
  sss = StratifiedShuffleSplit(
      n_splits=n_repeats, test_size=test_size, random_state=0)
  for train_idx, test_idx in sss.split(X, y):
    X_train = X[train_idx, :]
    y_train = y[train_idx]
    X_test = X[test_idx, :]
    y_test = y[test_idx]
    testing_prediction, testing_confidence_raw = trainer(
        X_train, y_train, X_test, y_test)
    target_points = np.where(
        testing_prediction == y_test)[0] if predict_when_correct else np.where(
            testing_prediction != y_test)[0]

    final_signals = [testing_confidence_raw]
    for signal in signals:
      signal.fit(X_train, y_train)
      final_signals.append(signal.get_score(X_test, testing_prediction))

    for p, percentile_level in enumerate(percentile_levels):
      all_high_confidence_points = [
          np.where(sign * signal >= np.percentile(sign *
                                                  signal, percentile_level))[0]
          for signal in final_signals
      ]

      if 0 in map(len, all_high_confidence_points):
        continue
      TP = [
          len(np.intersect1d(high_confidence_points, target_points)) /
          (1. * len(high_confidence_points))
          for high_confidence_points in all_high_confidence_points
      ]
      for i in range(len(all_signal_names)):
        all_TPs[i][p].append(TP[i])
    misclassifications.append(len(target_points) / (1. * len(X_test)))

  final_TPs = [[] for signal in all_signal_names]
  final_stderrs = [[] for signal in all_signal_names]
  for p, percentile_level in enumerate(percentile_levels):
    for i in range(len(all_signal_names)):
      final_TPs[i].append(np.mean(all_TPs[i][p]))
      final_stderrs[i].append(get_stderr(all_TPs[i][p]))

    if not skip_print:
      print("Precision at percentile", percentile_level)
      ss = ""
      for i, signal_name in enumerate(all_signal_names):
        ss += (signal_name + (": %.4f  " % final_TPs[i][p]))
      print(ss)
      print()

  final_misclassification = np.mean(misclassifications)

  if not skip_print:
    print("Misclassification rate mean/std", np.mean(misclassifications),
          get_stderr(misclassifications))

  for i in range(len(all_signal_names)):
    final_TPs[i] = np.array(final_TPs[i])
    final_stderrs[i] = np.array(final_stderrs[i])

  plot_precision_curve(extra_plot_title, percentile_levels, all_signal_names,
                       final_TPs, final_stderrs, final_misclassification)
  return (all_signal_names, final_TPs, final_stderrs, final_misclassification)
