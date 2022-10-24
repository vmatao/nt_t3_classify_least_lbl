--------
TASK 3
--------

In most real-world applications, labelled data is scarce. Suppose you are given
the Fashion-MNIST dataset (https://github.com/zalandoresearch/fashion-mnist), but without any labels
in the training set. The labels are held in a database, which you may query to
reveal the label of any particular image it contains. Your task is to build a classifier to
>90% accuracy on the test set, using the smallest number of queries to this
>database.

You may use any combination of techniques you find suitable
(supervised, self-supervised, unsupervised). However, using other datasets or
pre-trained models is not allowed.

---
Methods of solving from most to least number of queries to the train labels dataset:
---

1. Query all labels and use supervised learning.

2. Semi-supervised / pseudo-labelling - order images data by labels and query for example 500 labels for each class.
Use supervised learning to train on these 1000*number of classes. Then classify the rest keeping the high confidence
labels. Repeat the process until all are labeled and then test. (The method that I've used)
Nr of labels retrieved:~10400/60000   Overall accuracy: 90.2%

3. Use unsupervised learning to attribute the images to n clusters. Where n is the number of classes. Then cluster the
test data, use the most prevailing label as the cluster label and then test against the test labels. Having the requirement of
>90% accuracy makes this method unlikely to succeed as unsupervised learning is not that precise.
