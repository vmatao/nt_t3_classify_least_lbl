import numpy as np
from keras.utils import to_categorical


class MistFashionEngine:
    def __init__(self, model):
        super(MistFashionEngine, self).__init__()
        self.model = model.build_model()
        self.accuracy = 0
        self.best_model = None

    def classify_high_confidence(self, x_train_remaining, y_train_remaining, x_train, y_train):
        input = x_train_remaining.reshape(-1, 28, 28, 1).astype('float32') / 255
        predictions = self.model.predict(input)
        certainty = predictions.copy()
        certainty = np.max(certainty, axis=1)
        certainty = np.expand_dims(certainty, axis=1)

        label = predictions.copy()
        label = np.argmax(label, axis=1)

        categ_label = to_categorical(label, num_classes=10)

        certainty_threshold = 0.95
        indices_over_threshold = np.where(np.any(certainty > certainty_threshold, axis=1))

        # add data predicted with high confidence to the train data
        x_train = np.append(x_train, input[indices_over_threshold], axis=0)
        y_train = np.append(y_train, categ_label[indices_over_threshold], axis=0)

        # delete data already labeled
        x_train_remaining = np.delete(x_train_remaining, [indices_over_threshold], axis=0)
        y_train_remaining = np.delete(y_train_remaining, [indices_over_threshold], axis=0)

        print("Remaining unlabeled: " + str(y_train_remaining.size))
        return x_train, y_train, x_train_remaining, y_train_remaining

    def evaluate_model(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print('loss = {}, accuracy = {}'.format(loss, accuracy))
        if self.accuracy < accuracy:
            self.accuracy = accuracy
            self.best_model = self.model
        return accuracy

    def train_model(self, epochs, batch_size, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def save_best_model(self):
        self.best_model.save("best.h5")
