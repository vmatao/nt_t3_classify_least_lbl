import util
import model
import engine
from keras.utils import to_categorical

if __name__ == '__main__':
    x_train_full, y_train_full = util.load_mnist('data', kind='train')
    x_test, y_test = util.load_mnist('data', kind='t10k')
    num_classes = 10

    x_train, y_train, x_train_remaining, y_train_remaining = \
        util.get_data_with_n_labels_for_each_class(x_train_full,
                                                   y_train_full,
                                                   nr_of_labels=1000,
                                                   num_classes=num_classes)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = model.Model()
    engine = engine.MistFashionEngine(model)

    best_model = None
    engine.train_model(epochs=5, batch_size=64, x_train=x_train, y_train=y_train)

    while y_train_remaining.size > 0:
        curr_accuracy = engine.evaluate_model(x_test, y_test)
        if curr_accuracy > 0.9:
            engine.save_best_model()
            break
        x_train, y_train, x_train_remaining, y_train_remaining = engine.classify_high_confidence(x_train_remaining,
                                                                                                 y_train_remaining,
                                                                                                 x_train,
                                                                                                 y_train)
        engine.train_model(batch_size=64, epochs=5, x_train=x_train, y_train=y_train)
    engine.evaluate_model(x_test, y_test)
    engine.save_best_model()
