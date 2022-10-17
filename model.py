import keras.layers
from keras.constraints import maxnorm


class Model:

    def __init__(self, num_classes=10, activation='relu', padding='same', dropout=0.2, shape=(28, 28, 1),
                 pool_size=(2, 2), kernel_size=(3, 3)):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.inputs = keras.Input(shape=shape)
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.dropout = dropout
        self.pool_size = pool_size

    def make_stem(self, filters=[32, 32, 64], shape=(32, 32, 3)):
        filter1, filter2, filter3 = filters
        stem = keras.layers.Conv2D(filter1, self.kernel_size, input_shape=shape, activation=self.activation,
                                   padding=self.padding)(self.inputs)
        stem = keras.layers.Conv2D(filter2, self.kernel_size, input_shape=shape, activation=self.activation,
                                   padding=self.padding)(stem)
        stem = keras.layers.Conv2D(filter3, self.kernel_size, input_shape=shape, activation=self.activation,
                                   padding=self.padding)(stem)
        stem = keras.layers.MaxPooling2D(self.kernel_size, strides=(2, 2), padding=self.padding)(stem)
        return stem

    def make_skip_connection(self, input, filter=64):
        skip = keras.layers.Conv2D(filter, self.kernel_size, activation=self.activation, padding=self.padding)(input)
        layer = keras.layers.Dropout(self.dropout)(skip)
        layer = keras.layers.Conv2D(filter, self.kernel_size, padding=self.padding)(layer)
        merge = keras.layers.add([layer, skip])
        activation = keras.layers.Activation('relu')(merge)
        return activation

    def make_main_block(self, input, filter):
        block = keras.layers.Conv2D(filter, self.kernel_size, activation=self.activation, padding=self.padding)(input)
        block = keras.layers.Dropout(self.dropout)(block)
        block = keras.layers.Conv2D(filter, self.kernel_size, activation=self.activation, padding=self.padding)(block)
        block = keras.layers.MaxPooling2D(pool_size=self.pool_size)(block)
        return block

    def make_dense_dropout(self, input, filter, kernel_constraint=maxnorm(3)):
        dense = keras.layers.Dense(filter, activation=self.activation, kernel_constraint=kernel_constraint)(input)
        dropout = keras.layers.Dropout(self.dropout)(dense)
        return dropout

    def make_model(self):
        output = self.make_stem()

        output = self.make_skip_connection(output)

        output = keras.layers.MaxPooling2D(pool_size=self.pool_size)(output)

        output = self.make_main_block(output, 128)
        output = self.make_main_block(output, 256)

        output = keras.layers.Flatten()(output)
        output = keras.layers.Dropout(self.dropout)(output)

        output = self.make_dense_dropout(output, 1024)
        output = self.make_dense_dropout(output, 512)

        output = keras.layers.Dense(self.num_classes, activation='softmax')(output)

        return keras.Model(inputs=self.inputs, outputs=output)

    def compile_model(self, model):
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def build_model(self):
        model = self.make_model()
        print(model.summary())
        self.compile_model(model)
        return model
