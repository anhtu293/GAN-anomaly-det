import tensorflow as tf
from tensorflow.keras import layers
import os


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_layers = config.get('enc_num_layers', 4)
        self.channels = config.get('enc_channels', [64, 128, 256, 512])
        self.conv = [tf.keras.layers.Conv2D(self.channels[0], (5, 5),
                                            strides=(2, 2), padding='same')]
        self.conv.extend([tf.keras.layers.Conv2D(self.channels[i], (3, 3),
                                            strides=(2, 2), padding='same')
                        for i in range(1, self.num_layers)])
        self.batch_norm = [tf.keras.layers.BatchNormalization()
                            for i in range(self.num_layers)]
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, x, training=False):
        for i in range(self.num_layers):
            x = self.conv[i](x)
            x = self.batch_norm[i](x, training=training)
            x = self.lrelu(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.num_layers = config.get('dec_num_layers', 4)
        self.channels = config.get('dec_channels', [256, 128, 64, 3])
        self.dconv = [tf.keras.layers.Conv2DTranspose(self.channels[i], (3, 3),
                                                      strides=(2, 2), padding='same')
                        for i in range(self.num_layers)]
        self.conv = tf.keras.layers.Conv2D(3, (3, 3), 
                                           padding='same', activation='sigmoid')
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, x, training=False):
        for i in range(self.num_layers):
            x = self.dconv[i](x)
            x = self.lrelu(x)
            # if i < self.num_layers - 1:
            #     x = self.lrelu(x)
            # else:
            #     x = tf.keras.activations.sigmoid(x)
        x = self.conv(x)
        return x


class Generator(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def call(self, x, training=False):
        x = self.encoder(x, training)
        x = self.decoder(x, training)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.num_layers = config.get('D_num_layers', 4)
        self.channels = config.get('D_channels', [64, 128, 256, 512])
        self.conv = [tf.keras.layers.Conv2D(self.channels[i], (3, 3),
                                            strides=(2, 2), padding='same')
                        for i in range(self.num_layers)]
        self.batch_norm = [tf.keras.layers.BatchNormalization()
                            for i in range(self.num_layers)]
        self.dropouts = [tf.keras.layers.Dropout(0.25)
                            for i in range(self.num_layers - 1)]
        self.fc = tf.keras.layers.Dense(1)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, x, training=False):
        for i in range(self.num_layers):
            x = self.conv[i](x)
            x = self.batch_norm[i](x, training=training)
            if i < self.num_layers - 1:
                x = tf.keras.activations.relu(x)
                x = self.dropouts[i](x, training=training)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc(x)
        return x


class OneClassClassifier(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(OneClassClassifier, self).__init__(**kwargs)
        self.generator = Generator(config)
        self.detector = Discriminator(config)

    def call(self, x, training=False):
        generated_imgs = self.generator(x, training)
        x = self.detector(generated_imgs, training)
        return generated_imgs, x


if __name__ == '__main__':
    config = {}
    classifier = OneClassClassifier(config)
    x = tf.random.normal((1, 64, 64, 3))
    x = classifier(x)
    print(x)
