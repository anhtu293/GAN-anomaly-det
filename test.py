import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
import json
import os
import shutil
from tqdm import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

import config
from src.data_loader import DataGenerator
from src.dcgans import OneClassClassifier, Generator, Discriminator
from src.loss import loss_function, dcgans_discriminator_loss, dcgans_generator_loss
from utils import Config, TransformerScheduler


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config',
                        help='path to config file')
    parser.add_argument('gpu',
                        help='id of gpu')
    parser.add_argument('gen_cp',
                         help='checkpoint for generator')
    parser.add_argument('dis_cp',
                         help='checkpoint for discriminator')
    parser.add_argument('model_name',
                         help='model name')                     
    args = parser.parse_args()
    return args


def plot_roc(labels, preds, auc, model_name='DCGANS'):
    labels = labels[:, 0]
    preds = preds[:, 0]
    fpr, tpr, thresholds = roc_curve(labels, preds)

    # plot
    f, ax = plt.subplots(figsize=(12, 12))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(fpr, tpr, label='{} (area = {:.3f})'.format(model_name, auc))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')
    ax.legend(loc='best')
    return f, ax


def test():
    # get config
    train_cfg = Config(config.train_cfg)
    model_cfg = Config(config.model_cfg)

    # Init model
    generator = Generator(model_cfg)
    discriminator = Discriminator(model_cfg)
    discriminator.load_weights(args.dis_cp)
    generator.load_weights(args.gen_cp)

    # Init data generators
    pos_test_generator = DataGenerator(path=train_cfg.pos_test_set,
                                       batch_size=train_cfg.batch_size,
                                       img_size=train_cfg.img_size)
    neg_test_generator = DataGenerator(path=train_cfg.neg_test_set,
                                       batch_size=train_cfg.batch_size,
                                       img_size=train_cfg.img_size)
    test_generators = [pos_test_generator, neg_test_generator]

    # Init metric
    metric = tf.keras.metrics.AUC()

    # Evaluate
    all_labels = []
    all_preds = []
    for idx, test_generator in enumerate(test_generators):
        for step in range(len(test_generator)):
            org_imgs, noisy_imgs = test_generator[step]

            # forward
            gen_imgs = generator(org_imgs)
            preds = discriminator(gen_imgs)
            preds = tf.math.sigmoid(preds)

            # commulate
            labels = tf.ones_like(preds) if idx == 0 else tf.zeros_like(preds)
            metric.update_state(labels, preds)
            all_labels.append(labels)
            all_preds.append(preds)

    # Get auc
    auc = metric.result().numpy()

    # Concatenate labels and preds for ROC plot
    all_labels = tf.concat(all_labels, 0)
    all_preds = tf.concat(all_preds, 0)

    fig, roc = plot_roc(all_labels, all_preds, auc)
    fig.savefig('{}.png'.format(args.model_name))
    plt.close(fig)


if __name__ == '__main__':
    args = parse_args()
    test()
