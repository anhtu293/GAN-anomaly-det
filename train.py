import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
import json
import os
import shutil
from tqdm import tqdm

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
    args = parser.parse_args()
    return args


def main():
    # get config
    train_cfg = Config(config.train_cfg)
    model_cfg = Config(config.model_cfg)
    if not os.path.exists(train_cfg.work_dirs):
        os.makedirs(train_cfg.work_dirs)
    logs_dir = os.path.join(train_cfg.work_dirs, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    shutil.copyfile('config.py',
                    os.path.join(train_cfg.work_dirs, 'config.py'))

    # checkpoint dir
    checkpoint_dir = os.path.join(train_cfg.work_dirs, 'per_epoch_cp')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    gen_checkpoint_dir = os.path.join(checkpoint_dir, 'gen')
    if not os.path.exists(gen_checkpoint_dir):
        os.makedirs(gen_checkpoint_dir)
    det_checkpoint_dir = os.path.join(checkpoint_dir, 'det')
    if not os.path.exists(det_checkpoint_dir):
        os.makedirs(det_checkpoint_dir)

    # Init model
    generator = Generator(model_cfg)
    detector = Discriminator(model_cfg)

    # Init data generators
    train_generator = DataGenerator(path=train_cfg.train_set,
                                    batch_size=train_cfg.batch_size,
                                    img_size=train_cfg.img_size)
    val_generator = DataGenerator(path=train_cfg.val_set,
                                      batch_size=train_cfg.batch_size,
                                      img_size=train_cfg.img_size)

    # Init optimizer
    d_lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        train_cfg.d_lr,
        decay_steps=50000,
        decay_rate=0.96,
        staircase=True)
    g_lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        train_cfg.g_lr,
        decay_steps=50000,
        decay_rate=0.96,
        staircase=True)
    d_optimizer = getattr(tf.keras.optimizers, train_cfg.d_optimizer)(learning_rate=d_lr_scheduler)
    g_optimizer = getattr(tf.keras.optimizers, train_cfg.g_optimizer)(learning_rate=g_lr_scheduler)

    # Init AUC metric
    auc = tf.keras.metrics.AUC()

    # Init logs writer
    train_log_dir = os.path.join(logs_dir, 'train')
    test_log_dir = os.path.join(logs_dir, 'test')
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if not os.path.exists(test_log_dir):
        os.makedirs(test_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # training loop
    steps_per_epoch = len(train_generator)
    best_loss = 1e6
    label_smoothing = False
    for epoch in range(train_cfg.epochs):
        print('Epoch {}/{} \n'.format(epoch, train_cfg.epochs))

        # shuffle dataset
        train_generator.shuffle()

        for step in range(steps_per_epoch):
            # get mini batch
            org_imgs, noisy_imgs = train_generator[step]

            # Train
            all_d_loss = []
            all_g_loss = []
            all_l1_loss = []
            all_d_ce_loss = []

            # Forward
            with tf.GradientTape() as gen_tape, tf.GradientTape() as det_tape:
                gen_imgs = generator(noisy_imgs)
                real_output = detector(org_imgs, training=True)
                fake_output = detector(gen_imgs, training=True)

                ce_loss, l1_loss, g_loss = dcgans_generator_loss(org_imgs, gen_imgs, fake_output, train_cfg.alpha)
                d_loss = dcgans_discriminator_loss(real_output, fake_output, label_smoothing=label_smoothing)

            # Back propagation
            gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
            gradients_of_discriminator = det_tape.gradient(d_loss, detector.trainable_variables)
            g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            d_optimizer.apply_gradients(zip(gradients_of_discriminator, detector.trainable_variables))
            all_d_loss.append(d_loss)
            all_g_loss.append(g_loss)
            all_l1_loss.append(l1_loss)
            all_d_ce_loss.append(ce_loss)

            mean_d_loss = tf.math.reduce_mean(all_d_loss).numpy()
            mean_g_loss = tf.math.reduce_mean(all_g_loss).numpy()
            mean_l1_loss = tf.math.reduce_mean(all_l1_loss).numpy()
            mean_d_ce_loss = tf.math.reduce_mean(all_d_ce_loss).numpy()

            print('Epoch {}: Step {}/{} --> detector loss {:.3f} / reconstructor loss {:.3f} - l1 loss {:.3f} - ce {:.3f}'
                                    .format(epoch, step, steps_per_epoch, mean_d_loss, mean_g_loss, mean_l1_loss, mean_d_ce_loss))

            # log detector loss and generator loss accuracy
            overal_steps = epoch*steps_per_epoch + step
            with train_summary_writer.as_default():
                tf.summary.scalar('detector loss', mean_d_loss, step=overal_steps)
                tf.summary.scalar('generator loss', mean_g_loss, step=overal_steps)
                tf.summary.scalar('generator l1 loss', mean_l1_loss, step=overal_steps)
                tf.summary.scalar('generator CE', mean_d_ce_loss, step=overal_steps)

        # evaluate
        all_accuracy = []
        all_generator_loss = []
        all_detector_loss = []
        all_l1_loss = []
        all_d_ce_loss = []
        print('Evaluating ... \n') 
        for step in range(len(val_generator)):
            org_imgs, noisy_imgs = val_generator[step]
            org_img_preds = detector(org_imgs)
            gen_imgs = generator(noisy_imgs)
            gen_img_preds = detector(gen_imgs)
            d_loss = dcgans_discriminator_loss(org_img_preds,
                                                gen_img_preds)
            ce, l1_loss, g_loss = dcgans_generator_loss(org_imgs,
                                                        gen_imgs,
                                                        gen_img_preds,
                                                        train_cfg.alpha)

            # validation AUC
            gen_imgs = generator(org_imgs)
            preds = detector(gen_imgs)
            preds = tf.math.sigmoid(preds)
            labels = tf.zeros_like(preds)
            auc.update_state(labels, preds)

            preds = detector(org_imgs)
            preds = tf.math.sigmoid(preds)
            labels = tf.ones_like(preds)
            auc.update_state(labels, preds) 

            # save loss
            all_generator_loss.append(g_loss)
            all_detector_loss.append(d_loss)
            all_l1_loss.append(l1_loss)
            all_d_ce_loss.append(ce)

        # log generator loss and detector loss
        mean_auc = auc.result().numpy()
        auc.reset_states()
        mean_generator_loss = tf.math.reduce_mean(tf.concat(all_generator_loss, axis=0)).numpy()
        mean_detector_loss = tf.math.reduce_mean(tf.concat(all_detector_loss, axis=0)).numpy()
        mean_l1_loss = tf.math.reduce_mean(tf.concat(all_l1_loss, axis=0)).numpy()
        mean_ce_loss = tf.math.reduce_mean(tf.concat(all_d_ce_loss, axis=0)).numpy()

        with test_summary_writer.as_default():
            tf.summary.scalar('detector loss', mean_detector_loss, step=(epoch + 1) * steps_per_epoch)
            tf.summary.scalar('generator loss', mean_generator_loss, step=(epoch + 1) * steps_per_epoch)
            tf.summary.scalar('generator l1 loss', mean_l1_loss, step=(epoch + 1) * steps_per_epoch)
            tf.summary.scalar('generator CE', mean_ce_loss, step=(epoch + 1) * steps_per_epoch)
            tf.summary.scalar('AUC', mean_auc, step=(epoch + 1) * steps_per_epoch)

        print('Epoch {}: detector loss {:.3f} / generator loss {:.3f} - l1 loss {:.3f} - ce {:.3f} / AUC {:.3f}'
                    .format(epoch, mean_detector_loss, mean_generator_loss, mean_l1_loss , mean_ce_loss, mean_auc))

        # save checkpoint
        checkpoint_name = os.path.join(gen_checkpoint_dir,
                                        'epoch_{}_gloss_{:.3f}_dloss_{:.3f}_gl1loss_{:.3f}_gCE_{:.3f}_auc_{:.3f}'
                                        .format(epoch, mean_generator_loss, mean_detector_loss, mean_l1_loss, mean_ce_loss, mean_auc))
        generator.save_weights(checkpoint_name)

        checkpoint_name = os.path.join(det_checkpoint_dir,
                                        'epoch_{}_gloss_{:.3f}_dloss_{:.3f}_gl1loss_{:.3f}_gCE_{:.3f}_auc_{:.3f}'
                                        .format(epoch, mean_generator_loss, mean_detector_loss, mean_l1_loss, mean_ce_loss, mean_auc))
        detector.save_weights(checkpoint_name)

        # check if best, save as best model
        if mean_l1_loss <= best_loss:
            best_loss = mean_l1_loss
            generator.save_weights(os.path.join(train_cfg.work_dirs, 'gen_best_model'))
            detector.save_weights(os.path.join(train_cfg.work_dirs, 'det_best_model'))


if __name__ == '__main__':
    args = parse_args()

    # choose gpu and allow memory
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    main()
