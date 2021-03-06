import tensorflow as tf


@tf.function
def dcgans_discriminator_loss(org_img_preds, gen_img_preds,
                  label_smoothing=True, flip_label=True):
    """GANS-liked loss
    params:
        - real_img_preds: [batch_size, 1] predictions for real images
        - gen_img_preds: [batch_size, 1] predictions for images generated by gans
    return:
        - CE: scalar - gans-liked loss
    """
    pos_labels = tf.ones_like(org_img_preds)
    neg_labels = tf.zeros_like(org_img_preds)
    if label_smoothing:
        # smooth positive label
        pos_labels = pos_labels - 0.3 + 0.5 * tf.random.uniform(shape=tf.shape(pos_labels),
                                                                maxval=1)
        # smooth negative label
        neg_labels = neg_labels + 0.3 * tf.random.uniform(shape=tf.shape(neg_labels),
                                                          maxval=1)
    loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=pos_labels, logits=org_img_preds)
    loss_2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=gen_img_preds)
    ce = tf.math.reduce_mean(tf.math.add(loss_1, loss_2))
    return ce


@tf.function
def dcgans_generator_loss(imgs, gen_imgs, gen_img_preds, alpha):
    """ Normˆ2 of original img and generated img by Reconstructor
    params:
        - imgs: [batch_size, width, height, channels] - original imgs
        - gen_imgs: [batch_size, width, height, channels] - generated imgs
    return:
        - loss: scalar l2 loss
    """
    pos_labels = tf.ones_like(gen_img_preds)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=pos_labels, logits=gen_img_preds)
    ce = tf.math.reduce_mean(ce)
    l1 = tf.keras.losses.BinaryCrossentropy()(imgs, gen_imgs)

    loss = tf.math.add(ce, alpha * l1)
    return ce, l1, loss


@tf.function
def loss_function(imgs, org_img_preds, gen_imgs, gen_img_preds, alpha=0.4):
    """Loss function for training
    params:
        - imgs: [batch_size, w, h, c] original imgs
        - org_img_preds: [batch_size, 1] predictions for original imgs
        - gen_imgs: [batch_size, w, h, c] generated imgs by Reconstructor
        - gen_img_preds: [batch_size, 1] predictions for generated imgs
        - weights: weights for gans-like loss and l2_loss.
    return:
        - loss: scalar
    """
    d_loss = detector_loss(org_img_preds, gen_img_preds)
    _, _, g_loss = generator_loss(imgs, gen_imgs, gen_img_preds)
    return d_loss, g_loss


if __name__ == '__main__':
    # choose gpu and allow memory
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    imgs = tf.random.normal((8, 64, 64, 3))
    gen_imgs = tf.random.normal((8, 64, 64, 3))
    img_preds = tf.random.uniform((8, 1), minval=0, maxval=1)
    gen_img_preds = tf.random.uniform((8, 1), minval=0, maxval=1)
    tf.print(gans_loss(img_preds, gen_img_preds))
    tf.print(l2_loss(imgs, gen_imgs))
    tf.print(loss_function(imgs, img_preds, gen_imgs, gen_img_preds, weights=[4.0, 1.]))
