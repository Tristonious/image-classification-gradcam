import numpy as np
import tensorflow as tf


def grad_cam(model, img_array, layer_name='block5_conv3'):
    """
    Compute a Grad-CAM heatmap for the top predicted class.

    Builds a sub-model that outputs both the target conv layer activations
    and the final predictions, then uses GradientTape to compute the gradient
    of the top-class score with respect to those activations. The channel-wise
    mean of the gradients weights the activation maps to produce the heatmap.

    Returns a 2-D numpy array (H x W) normalized to [0, 1].
    """
    grad_model = tf.keras.models.Model(
        model.input,
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    denom = tf.math.reduce_max(heatmap)
    if denom == 0:
        return heatmap.numpy()
    heatmap /= denom
    return heatmap.numpy()
