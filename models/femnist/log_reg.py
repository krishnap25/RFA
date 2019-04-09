import tensorflow as tf

from model import Model

IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, lr, num_classes, max_batch_size=None, seed=None):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(lr, seed, max_batch_size)

    def create_model(self):
        """Model function for linear model."""
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name="features")
        labels = tf.placeholder(tf.int64, shape=[None], name="labels")
        logits = tf.layers.dense(inputs=features, units=self.num_classes)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax-tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, loss, train_op, eval_metric_ops
