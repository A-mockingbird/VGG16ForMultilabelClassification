import tensorflow as tf
from PIL import Image
import numpy as np
import vgg_multilabel as VGG
import ReadMultilabelDataset as rmld

def test(model_path, tfrecord_path):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='output')
    keep_prob = tf.placeholder(tf.float32)
    predict = VGG.VGG16(x, keep_prob, 3)
    batch_images_test, batch_labels_test = rmld.read_multilabel_tfrecord(tfrecord_path+'test.tfrecords', 
                                                [224, 224, 3], 50)
    results =  1 - tf.to_float(tf.less(tf.nn.sigmoid(predict), 0.5))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(results, y), tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    images, labels = sess.run([batch_images_test, batch_labels_test])
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    accur = sess.run([accuracy], feed_dict={x:images, y:labels})
    print('accuary : {}'.format(accur))
    sess.close()

if __name__ == "__main__":
    test('./model.ckpt-9999', 'F:/数据集/螺栓多标记数据集初建/multilabel-tfrecords-224/')