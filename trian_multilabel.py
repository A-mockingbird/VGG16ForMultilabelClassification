import tensorflow as tf
import ReadRecognitionDataset as rrd
import ReadMultilabelDataset as rmld
import vgg_multilabel as VGG16

def train(n_class, tfrecord_path, batch_size, image_size, max_steps, lr=0.0001):
    x = tf.placeholder(dtype=tf.float32, 
                    shape=[None, image_size[0], image_size[1], 3], name='input')
    y = tf.placeholder(dtype=tf.float32, 
                    shape=[None, n_class], name='label')
    keep_prob = tf.placeholder(tf.float32)
    x_nor = x - [107, 106, 108]
    predict = VGG16.VGG16(x, keep_prob, n_class)
    results = tf.nn.sigmoid(predict)
    
    #smooth
    '''abs_diff = tf.abs(results - y)
    sign = tf.to_float(tf.less(abs_diff, 4))
    y_p = (1-sign) * tf.abs(results- y) - 0.5 * 4
    y_f = 0.25 * 0.5 * sign * tf.pow(abs_diff, 2)
    loss = tf.reduce_mean(tf.reduce_sum(y_f + y_p, 1))'''

    #mse
    #loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(results, y), 1))
    #loss = tf.reduce_sum(tf.squared_difference(results, y), 1)

    #cross entropy
    '''y_p = -y * tf.log(tf.clip_by_value(results, 1e-10, 1.0))
    y_f = -(1-y) * tf.log(tf.clip_by_value(tf.abs(1-results), 1e-10, 1.0))
    loss = 0.5 * tf.reduce_mean(y_p + y_f)'''
    #loss = -tf.reduce_mean(y * tf.log(tf.clip_by_value(results, 1e-10, 1.0)))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=y))

    #train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    #results = tf.nn.softmax(predict)
    results_less =  1 - tf.to_float(tf.less(results, 0.5))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(results_less, y), tf.float32))
    #accuracy = tf.reduce_mean(tf.cast(tf.equal(results, y), tf.float32))

    batch_images, batch_labels = rmld.read_multilabel_tfrecord(tfrecord_path+'train.tfrecords', 
                                                image_size, batch_size)
    batch_images_test, batch_labels_test = rmld.read_multilabel_tfrecord(tfrecord_path+'test.tfrecords', 
                                                image_size, batch_size)
    '''with tf.device('/cpu:0'):
        batch_labels = tf.one_hot(batch_labels, n_class, 1, 0)
        batch_labels_test = tf.one_hot(batch_labels_test, n_class, 1, 0)'''

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    print('###################################')
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)     
        for i in range(max_steps):
            batch_x, batch_y = sess.run([batch_images, batch_labels])
            _, loss_val = sess.run([train_step, loss], 
                         feed_dict={x:batch_x, y:batch_y, keep_prob:0.7})
            # print('loss value: {}'.format(loss_val))
            if i%100 == 0:
                test_x, test_y = sess.run([batch_images_test, batch_labels_test])
                accuracy_val, p, l = sess.run([accuracy, results_less, y], 
                            feed_dict={x:test_x, y:test_y, keep_prob:1.0})
                print('--------------------------')
                print('Step: {}, loss value: {}, accuracy: {}'.format(
                    i, loss_val, accuracy_val))
                print("predict ： {}\nlabel : {}".format(p, l))
        saver.save(sess, './model.ckpt', global_step=i)
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tfrecord_path = 'F:/数据集/螺栓多标记数据集初建/multilabel-tfrecords-224/'
    train(3, tfrecord_path, 20, [224, 224, 3], 10000)