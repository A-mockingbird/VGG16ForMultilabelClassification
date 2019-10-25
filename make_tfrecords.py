import ReadMultilabelDataset as rmld
import ReadRecognitionDataset as rrd
import tensorflow as tf
from PIL import Image
import numpy as np

#IMAGEDIR = 'F:/数据集/螺栓多标记数据集初建/crops/'
#TFRECORDDIR = 'F:/数据集/螺栓多标记数据集初建/tfrecords-224/'
IMAGEDIR = 'F:/数据集/螺栓多标记数据集初建/multilabel-image/'
TFRECORDDIR = 'F:/数据集/螺栓多标记数据集初建/multilabel-tfrecords-224/'
CLASSNAME = ['pin', 'nut', 'slim']

dataset = rmld.get_multilabel_dataset_dict(IMAGEDIR, CLASSNAME)

rmld.create_multilabel_tfrecord(dataset, TFRECORDDIR, 'train', (224, 224))
rmld.create_multilabel_tfrecord(dataset, TFRECORDDIR, 'test', (224, 224))

'''dataset, label = rrd.get_dataset_dict(IMAGEDIR, 7)
rrd.create_tfrecord(dataset, label, TFRECORDDIR, 'train', (224, 224))
rrd.create_tfrecord(dataset, label, TFRECORDDIR, 'test', (224, 224))'''

'''image, label = rrd.read_tfrecord(TFRECORDDIR+'train.tfrecords', [224, 224, 3], 20)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    img, l = sess.run([image, label])
    for i, x in enumerate(img):
        print(l[i])
        imgs = Image.fromarray(np.array(x)[:,:,:])
        imgs.show()
        imgs.close()'''