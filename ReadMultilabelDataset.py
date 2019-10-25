"""
File: ReadMultilabelDataset.py
"""
import json
import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image

IMAGEDIR = 'F:/数据集/螺栓多标记数据集初建/multilabel-image/'
TFRECORDDIR = 'F:/数据集/螺栓多标记数据集初建/multilabel-tfrecords/'
CLASSNAME = ['pin', 'nut', 'slim']
slim = tf.contrib.slim

def get_multilabel_dataset_dict(imagedir, class_name, train_percentage=8):
    rootdir = imagedir
    category = [x[1] for x in os.walk(imagedir)][0]
    dataset = {}
    for j, cat in enumerate(category):
        sub_label = get_label(class_name, cat)
        subdir = os.path.join(rootdir, cat)
        imagelist = os.listdir(subdir)
        number = len(imagelist)
        train_dataset = []
        test_dataset = []
        print('{}: {}'.format(cat, sub_label))
        for i, image in enumerate(imagelist):
            r = random.randint(0, number)
            if r < number / 10.0 *train_percentage:
                train_dataset.append(image)
            else:
                test_dataset.append(image)
        dataset[cat] = {
            'dir':subdir,
            'label':sub_label, 
            'train':train_dataset,
            'test':test_dataset
        }
    return dataset

def get_label(class_name, cat):   
    label = []
    cls = cat.split('+')
    for i, x in enumerate(class_name):
        if x in cls:
            label.append(1)
        else:
            label.append(0)
    return label 

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord_example(label, imagefile, resize=None):
    pil_image = Image.open(imagefile)
    if resize != None:
        pil_image = pil_image.resize(resize)
    bytes_image = pil_image.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': int64_list_feature(label), 
        'image': bytes_feature(bytes_image)
        #'format': bytes_feature('jpg')
    }))    
    return example

def create_multilabel_tfrecord(dataset, tfrecord_dir, dataset_type='train',resize=None):
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_dir, 
                dataset_type + '.tfrecords'))
    for classname, info in dataset.items():
        for imagefile in info[dataset_type]:
            example = create_tfrecord_example(info['label'], 
                      os.path.join(info['dir'], imagefile), resize)
            writer.write(example.SerializeToString())
    writer.close()

def read_multilabel_tfrecord(tfrecord_path, resize, batch_size=1):
    print('tfrecord:{}'.format(tfrecord_path))
    filename_queue = tf.train.string_input_producer([tfrecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,
                                   features={
                                   'label': tf.FixedLenFeature([3], tf.int64), 
                                   'image': tf.FixedLenFeature([], tf.string),
                                   })
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, resize)
    label = tf.cast(features['label'], tf.int64)
    img_batch, label_batch = tf.train.shuffle_batch([image, label],
                    batch_size=batch_size,
                    capacity=5000,
                    min_after_dequeue=1000)

    return img_batch, label_batch

'''dataset = get_multilabel_dataset_dict(IMAGEDIR, CLASSNAME)
#print(dataset)
create_multilabel_tfrecord(dataset, TFRECORDDIR, 'train', (256, 256))
create_multilabel_tfrecord(dataset, TFRECORDDIR, 'test', (256, 256))'''
'''dataset, label = get_dataset_dict(IMAGEDIR, 8)
create_tfrecord(dataset, label, TFRECORDDIR, 'train', (256, 256))
create_tfrecord(dataset, label, TFRECORDDIR, 'test', (256, 256))'''
'''#dataset, label = get_dataset_dict(IMAGEDIR, 8)
#create_tfrecord(dataset, label, TFRECORDDIR, 'train', (256, 256))
#create_tfrecord(dataset, label, TFRECORDDIR, 'test', (256, 256))
image, label = read_tfrecord(TFRECORDDIR+'train.tfrecords', [256, 256, 3], 200)
image2, label2 = read_tfrecord(TFRECORDDIR+'train.tfrecords', [256, 256, 3], 200)
#print(image)
#config=tf.ConfigProto(log_device_placement=True)
image2, label2 = read_multilabel_tfrecord(TFRECORDDIR+'train.tfrecords', [256, 256, 3], 5)
#print(image)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    img, l = sess.run([image2, label2])
    print(l)
    print(np.array(img).shape)
    imgs = Image.fromarray(np.array(img)[0,:,:,:])
    imgs.show()'''
