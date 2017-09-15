import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim

#yolo_net.py定义了YOLONet类，该类包含了网络初始化（__init__()），建立网络（build_networks()）和loss函数（loss_layer（））等方法。
class YOLONet(object):

    #网络的所有初始化参数包含于__init__()方法之中。
    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES  #类别
        self.num_class = len(self.classes)  #类别数量，值为20
        self.image_size = cfg.IMAGE_SIZE   #图像尺寸,值为448
        self.cell_size = cfg.CELL_SIZE  #cell尺寸，值为7
        self.boxes_per_cell = cfg.BOXES_PER_CELL    #每个grid cell负责的boxes，默认为2
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)   #输出尺寸
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class   #7×7×20
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell #7×7×20+7×7×2

        self.object_scale = cfg.OBJECT_SCALE    #值为1
        self.noobject_scale = cfg.NOOBJECT_SCALE    #值为1
        self.class_scale = cfg.CLASS_SCALE  #值为2.0
        self.coord_scale = cfg.COORD_SCALE  #值为5.0

        self.learning_rate = cfg.LEARNING_RATE  #学习速率LEARNING_RATE = 0.0001
        self.batch_size = cfg.BATCH_SIZE    #BATCH_SIZE = 45
        self.alpha = cfg.ALPHA  #ALPHA = 0.1

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))  #偏置

        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = self.build_network(self.images, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    #网络建立是通过build_networks()方法实现的，网络由卷积层-pooling层和全连接层组成
    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(net, keep_prob=keep_prob,
                                   is_training=is_training, scope='dropout_35')
                net = slim.fully_connected(net, num_outputs,
                                           activation_fn=None, scope='fc_36')
        return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            # 将网络输出分离为类别和定位以及box大小，输出维度为7*7*20+7*7*2+7*7*2*4=1470
            # 类别，shape为(45, 7, 7, 20)
            predict_classes = tf.reshape(predicts[:, :self.boundary1], [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            # 定位，shape为(45, 7, 7, 2)
            predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            ##box大小，长宽等 shape为(45, 7, 7, 2, 4)
            predict_boxes = tf.reshape(predicts[:, self.boundary2:], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            # label的类别结果，shape为(45, 7, 7, 1)
            response = tf.reshape(labels[:, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, 1])
            # label的定位结果，shape为(45, 7, 7, 1, 4)
            boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            # label的大小结果，shapewei (45, 7, 7, 2, 4)
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            # shape 为(45, 7, 7, 20)
            classes = labels[:, :, :, 5:]

            # offset shape为(7, 7, 2)
            offset = tf.constant(self.offset, dtype=tf.float32)
            # shape为 (1,7, 7, 2)
            offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            # shape为(45, 7, 7, 2)
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            # shape为(4, 45, 7, 7, 2)
            predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                           (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                                           tf.square(predict_boxes[:, :, :, :, 2]),
                                           tf.square(predict_boxes[:, :, :, :, 3])])
            # shape为(45, 7, 7, 2, 4)
            predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

            # shape为(45, 7, 7, 2)
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # shape为 (45, 7, 7, 1)
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            # shape为(45, 7, 7, 2)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response
            # mask = tf.tile(response, [1, 1, 1, self.boxes_per_cell])

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # shape为(45, 7, 7, 2)
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            # shape为(4, 45, 7, 7, 2)
            boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                                   boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                                   tf.sqrt(boxes[:, :, :, :, 2]),
                                   tf.sqrt(boxes[:, :, :, :, 3])])
            boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * self.noobject_scale

            # coord_loss
            # shape 为 (45, 7, 7, 2, 1)
            coord_mask = tf.expand_dims(object_mask, 4)
            # shape为(45, 7, 7, 2, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
    return op
