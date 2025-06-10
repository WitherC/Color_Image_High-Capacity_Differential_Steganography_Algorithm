
from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import datetime

import numpy as np
import tensorlayer as tl
from model_channeldif import U_Net_S,Extractor,XuNet,SRNet,Lion
import imageio
import csv
import time
from tensorflow.python.client import device_lib
import tensorflow as tf

gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)
print(tf.test.is_gpu_available())
local_device_protos = device_lib.list_local_devices()

def g_loss_fn(u_net_s,extractor,Xunet,SRnet, batch_x, batch_y, is_training, batch_size = 16):
    batch_x_r, batch_x_g, batch_x_b = tf.split(batch_x, num_or_size_splits=3, axis=-1)
    batch_x_r = tf.cast(batch_x_r, dtype=tf.float32)
    batch_x_g = tf.cast(batch_x_g, dtype=tf.float32)
    diff_image = tf.math.subtract(batch_x_r, batch_x_g)
    diff_image = tf.concat([diff_image, batch_y], -1)
    stego_ori = u_net_s(diff_image, is_training)  # 此语句为将秘密图片嵌入到差值中
    batch_x = tf.constant(batch_x)
    batch_x = tf.cast(batch_x, dtype=tf.float32)
    stego_mid = stego_ori + batch_x
    stego = tf.clip_by_value(stego_mid, 0, 255)
    # print(stego.dtype)--->float
    extra1 = extractor(stego, is_training)
    extra = tf.clip_by_value(extra1, 0, 255)
    # print(extra.dtype)--->float

    Gloss_MSE_stego = 0.1 * tl.cost.mean_squared_error(batch_x, stego, is_mean=True)
    Gloss_MSE_extra = 0.1 * tl.cost.mean_squared_error(batch_y, extra, is_mean=True)

    a = tf.squeeze(batch_x)
    b = tf.squeeze(stego)
    c = tf.squeeze(batch_y)
    d = tf.squeeze(extra)
    Gloss_ssim_stego = 100 * (1 - tf.reduce_mean(tf.image.ssim(tf.cast(b, dtype='uint8'), tf.cast(a, dtype='uint8'), max_val=255.0)))
    Gloss_ssim_extra = 100 * (1 - tf.reduce_mean(tf.image.ssim(tf.cast(d, dtype='uint8'), tf.cast(c, dtype='uint8'), max_val=255.0)))


    SR_array = np.zeros([batch_size, 2], dtype=np.float32)
    for i in range(0, batch_size):
        SR_array[i, 0] = 1
    Img_label_SR = tf.constant(SR_array)
    d_logits_SR = SRnet(stego / 255.0, training=False)
    correct_prediction_SR = tf.equal(tf.argmax(d_logits_SR, 1), tf.argmax(Img_label_SR, 1))
    loss_SR = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_SR, labels=Img_label_SR))
    #loss_SR = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_SR, labels=Img_label_SR)

    XU_array = np.zeros([batch_size, 2], dtype=np.float32)
    for i in range(0, batch_size):
        XU_array[i, 0] = 1
    Img_label_XU = tf.constant(XU_array)
    d_logits_XU = Xunet(stego / 255.0, training=False)
    correct_prediction_XU = tf.equal(tf.argmax(d_logits_XU, 1), tf.argmax(Img_label_XU, 1))
    loss_XU = 10*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_XU, labels=Img_label_XU))

    Eloss = Gloss_MSE_extra + Gloss_ssim_extra
    Sloss = Gloss_MSE_stego + Gloss_ssim_stego

    Antiloss = loss_SR + loss_XU
    Gloss_All = 0.05*Antiloss + (Gloss_MSE_extra+Gloss_MSE_stego) + (Gloss_ssim_extra+Gloss_ssim_stego)

    return Gloss_All,Sloss,Eloss,Antiloss,Gloss_MSE_stego,Gloss_MSE_extra,Gloss_ssim_stego,Gloss_ssim_extra,stego,extra,loss_SR,loss_XU

def SR_loss_fn(u_net_s, SRnet, batch_x, batch_y, is_training, batch_size=16):
    batch_x_r, batch_x_g, batch_x_b = tf.split(batch_x, num_or_size_splits=3, axis=-1)
    batch_x_r = tf.cast(batch_x_r, dtype=tf.float32)
    batch_x_g = tf.cast(batch_x_g, dtype=tf.float32)

    diff_image = tf.math.subtract(batch_x_r, batch_x_g)
    diff_image = tf.concat([diff_image, batch_y], -1)

    stego_ori = u_net_s(diff_image)
    batch_x = tf.constant(batch_x)
    batch_x = tf.cast(batch_x, dtype=tf.float32)
    stego_mid = stego_ori + batch_x
    stego = tf.clip_by_value(stego_mid, 0, 255)

    batch_SR2 = tf.concat([batch_x, stego], 0)
    SR_array2 = np.zeros([batch_size * 2, 2], dtype=np.float32)
    for i in range(0, batch_size):
        SR_array2[i, 0] = 1
    for i in range(batch_size, batch_size * 2):
        SR_array2[i, 1] = 1
    Img_label_SR2 = tf.constant(SR_array2)
    d_logits_SR2 = SRnet(batch_SR2 / 255.0, is_training)
    correct_prediction_SR2 = tf.equal(tf.argmax(d_logits_SR2, 1), tf.argmax(Img_label_SR2, 1))
    accuracyD_SR2 = tf.reduce_mean(tf.cast(correct_prediction_SR2, tf.float32))

    loss_SR2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_SR2, labels=Img_label_SR2))

    loss_all_SR = loss_SR2
    return loss_all_SR,loss_SR2, accuracyD_SR2

def Xu_loss_fn(u_net_s, Xunet, batch_x, batch_y, is_training, batch_size=16):
    batch_x_r, batch_x_g, batch_x_b = tf.split(batch_x, num_or_size_splits=3, axis=-1)
    batch_x_r = tf.cast(batch_x_r, dtype=tf.float32)
    batch_x_g = tf.cast(batch_x_g, dtype=tf.float32)

    diff_image = tf.math.subtract(batch_x_r, batch_x_g)
    diff_image = tf.concat([diff_image, batch_y], -1)

    stego_ori = u_net_s(diff_image)
    batch_x = tf.constant(batch_x)
    batch_x = tf.cast(batch_x, dtype=tf.float32)
    stego_mid = stego_ori + batch_x
    stego = tf.clip_by_value(stego_mid, 0, 255)

    batch_XU2 = tf.concat([batch_x, stego], 0)
    XU_array2 = np.zeros([batch_size * 2, 2], dtype=np.float32)
    for i in range(0, batch_size):
        XU_array2[i, 0] = 1
    for i in range(batch_size, batch_size * 2):
        XU_array2[i, 1] = 1
    Img_label_XU2 = tf.constant(XU_array2)
    d_logits_XU2 = Xunet(batch_XU2 / 255.0, is_training)
    correct_prediction_XU2 = tf.equal(tf.argmax(d_logits_XU2, 1), tf.argmax(Img_label_XU2, 1))
    accuracyD_XU2 = tf.reduce_mean(tf.cast(correct_prediction_XU2, tf.float32))

    loss_XU2 = 10*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_XU2, labels=Img_label_XU2))

    loss_all_XU = loss_XU2
    return loss_all_XU,loss_XU2, accuracyD_XU2


def main():
    global temp_image_index
    tf.random.set_seed(22)
    np.random.seed(22)
    epochs = 300000
    batch_size = 16
    time_start = time.time()
    is_training = True
    NUM_IMG = 35000
    gpu_device_name = tf.test.gpu_device_name()
    print(gpu_device_name)
    print(tf.test.is_gpu_available())
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    path1 = r"D:\WHC\img_img_TRAIN_PNG for channeldif\train_A"
    path2 = r"D:\WHC\img_img_TRAIN_PNG for channeldif\train_B"

    Xunet = XuNet()
    #Xunet.build(input_shape=(24, 256, 256, 3))
    SRnet = SRNet()
    #SRnet.build(input_shape=(24, 256, 256, 3))

    u_net_s = U_Net_S()
    u_net_s.build(input_shape=(batch_size,256,256,4))
    extractor = Extractor()
    extractor.build(input_shape=(batch_size, 256, 256, 3))
    # u_net_s.load_weights('./train_anti_weight/' + 'encode220000.ckpt').expect_partial()
    # extractor.load_weights('./train_anti_weight/' + 'decode220000.ckpt').expect_partial()
    image_index = range(1, NUM_IMG + 1)
    seed = 0

    # u_net_s.load_weights(r"D:\WHC\img_img_test\u_net_s51500.ckpt").expect_partial()
    # extractor.load_weights(r"D:\WHC\img_img_test\extractor51500.ckpt").expect_partial()

    # Xunet.load_weights(r"D:\WHC\img_img_test\some_weight\XuNet315500.ckpt").expect_partial()
    # SRnet.load_weights(r"D:\WHC\img_img_test\SR-net299500.ckpt").expect_partial()

    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir_Gloss = r"D:\WHC\img_img_test\2023.12.18\lr\0.001\log" + current_time
    # log_dir_MSE_stego = r"D:\WHC\img_img_test\2023.12.18\lr\0.001\log" + current_time
    # log_dir_MSE_extra = r"D:\WHC\img_img_test\2023.12.18\lr\0.001\log" + current_time
    # log_dir_Loss_XU = r"D:\WHC\img_img_test\2023.12.18\lr\0.001\log" + current_time
    # log_dir_Loss_SR = r"D:\WHC\img_img_test\2023.12.18\lr\0.001\log" + current_time

    # summary_writer_Gloss = tf.summary.create_file_writer(log_dir_Gloss)  # 创建监控类，监控数据写入到log_dir目录
    # summary_writer_MSE_stego = tf.summary.create_file_writer(log_dir_MSE_stego)
    # summary_writer_MSE_extra = tf.summary.create_file_writer(log_dir_MSE_extra)
    # summary_writer_Loss_XU = tf.summary.create_file_writer(log_dir_Loss_XU)
    # summary_writer_Loss_SR = tf.summary.create_file_writer(log_dir_Loss_SR)

    # csv_Xu = 'Xu-loss_Lion_history2.csv'
    # with open(csv_Xu, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     # 写入CSV文件的表头
    #     writer.writerow(['iteration','Epoch', 'XU-Loss','time'])
    #
    # csv_SR = 'SR-loss_Lion_history2.csv'
    # with open(csv_SR, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     # 写入CSV文件的表头
    #     writer.writerow(['iteration','Epoch', 'SR-Loss', 'time'])
    csv_loss = 'loss_all_Adam_history.csv'
    with open(csv_loss, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入CSV文件的表头
        writer.writerow(['iteration', 'Epoch', 'csv_loss_all', 'time'])

    count = 0
    for epoch in range(epochs):
        batch_x = np.zeros([batch_size, 256, 256, 3])
        batch_y = np.zeros([batch_size, 256, 256, 3])
        for j in range(batch_size):
            count = count % NUM_IMG
            if (count == 0):
                print('----------- Epoch %d------------' % seed)
                np.random.seed(seed)
                seed = seed + 1
                temp_image_index = np.random.permutation(image_index)  # shuffle the training set every epoch
            imc_x = imageio.v2.imread(path1 + '/' + '%d' % (temp_image_index[count]) + '.png')
            imc_y = imageio.v2.imread(path2 + '/' + '%d' % (temp_image_index[count]) + '.png')
            batch_x[j, :, :, :] = imc_x
            batch_y[j, :, :, :] = imc_y
            count = count + 1

        log_dir = os.path.join('logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
        count_jian = seed / 3

#-----------Adam优化时
        XU_optimizer = tf.optimizers.Adam(learning_rate=0.00001,beta_1=0.5)
        SR_optimizer = tf.optimizers.Adam(learning_rate=0.00001,beta_1=0.5)
        e_optimizer = tf.optimizers.Adam(learning_rate=0.00001,beta_1=0.5)
        g_optimizer = tf.optimizers.Adam(learning_rate=0.00001,beta_1=0.5)
#--------------
#------------Lion优化时
        # XU_optimizer = Lion(learning_rate=0.00001,wd=0.00035)
        # SR_optimizer = Lion(learning_rate=0.00001,wd=0.00035)
        # e_optimizer = Lion(learning_rate=0.00001,wd=0.003)
        # g_optimizer = Lion(learning_rate=0.00001,wd=0.003)
        # e_optimizer = Lion(learning_rate=0.000001, wd=0.003)
        # g_optimizer = Lion(learning_rate=0.000001, wd=0.003)
#---------------------
        with tf.GradientTape() as tape:
            loss_all_SR,loss_SR2, accuracyD_SR2 = SR_loss_fn(u_net_s, SRnet, batch_x, batch_y, is_training)
        grads = tape.gradient(loss_all_SR, SRnet.trainable_variables)
        SR_optimizer.apply_gradients(zip(grads, SRnet.trainable_variables))

        with tf.GradientTape() as tape:
            loss_all_XU,loss_XU2, accuracyD_XU2 = Xu_loss_fn(u_net_s, Xunet, batch_x, batch_y, is_training)
        grads = tape.gradient(loss_all_XU, Xunet.trainable_variables)
        XU_optimizer.apply_gradients(zip(grads, Xunet.trainable_variables))

        with tf.GradientTape() as tape:
            Gloss_All,Sloss,Eloss,Antiloss,Gloss_MSE_stego,Gloss_MSE_extra,Gloss_ssim_stego,Gloss_ssim_extra,stego,extra,loss_SR,loss_XU = g_loss_fn(u_net_s,extractor,Xunet,SRnet, batch_x, batch_y, is_training)
        grads = tape.gradient(Gloss_All, u_net_s.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, u_net_s.trainable_variables))

        with tf.GradientTape() as tape:
            Gloss_All,Sloss,Eloss,Antiloss,Gloss_MSE_stego,Gloss_MSE_extra,Gloss_ssim_stego,Gloss_ssim_extra,stego,extra,loss_SR,loss_XU = g_loss_fn(u_net_s,extractor,Xunet,SRnet, batch_x, batch_y, is_training)
        grads = tape.gradient(Eloss, extractor.trainable_variables)
        e_optimizer.apply_gradients(zip(grads, extractor.trainable_variables))

        time_end = time.time()
#----------------
        # with open(csv_Xu, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([int(epoch), int(epoch/2187.5), float(loss_XU), float(time_end)-float(time_start)])
        # with open(csv_SR, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([int(epoch), int(epoch/2187.5), float(loss_SR), float(time_end)-float(time_start)])
        with open(csv_loss, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([int(epoch), int(epoch/2187.5), float(Gloss_All), float(time_end)-float(time_start)])
#------------------

        print('---------------------------------------------------------------------------------------------------------','\n',
              'epoch：',int(epoch/2187.5),'训练次数：',epoch,'\n',
              'Gloss', float(Sloss),'Eloss', float(Eloss),
              'Gloss_MSE_stego',float(Gloss_MSE_stego),'Gloss_MSE_extra',float(Gloss_MSE_extra),'\n',
              'gloss_SR', float(loss_SR),'\n',
              'gloss_XU', float(loss_XU),'\n',
              'Time:',float(time_end)-float(time_start),'\n',
              '---------------------------------------------------------------------------------------------------------')

        # if epoch % 10 == 0:
        #     imageio.imwrite(r"D:\WHC\img_img_test\2023.12.18\lr\0.001\stego" + '/' + str(epoch) + '.png', np.uint8(stego[0, :, :, :]))
        #     imageio.imwrite(r"D:\WHC\img_img_test\2023.12.18\lr\0.001\stego" + '/' + str(epoch) + 'ori' + '.png',np.uint8(batch_x[0, :, :, :]))
        #     imageio.imwrite(r"D:\WHC\img_img_test\2023.12.18\lr\0.001\extra" + '/' + str(epoch) + '.png', np.uint8(extra[0, :, :, :]))
        #     imageio.imwrite(r"D:\WHC\img_img_test\2023.12.18\lr\0.001\extra" + '/' + str(epoch) + 'ori' + '.png',np.uint8(batch_y[0, :, :, :]))

        # with summary_writer_Gloss.as_default():
        #     tf.summary.scalar('Gloss_All', Gloss_All, step=epoch)
        # with summary_writer_MSE_stego.as_default():
        #     tf.summary.scalar('Sloss', Sloss, step=epoch)
        # with summary_writer_MSE_extra.as_default():
        #     tf.summary.scalar('Eloss', Eloss, step=epoch)
        # with summary_writer_Loss_SR.as_default():
        #     tf.summary.scalar('loss_SR', loss_SR, step=epoch)
        # with summary_writer_Loss_XU.as_default():
        #     tf.summary.scalar('loss_XU', loss_XU, step=epoch)


        # if epoch % 500 == 0:
        #     u_net_s.save_weights(r"D:\WHC\img_img_test\2023.12.18\lr\0.001\weight" + '/' + 'u_net_s' + '%d' % (epoch) + '.ckpt')
        #     extractor.save_weights(r"D:\WHC\img_img_test\2023.12.18\lr\0.001\weight" + '/' + 'extractor' + '%d' % (epoch) + '.ckpt')

if __name__ == '__main__':
    main()