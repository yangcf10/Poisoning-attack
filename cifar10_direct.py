# directly calculate gradient of normal data loss wrt the poisoned input
# cifar 10, direct method

import mxnet as mx
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time
import copy

# load dataset
dev = mx.gpu(0)
batch_size = 1
# data_shape = (batch_size, 3072)
data_shape = (batch_size, 3, 28, 28)
train_iter = mx.io.ImageRecordIter(
    path_imgrec="data/cifar10/cifar10_train.rec",
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    # mean_img="data/cifar10/mean.bin",
    rand_crop=True,
    rand_mirror=True,
    round_batch=True)

val_iter = mx.io.ImageRecordIter(
    path_imgrec="data/cifar10/cifar10_val.rec",
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    # mean_img="data/cifar10/mean.bin",
    rand_crop=True,
    rand_mirror=True,
    round_batch=True)

def CalLogLoss(pred_prob, label):
    loss = 0.
    for i in range(pred_prob.shape[0]):
        loss += -np.log(max(pred_prob[i, int(label[i])], 1e-10))
    return loss

with mx.Context(dev):
    # load original model
    softmax, arg_params, aux_params = mx.model.load_checkpoint('model/cifar10_model', 300)
    model = mx.mod.Module(softmax, context=dev)
    model.bind(data_shapes=train_iter.provide_data,
             label_shapes=train_iter.provide_label)
    model.set_params(arg_params, aux_params)
    # -------- parameters ----------------
    train_iter.reset()
    dataBatchP = copy.deepcopy(train_iter.next())
    dataBatchP.label[0] = dataBatchP.label[0]+1
    num_normal = 10
    total_round = 7
    attacked_model_lr = 0.01
    model.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=(('learning_rate', attacked_model_lr),))
    # -----------get normal data loss and accuracy----------
    loss = 0
    for num in range(num_normal):
        dataBatch = copy.deepcopy(train_iter.next())
        label = dataBatch.label[0]
        model.forward(dataBatch)
        output = model.get_outputs()[0].asnumpy()
        loss += CalLogLoss(output, label.asnumpy())
    print 'normal data loss: %.4f' % loss

    val_iter.reset()
    val_acc = model.score(val_iter,'acc')
    print 'Val Acc: %.4f' % val_acc[0][1]

    # -----------get loss and accuracy with initial poisoned data----------
    # load pre-trained weight
    model.forward(dataBatchP,is_train=True)
    model.backward()
    model.update()
    val_iter.reset()
    val_acc = model.score(val_iter,'acc')
    print 'Val Acc: %.4f' % val_acc[0][1]
    # re-evaluate normal data loss
    loss = 0
    train_iter.reset()
    dataBatch = copy.deepcopy(train_iter.next())
    for num in range(num_normal):
        dataBatch = copy.deepcopy(train_iter.next())
        label = dataBatch.label[0]
        model.forward(dataBatch)
        loss += CalLogLoss(model.get_outputs()[0].asnumpy(), label.asnumpy())
    print 'normal data loss: %.4f' % loss
    # ---------generate poisoned data------------
    plt.figure('poisoned data')
    # initial poisoned data
    plt.subplot(1, 5, 1)
    plt.imshow((dataBatchP.data[0]).asnumpy()[0].astype(np.uint8).transpose(1, 2, 0))
    # print dataBatchP.data[0].asnumpy()[0]

    pre_loss = loss
    for round in range(total_round):
        start = time.time()
        print 'round %d' % round
        # calculate gradient wrt poisoned data
        dir = np.zeros(data_shape).reshape(1,2352)
        label_tmp = copy.deepcopy(dataBatchP.label)
        for gradient_round in range(data_shape[-1]*data_shape[-2]*data_shape[-3]):
            data_tmp = copy.deepcopy(dataBatchP.data[0])
            data_tmp = data_tmp.asnumpy().reshape(1,2352)
            data_tmp[0][gradient_round] += 1
            # load pre-trained weight
            model.set_params(arg_params, aux_params)

            dataBatch_tmp = copy.deepcopy(mx.io.DataBatch([mx.nd.array(data_tmp.reshape(1,3,28,28))],label_tmp))
            model.forward(dataBatch_tmp,is_train=True)
            model.backward()
            # update attacked model
            model.update()
            # calculate normal data loss
            loss = 0
            train_iter.reset()
            dataBatch = copy.deepcopy(train_iter.next())
            for num in range(num_normal):
                dataBatch = copy.deepcopy(train_iter.next())
                label = dataBatch.label[0]
                model.forward(dataBatch)
                output = model.get_outputs()[0].asnumpy()
                loss += CalLogLoss(output, label.asnumpy())
            dir[0][gradient_round] = np.sign(loss-pre_loss)
        tmp = (dataBatchP.data[0]).asnumpy().reshape(1,2352)+dir*10
        tmp[tmp > 255] = 255
        tmp[tmp < 0] = 0
        # print dataBatchP.data[0].asnumpy()[0]
        dataBatchP.data[0] = mx.nd.array(tmp.reshape(1,3,28,28))
        end = time.time()
        print 'time: %.4f' % (end-start)
        if round%4 == 0:
            plt.subplot(1, 5, round/2+2)
            plt.imshow(dataBatchP.data[0].asnumpy()[0].astype(np.uint8).transpose(1, 2, 0))

        # print dataBatchP.data[0].asnumpy()[0]
        # make one attack
        # load pre-trained weight
        model.set_params(arg_params, aux_params)

        model.forward(dataBatchP,is_train=True)
        model.backward()
        # update attacked model
        model.update()

        val_iter.reset()
        val_acc = model.score(val_iter,'acc')
        print 'Val Acc: %.4f' % val_acc[0][1]

        # re-evaluate normal data loss
        loss = 0
        dataBatch = copy.deepcopy(train_iter.next())
        for num in range(num_normal):
            dataBatch = copy.deepcopy(train_iter.next())
            label = dataBatch.label[0]
            model.forward(dataBatch)
            output = model.get_outputs()[0].asnumpy()
            loss += CalLogLoss(output, label.asnumpy())
        print 'normal data loss: %.4f' % loss

        pre_loss = loss
