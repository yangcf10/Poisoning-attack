# mnist, generative method, gradient to output of autoencoder

import mxnet as mx
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time

# load dataset, model structure, parameter
softmax = mx.symbol.load('model/mymodel.json')
arg_arrays_load = mx.nd.load('model/mymodel_args')
dev = mx.gpu(0)
batch_size = 1
data_shape = (batch_size, 784)
input_shape = (784,)
flat = True
train_iter = mx.io.MNISTIter(
        image="data/mnist/train-images-idx3-ubyte",
        label="data/mnist/train-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=False,
        flat=flat)
val_iter = mx.io.MNISTIter(
        image="data/mnist/t10k-images-idx3-ubyte",
        label="data/mnist/t10k-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=flat)

def SGD(key, weight, grad, lr=0.01, grad_norm=batch_size):
    # key is key for weight, we can customize update rule
    # weight is weight array
    # grad is grad array
    # lr is learning rate
    # grad_norm is scalar to norm gradient, usually it is batch_size
    norm = 1.0 / grad_norm
    # here we can bias' learning rate 2 times larger than weight
    if "weight" in key or "gamma" in key:
        weight[:] -= lr * (grad * norm)
    elif "bias" in key or "beta" in key:
        weight[:] -= 2.0 * lr * (grad * norm)
        # weight[:] -= lr * (grad * norm)
    else:
        pass

def Accuracy(label, pred_prob):
    pred = np.argmax(pred_prob, axis=1)
    return np.sum(label == pred) * 1.0 / label.shape[0]

def CalLogLoss(pred_prob, label):
    loss = 0.
    for i in range(pred_prob.shape[0]):
        loss += -np.log(max(pred_prob[i, int(label[i])], 1e-10))
    return loss

with mx.Context(dev):
    # load original model
    arg_shapes, output_shapes, aux_shapes = softmax.infer_shape(data=data_shape)
    grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
    arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
    model = softmax.bind(ctx=dev, args=arg_arrays, args_grad = grad_arrays)

    # load pre-trained weight
    for i in range(1,len(arg_arrays_load)-1):
        arg_arrays_load[i].copyto(model.arg_arrays[i])

    # define autoencoder
    de_out = mx.symbol.load('model/aemodel_2.json')
    ae_arg_arrays_load = mx.nd.load('model/ae_args_2')

    ae_arg_shapes, ae_output_shapes, ae_aux_shapes = de_out.infer_shape(data=data_shape)
    ae_grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in ae_arg_shapes]
    ae_arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in ae_arg_shapes]

    en_out = de_out.get_internals()['en_out_output']
    group = mx.symbol.Group([de_out, en_out])
    ae_model = group.simple_bind(ctx=dev, data=data_shape, grad_req='write')
    # ae_model = group.bind(ctx=dev, args=[ae_arg_arrays], args_grad = [ae_grad_arrays])
    # load pre-trained weight
    for i in range(1,len(ae_arg_arrays_load)):
        ae_arg_arrays_load[i].copyto(ae_model.arg_arrays[i])

    train_iter.reset()
    dataBatch = train_iter.next()
    data = dataBatch.data[0]
    # data_p = mx.nd.zeros(data.shape)
    data_p = mx.random.uniform(0, 1, data.shape)
    data.copyto(data_p)
    label_p = dataBatch.label[0]+1
    total_round = 10
    num_normal = 10
    attacked_model_lr = 0.05
    generative_model_lr = 0.005
    # get normal data loss
    loss = 0
    for num in range(num_normal):
        dataBatch = train_iter.next()
        data = dataBatch.data[0]
        label = dataBatch.label[0]
        model.arg_dict['data'][:] = data
        model.arg_dict['softmax_label'][:] = label
        model.forward(is_train=True)
        output = model.outputs[0].asnumpy()
        loss += CalLogLoss(output, label.asnumpy())
    print 'normal data loss: %.4f' % loss

    val_acc = 0
    nbatch = 0
    val_iter.reset()
    for databatch in val_iter:
        data = databatch.data[0]
        label = databatch.label[0]
        model.arg_dict["data"][:] = data
        model.forward(is_train=False)
        val_acc += Accuracy(label.asnumpy(), model.outputs[0].asnumpy())
        nbatch += 1.
    val_acc /= nbatch
    print 'Val Acc: %.4f' % val_acc

    # generate poisoned data
    plt.figure('poisoned data')
    ae_en_grad = mx.nd.zeros(ae_model.outputs[1].shape, ctx=dev)
    ae_de_grad = mx.nd.zeros(ae_model.outputs[0].shape, ctx=dev)
    pre_loss = 0
    for round in range(total_round):
        start = time.time()
        print 'round %d' % round
        # update original model

        # load pre-trained weight
        for i in range(1, len(arg_arrays_load) - 1):
            arg_arrays_load[i].copyto(model.arg_arrays[i])

        val_acc = 0
        nbatch = 0
        val_iter.reset()
        for databatch in val_iter:
            data = databatch.data[0]
            label = databatch.label[0]
            model.arg_dict["data"][:] = data
            model.forward(is_train=False)
            val_acc += Accuracy(label.asnumpy(), model.outputs[0].asnumpy())
            nbatch += 1.
        val_acc /= nbatch
        print 'Val Acc: %.4f' % val_acc

        train_iter.reset()
        dataBatch = train_iter.next()

        ae_model.arg_dict['data'][:] = data_p
        ae_model.forward(is_train=True)
        ae_output = ae_model.outputs[0].asnumpy()

        model.arg_dict['data'][:] = mx.nd.array(ae_output)
        model.arg_dict['softmax_label'][:] = label_p
        model.forward(is_train=True)
        output = model.outputs[0].asnumpy()[0]
        model.backward()
        for key in model.arg_dict.keys():
                SGD(key, model.arg_dict[key], model.grad_dict[key], attacked_model_lr)
        print 'poisoned network'

        val_acc = 0
        nbatch = 0
        val_iter.reset()
        for databatch in val_iter:
            data = databatch.data[0]
            label = databatch.label[0]
            model.arg_dict["data"][:] = data
            model.forward(is_train=False)
            val_acc += Accuracy(label.asnumpy(), model.outputs[0].asnumpy())
            nbatch += 1.
        val_acc /= nbatch
        print 'Val Acc: %.4f' % val_acc

        if(round%(total_round/5)==0):
            plt.subplot(1, 5, round/(total_round/5)+1)
            plt.imshow(ae_output.reshape(28, 28), cmap=cm.Greys_r)

        # get normal data loss
        loss = 0
        # tmpGrad = np.zeros(ae_model.outputs[1].shape)
        tmpGrad = np.zeros(data.shape)
        for num in range(num_normal):
            dataBatch = train_iter.next()
            data = dataBatch.data[0]
            label = dataBatch.label[0]
            model.arg_dict['data'][:] = data
            model.arg_dict['softmax_label'][:] = label
            model.forward(is_train=True)
            output = model.outputs[0].asnumpy()
            loss += CalLogLoss(output, label.asnumpy())
            model.backward()
            tmpGrad += model.grad_dict['data'].asnumpy()
        ae_de_grad[:] = -np.sign(tmpGrad)*np.sign(loss-pre_loss)
        ae_model.backward([ae_de_grad, ae_en_grad])
        for key in ae_model.arg_dict.keys():
            SGD(key, ae_model.arg_dict[key], ae_model.grad_dict[key], generative_model_lr)
        end = time.time()
        print 'time: %.4f' % (end-start)
        print 'Update autocoder'
        print 'normal data loss: %.4f' % loss
        pre_loss = loss
