# Generative Poisoning Attack Method Against Neural Networks
## Chaofei Yang

Paper is [here](https://arxiv.org/pdf/1703.01340.pdf).

Poisoning attack project, with direct and generative methods on MNIST and Cifar-10.

mnist_direct.py: Direct gradient method on MNIST.
mnist_generative.py: Generative gradient method on MNIST.
cifar_direct.py: Direct gradient method on Cifar-10.
cifar_generative.py: Generative gradient method on Cifar-10.

Training and testing data for MNIST should be saved at:
image="data/mnist/train-images-idx3-ubyte"
label="data/mnist/train-labels-idx1-ubyte"
image="data/mnist/t10k-images-idx3-ubyte"
label="data/mnist/t10k-labels-idx1-ubyte"

Training and testing data for Cifar-10 should be saved at:
path_imgrec="data/cifar10/cifar10_train.rec"
path_imgrec="data/cifar10/cifar10_val.rec"

Pre-trained models are saved at: model/

Learning rates for the target model and generative model (attacked_model_lr, generative_model_lr) are tricky, the accuray degradation may not converge without carefully tuning.
