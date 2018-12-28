from ..model import Model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda


class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss and SGD are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: int, num_classes: int, lr: float, wd: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (sgd with Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''

        if not np.issubdtype(type(input_shape), tuple):
            raise TypeError("The input shape must be of type tuple, it is of type: " + str(type(input_shape)) + ".")

        if not num_classes > 0:
            raise ValueError("The number of classes to classify must be must be at least 1, it is: " + str(num_classes)
                             + ".")

        if not np.issubdtype(type(lr), float):
            raise TypeError("The learning rate must be of type float, it is: " + str(type(lr)) + ".")

        if not np.issubdtype(type(wd), float):
            raise TypeError("The weight decay must be of type float, it is: " + str(type(wd)) + ".")

        self._net = net
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._lr = lr
        self._wd = wd

        self._trained_data = None
        self._trained_labels = None

        # check if you is available
        if cuda.is_available():
            net.cuda()
        else:
            net.cpu()

        # inside the train() and predict() functions you will need to know whether the network itself
        # runs on the cpu or on a gpu, and in the latter case transfer input/output tensors via cuda() and cpu().
        # do termine this, check the type of (one of the) parameters, which can be obtained via parameters() (there is an is_cuda flag).
        # you will want to initialize the optimizer and loss function here. note that pytorch's cross-entropy loss includes normalization so no softmax is required

        # self._module = net
        # self._shape = input_shape
        # self._num_classes = (num_classes,)
        # self._learning_rate = lr
        # self._weight_decay = wd
        # Initialize loss function

        self._loss = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self._net.parameters(), momentum=0.9, weight_decay=self._wd, lr=self._lr, nesterov=True)

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''

        # return self._shape

        return self._input_shape

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return self._num_classes

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the training loss.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        self.check_correctness_of_matrix_data(data)
        self.check_correctness_of_labels(labels)

        if not data.shape[0] == labels.shape[0]:
            raise RuntimeError("The batch do not have equal number of observations and labels, "
                               "observations number: " + str(len(data)) + " labels number" + str(len(labels)) + ".")

        train_data = torch.from_numpy(data).float()
        train_labels = torch.from_numpy(labels).long()

        # Set the net to train() mode

        self._net.train()

        # Set the parameter gradients to zero

        self._optimizer.zero_grad()

        # Forward pass and backward pass, optimize

        output = self._net(train_data)
        loss_train = self._loss(output, train_labels)
        loss_train.backward()
        self._optimizer.step()

        # make sure to set the network to train() mode
        # see above comments on cpu/gpu

        return loss_train

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()).
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # pass the network's predictions through a nn.Softmax layer to obtain softmax class scores
        # make sure to set the network to eval() mode
        # see above comments on cpu/gpu

        self.check_correctness_of_matrix_data(data)
        _d = torch.from_numpy(data).float()

        # Set the net to eval() mode

        self._net.eval()
        prediction = self._net(_d)
        sm = nn.Softmax()
        probabilities = sm(prediction)

        return probabilities

    def check_correctness_of_matrix_data(self, data: np.ndarray):

        if not isinstance(data, np.ndarray):
            raise TypeError("The batch size is not np.ndarray type, but: " + str(type(data)) + ".")

        if not np.issubdtype(data.dtype, float):
            raise TypeError("The data has not value type np.float32, data type is: " + str(data.dtype) + ".")

        if not data.shape[2] == self._input_shape[2]:
            raise RuntimeError("Size of inputs vectors in not equal with value specified in the constructor of"
                               " classifier: " + str(data.shape) + " != " + str(self.input_shape) + ".")

    def check_correctness_of_labels(self, labels: np.ndarray):
        if not isinstance(labels, np.ndarray):
            raise TypeError("The batch size is not np.ndarray type, but: " + str(type(labels)) + ".")

        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("The labels has not value type integer, labels type is: " + str(labels.dtype) + ".")

        if not (labels < self._num_classes).all():
            raise ValueError("The labels contain unknown class.")

