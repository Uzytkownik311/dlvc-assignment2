from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
from dlvc.batches import BatchGenerator
import dlvc.ops as ops
from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy
import torch
import os
import numpy as np
from cnn_cats_dogs import CNN

'''
Finetuning torchvision models
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
'''
# from __future__ import print_function
# from __future__ import division
import torch.nn as nn
from torchvision import models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name: str, num_classes: int, feature_extract: bool, use_pretrained: bool=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if model_name == "cnn":
        model_ft = CNN()
        set_parameter_requires_grad(model_ft, False)
        num_ftrs = model_ft._fc1.in_features
        model_ft._fc1 = nn.Linear(num_ftrs, num_classes)
        input_size = 32

    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def data_transformation(input_dim: int, d_a: bool, n_c: bool, n_f: bool):
    if d_a:
        if not n_c:
            op = ops.chain([ops.rcrop(32, 6, 'reflect'), ops.hwc2chw(), ops.add(-127.5), ops.mul(1 / 127.5), ops.resize(input_dim)])
        if not n_f:
            op = ops.chain([ops.hflip(), ops.hwc2chw(), ops.add(-127.5), ops.mul(1 / 127.5), ops.resize(input_dim)])
    else:
        op = ops.chain([ops.hwc2chw(), ops.add(-127.5), ops.mul(1 / 127.5), ops.resize(input_dim)])

    return op


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Check performance using different models.')
    # The store_true option automatically creates a default value of False
    parser.add_argument('--fpath', default=os.path.join(os.getcwd(), "../datasets/cifar-10-batches-py"),
                        help='Path to data set.')
    parser.add_argument('model', type=str, default='cnn', help='Type of model.')
    parser.add_argument('--batch_size', type=int, default=128, help='Size of batch.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--weight_decay_par', type=float, default=0.000, help='Weight decay parameter.')
    parser.add_argument('--epoch_number', type=int, default=100, help='Epoch number.')
    parser.add_argument('--data_augmentation', action='store_true', help='Apply data augmentation.')
    parser.add_argument('--best_model', action='store_true', help='It saves best model.')
    parser.add_argument('--no_flip', action='store_true', help='No flipping for data augmentation.')
    parser.add_argument('--no_crop', action='store_true', help='No cropping for data augmentation.')

    args = parser.parse_args()

    path = args.fpath
    net = args.model
    num_samples_per_batch = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay_par
    data_augmentation = args.data_augmentation
    best_model = args.best_model
    no_flip = args.no_flip
    no_crop = args.no_crop

    epochs = args.epoch_number

    trainingSet = PetsDataset(path, Subset.TRAINING)
    validationSet = PetsDataset(path, Subset.VALIDATION)

    num_batches = int(np.ceil(len(trainingSet)/num_samples_per_batch))

    ac = Accuracy()
    model_type, input_size = initialize_model(net, num_classes=2, feature_extract=True, use_pretrained=True)
    op = data_transformation(input_size, data_augmentation, no_crop, no_flip)

    #weight decay still has to be implemented
    if input_size == 299:
        clf = CnnClassifier(model_type, (num_samples_per_batch, 3, input_size, input_size), 2, lr=learning_rate, wd=weight_decay, is_inception=True)
    else:
        clf = CnnClassifier(model_type, (num_samples_per_batch, 3, input_size, input_size), 2, lr=learning_rate, wd=weight_decay)

    stored_train_losses = []
    stored_validation_accuracy = []

    v_batch_gen = BatchGenerator(validationSet, num_samples_per_batch, True, op)
    v_iter_gen = iter(v_batch_gen)
    v_batch = next(v_iter_gen)

    best_accuracy = 0.0

    for e in range(0, epochs):
        t_batch_gen = BatchGenerator(trainingSet, num_samples_per_batch, True, op)
        t_iter_gen = iter(t_batch_gen)

        losses = []

        for b in range(1, num_batches+1):
            t_batch = next(t_iter_gen)
            current_loss = clf.train(t_batch.data, t_batch.label)
            losses.append(np.float(current_loss))

        losses_np = np.asarray(losses)
        mean_loss = np.mean(losses_np)
        var_loss = np.var(losses_np)
        predictions = clf.predict(v_batch.data)
        predictions = predictions.detach().numpy()
        ac.update(predictions, v_batch.label)
        v_accuracy = ac.accuracy()

        if best_model:
            if v_accuracy > best_accuracy:
                torch.save(net.state_dict(), os.path.join(os.getcwd(), net+'_best_model.pth'))
                best_accuracy = v_accuracy

        stored_train_losses.append(str(mean_loss))
        stored_validation_accuracy.append(str(ac))

        print("epoch" + str(e))
        print("train loss: " + str(mean_loss) + " +- " + str(var_loss))
        print("val acc: " + str(ac))

    if best_model:
        print('\n' + "Best model has validation accuracy equal: " + str(best_accuracy) + ".")

    with open('train_losses_log', 'w') as file_loss:
        for i in stored_train_losses:
            file_loss.write(i + '\n')

    with open('validation_accuracy_log', 'w') as file_accuracy:
        for i in stored_validation_accuracy:
            file_accuracy.write(i + '\n')
