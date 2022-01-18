# Useful imports
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/scripts/')
from simclr import *

import numpy as np
import torch
from torchvision import transforms as T
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

import os
from PIL import Image
from collections import OrderedDict

import random

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
tsne = TSNE()

from tqdm import tqdm

import copy

class MyDataset(Dataset):
    def __init__(self, root_dir, filenames, labels, my_transform, train = False, mutation=False):
        self.root_dir = root_dir
        self.file_names = filenames
        self.labels = labels
        self.mutation = mutation
        self.transform = my_transform
        self.train = train

    def __len__(self):
        return len(self.file_names)

    def tensorify(self, img):
        res = T.ToTensor()(img)
        res = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(res)
        return res

    def mutate_image(self, img):
        res = self.transform(img)
        return res

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        image = T.Resize((250, 250))(image)

        if self.mutation:
            image1 = self.mutate_image(image)
            image1 = self.tensorify(image1)
            image2 = self.mutate_image(image)
            image2 = self.tensorify(image2)
            sample = {'image1': image1, 'image2': image2, 'label': label}
        else:
            if self.train:
                image = self.mutate_image(image)
            image = T.Resize((224, 224))(image)
            image = self.tensorify(image)
            sample = {'image': image, 'label': label}

        return sample

def get_comp_resnet18(pre_trained, fcs, num_classes):


    resnet = resnet18(pretrained=pre_trained)

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(resnet.fc.in_features, fcs[0])),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(fcs[0], num_classes))
    ]))

    resnet.fc = classifier

    return resnet

def train_resnet(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # unpack arguments
    num_epochs, model_path, resnet, data, optim_tag, criterion = args
    dataloader_training_dataset, dataloader_testing_dataset = data

    starting_epoch = 0

    losses_test = []
    losses_train = []
    test_accs = []
    train_accs = []
    best_test_acc = 0.0

    resnet_path = model_path + 'resnet.pt'


    # check for previous training
    if os.path.exists(resnet_path):
      
        model_content = torch.load(resnet_path)
        starting_epoch = model_content['epochs_trained']
        weights = model_content['weights']
        best_test_acc = model_content['max acc test']
        test_accs = model_content['acc test']
        train_accs = model_content['acc train']
        losses_test = model_content['losses test']
        losses_train = model_content['losses train']
        resnet.load_state_dict(weights)
        print(f'Loading {starting_epoch} epochs of training, with current best accuracy of {best_test_acc}')

    if optim_tag == 'SGD':
        # using SGD optimizer
        optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
    # train the model

    resnet.to(device)

    best_model_weights = resnet.state_dict()

    for epoch in range(starting_epoch, num_epochs):
        epoch_losses_train = []
        epoch_acc_train_num = 0.0
        epoch_acc_train_den = 0.0

        resnet.train()

        for (_, sample_batched) in enumerate(tqdm(dataloader_training_dataset, desc = f'Epoch {epoch + 1}/{num_epochs} -Train')):
            
            optimizer.zero_grad()

            # get x and y from the batch
            x = sample_batched['image']
            y_actual = sample_batched['label']

            # move them to the device
            x = x.to(device)
            y_actual  = y_actual.to(device)

            # get output from resnet architecture
            y_predicted = resnet(x)

            # get the cross entropy loss value
            loss = criterion(y_predicted, y_actual)

            # add the obtained loss value to this list
            epoch_losses_train.append(loss.data.item())
            
            # perform backprop through the loss value
            loss.backward()

            # call the optimizer step function
            optimizer.step()

            actual = y_actual.cpu().data

            # get predictions and actual values to cpu  
            pred = np.argmax(y_predicted.cpu().data, axis=1)

            #update the numerators and denominators of accuracy
            epoch_acc_train_num += (actual == pred).sum().item()
            epoch_acc_train_den += len(actual)

        losses_train.append(get_mean_of_list(epoch_losses_train))
        train_acc = epoch_acc_train_num/epoch_acc_train_den
        
        train_accs.append(train_acc)
        print(f'Train accuracy: {train_acc}')

        # run linear classifier in eval mode
        resnet.eval()

        # essential variables to keep track of losses and acc
        epoch_losses_test = []
        epoch_acc_test_num = 0.0
        epoch_acc_test_den = 0.0

        # run a for loop through each batch
        for (_, sample_batched) in enumerate(tqdm(dataloader_testing_dataset, desc = f'Epoch {epoch + 1}/{num_epochs} - Test')):
            x = sample_batched['image']
            y_actual = sample_batched['label']

            x = x.to(device)
            y_actual  = y_actual.to(device)

            y_predicted = resnet(x)

            pred = np.argmax(y_predicted.cpu().data, axis=1)
            
            actual = y_actual.cpu().data

            loss = criterion(y_predicted, y_actual)

            epoch_losses_test.append(loss.data.item())

            epoch_acc_test_num += (actual == pred).sum().item()
            epoch_acc_test_den += len(actual)

        # calculate test_acc
        losses_test.append(get_mean_of_list(epoch_losses_test))
        test_acc = epoch_acc_test_num / epoch_acc_test_den
        test_acc = round(test_acc, 6)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_model_weights = copy.deepcopy(resnet.state_dict())
            best_test_acc = test_acc
            print(f'Test accuracy: {test_acc}, {model_path} updated')

        # either way, save the model
        model_info = {'weights': best_model_weights,
                      'epochs_trained': epoch + 1,
                      'max acc test': best_test_acc,
                      'acc test': test_accs,
                      'acc train': train_accs,
                      'losses train': losses_train,
                      'losses test': losses_test}
        torch.save(model_info, resnet_path)

    return resnet, best_test_acc

def prep_for_experiments(initial_args, val_folder = 'val'):
    root_folder, data_tag, percentage_supervised, ft_aug, pre_trained, test_batch_size, train_batch_size, fcs, num_epochs, optim_tag = initial_args

    # defining a mapping between class names and numbers
    # defining a mapping between class names and numbers
    mapping, classes = get_mapping(root_folder)
    num_classes = len(classes)

    train_names = sorted(os.listdir(root_folder + '/train'))
    test_names = sorted(os.listdir(root_folder + f'/{val_folder}'))

    # setting random seed to ensure the same 10% labelled data is used when training the linear classifier
    random.seed(0)

    names_train = random.sample(train_names, int(len(train_names)*(percentage_supervised/100)))
    names_test = random.sample(test_names, len(test_names))

    # getting labels based on filenames, note that the filenames themselves contain classnames
    # also note that these labels won't be used to actually train the base model
    # these are just for visualization purposes
    labels_train = [mapping[x.split('_')[0]] for x in names_train]
    labels_test = [mapping[x.split('_')[0]] for x in names_test]

    transform = T.Compose([T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8), 
                          T.RandomGrayscale(p=0.2),
                          T.RandomHorizontalFlip(p=0.3),
                          T.RandomVerticalFlip(p=0.3),
                          T.RandomRotation(10),
                          T.RandomResizedCrop(224)
                          ])
    transform_name = 't6'

    # datasets
    training_dataset = MyDataset(root_folder + '/train', names_train, labels_train, transform, train = ft_aug, mutation = False)
    testing_dataset = MyDataset(root_folder + f'/{val_folder}', names_test, labels_test, transform, mutation = False)
    # dataloaders
    dataloader_training_dataset = DataLoader(training_dataset, 
                                            batch_size=train_batch_size, 
                                            shuffle=True, 
                                            num_workers=0)
    dataloader_testing_dataset = DataLoader(testing_dataset, 
                                            batch_size=test_batch_size, 
                                            shuffle=True, 
                                            num_workers=0)

    data = [dataloader_training_dataset, dataloader_testing_dataset]

    # defining our deep learning architecture
    resnet = get_comp_resnet18(pre_trained, fcs, num_classes)
    criterion = nn.CrossEntropyLoss()
    

    model_path = f'/content/drive/MyDrive/Colab Notebooks/SCHOOL/Advanced Computer Vision/Final Project/Model_Results/resnet_18/{data_tag}/comp-resnet_18_optim_{optim_tag}_batch_{train_batch_size}_transform_{transform_name}_epochs_{num_epochs}_sup_{percentage_supervised}/'

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    return num_epochs, model_path, resnet, data, optim_tag, criterion

def prep_final_test(initial_args, test_folder = 'test'):
    
    root_folder, data_tag, percentage_supervised, ft_aug, pre_trained, test_batch_size, train_batch_size, fcs, num_epochs, optim_tag = initial_args

    # defining a mapping between class names and numbers
    mapping, classes = get_mapping(root_folder)
    num_classes = len(classes)

    test_names = sorted(os.listdir(root_folder + f'/{test_folder}'))
    names_test = random.sample(test_names, len(test_names))

    # getting labels based on filenames, note that the filenames themselves contain classnames
    # also note that these labels won't be used to actually train the base model
    # these are just for visualization purposes
    labels_test = [mapping[x.split('_')[0]] for x in names_test]

    transform = T.Compose([T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8), 
                          T.RandomGrayscale(p=0.2),
                          T.RandomHorizontalFlip(p=0.3),
                          T.RandomVerticalFlip(p=0.3),
                          T.RandomRotation(10),
                          T.RandomResizedCrop(224)
                          ])
    transform_name = 't6'

    # datasets
    testing_dataset = MyDataset(root_folder + f'/{test_folder}', names_test, labels_test, transform, mutation = False)
    # dataloaders
    dataloader_testing_dataset = DataLoader(testing_dataset, 
                                            batch_size=test_batch_size, 
                                            shuffle=True, 
                                            num_workers=0)

    # defining our deep learning architecture
    resnet = get_comp_resnet18(pre_trained, fcs, num_classes)
    criterion = nn.CrossEntropyLoss()
    
    model_path = f'/content/drive/MyDrive/Colab Notebooks/SCHOOL/Advanced Computer Vision/Final Project/Model_Results/resnet_18/{data_tag}/comp-resnet_18_optim_{optim_tag}_batch_{train_batch_size}_transform_{transform_name}_epochs_{num_epochs}_sup_{percentage_supervised}/'

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    return num_epochs, model_path, resnet, dataloader_testing_dataset, optim_tag, criterion

def final_test(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # unpack arguments
    num_epochs, model_path, resnet, dataloader_testing_dataset, optim_tag, criterion = args

    resnet_path = model_path + 'resnet.pt'

    # check for previous training
    if os.path.exists(resnet_path):
      
        model_content = torch.load(resnet_path)
        starting_epoch = model_content['epochs_trained']
        weights = model_content['weights']
        best_test_acc = model_content['max acc test']
        test_accs = model_content['acc test']
        train_accs = model_content['acc train']
        losses_test = model_content['losses test']
        losses_train = model_content['losses train']
        resnet.load_state_dict(weights)
        print(f'Loading {starting_epoch} epochs of training, with current best accuracy of {best_test_acc}')

    if optim_tag == 'SGD':
        # using SGD optimizer
        optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
    # train the model

    resnet.to(device)

    best_model_weights = resnet.state_dict()

    # run linear classifier in eval mode
    resnet.eval()

    # essential variables to keep track of losses and acc
    epoch_losses_test = []
    epoch_acc_test_num = 0.0
    epoch_acc_test_den = 0.0

    # run a for loop through each batch
    for (_, sample_batched) in enumerate(tqdm(dataloader_testing_dataset, desc = f'Final Test')):
        x = sample_batched['image']
        y_actual = sample_batched['label']

        x = x.to(device)
        y_actual  = y_actual.to(device)

        y_predicted = resnet(x)

        pred = np.argmax(y_predicted.cpu().data, axis=1)
        
        actual = y_actual.cpu().data

        loss = criterion(y_predicted, y_actual)

        epoch_losses_test.append(loss.data.item())

        epoch_acc_test_num += (actual == pred).sum().item()
        epoch_acc_test_den += len(actual)

    # calculate test_acc
    losses_test.append(get_mean_of_list(epoch_losses_test))
    test_acc = epoch_acc_test_num / epoch_acc_test_den
    test_acc = round(test_acc, 4)
    test_accs.append(test_acc)

    print(f'Test accuracy: {test_acc}')


    return test_acc

