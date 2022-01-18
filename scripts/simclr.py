import numpy as np
import torch
from torchvision import transforms as T
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
import shutil
import copy
from sklearn.metrics import confusion_matrix
import seaborn as sns


# A function to perform color distortion in images
# It is used in SimCLR alongwith random resized cropping
# Here, s is the strength of color distortion.

def get_color_distortion(s=1.0):
    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
    
    # p is the probability of grayscale, here 0.2
    rnd_gray = T.RandomGrayscale(p=0.2)
    color_distort = T.Compose([rnd_color_jitter, rnd_gray])
    
    return color_distort

# this is the dataset class

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
            sample = {'image': image, 'label': label, 'image name': img_name}

        return sample

# Code for NT-Xent Loss function, explained in more detail in the article
def loss_function(a, b, tau):
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)

def get_simclr_resnet18_fc3(fcs = [100,50,25], pre_trained = False):

    resnet = resnet18(pretrained=pre_trained)

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(resnet.fc.in_features, fcs[0])),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(fcs[0], fcs[1])),
        ('added_relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(fcs[1], fcs[2]))
    ]))

    resnet.fc = classifier

    return resnet

class LinearNet(nn.Module):

    def __init__(self, num_classes, fcs):
        super(LinearNet, self).__init__()
        self.fc1 = torch.nn.Linear(fcs[0], num_classes)

    def forward(self, x):
        x = self.fc1(x)
        return(x)

def get_mean_of_list(L):
    return sum(L) / len(L)

def train_simclr_resnet(data, resnet, num_epochs, optimizer, tau, mod_path, device, save_plot = True):
    
    dataloader_training_dataset_mutated, dataloader_training_dataset, dataloader_testing_dataset = data
    
    # Defining data structures for storing training info
    losses_train = []
    # train simclr resnet
    starting_epoch = 0
    # test if any training has already taken place
    resnet_path = mod_path + 'resnet.pt'
    if os.path.exists(resnet_path):
        model_content = torch.load(resnet_path)
        resnet.load_state_dict(model_content['weights'])
        starting_epoch = model_content['epochs']
        try:
            losses_train = model_content['train losses']
        except:
            losses_train = []
        print(f'Loading {starting_epoch} epochs of training for resnet')
    
    # moving the resnet architecture to device
    resnet.to(device)
    
    # get resnet in train mode
    resnet.train()
    
    # run a for loop for num_epochs
    for epoch in range(starting_epoch, num_epochs):
    
        # a list to store losses for each epoch
        epoch_losses_train = []
    
        # run a for loop for each batch
        for (_, sample_batched) in enumerate(tqdm(dataloader_training_dataset_mutated, desc = f'Epoch {epoch + 1} of {num_epochs}')):
            
            # zero out grads
            optimizer.zero_grad()
    
            # retrieve x1 and x2 the two image batches
            x1 = sample_batched['image1']
            x2 = sample_batched['image2']
    
            # move them to the device
            x1 = x1.to(device)
            x2 = x2.to(device)
    
            # get their outputs
            y1 = resnet(x1)
            y2 = resnet(x2)
    
            # get loss value
            loss = loss_function(y1, y2, tau)
            
            # put that loss value in the epoch losses list
            epoch_losses_train.append(loss.cpu().data.item())
    
            # perform backprop on loss value to get gradient values
            loss.backward()
    
            # run the optimizer
            optimizer.step()
    
        # append mean of epoch losses to losses_train, essentially this will reflect mean batch loss
        losses_train.append(get_mean_of_list(epoch_losses_train))
    
        if save_plot:
            # Plot the training losses Graph and save it
            fig = plt.figure(figsize=(10, 10))
            sns.set_style('darkgrid')
            plt.plot(losses_train)
            plt.legend(['Training Losses'])
            plt.savefig(mod_path + 'resnet_losses.png')
            plt.close()
    
        # Store model and optimizer files
        resnet_content = {'weights': resnet.state_dict(),
                          'epochs': epoch + 1,
                          'train losses': losses_train}
        torch.save(resnet_content, mod_path + 'resnet.pt')
        torch.save(optimizer.state_dict(), mod_path + 'resnet_optim.pt')
    
    resnet_content = {'weights': resnet.state_dict(),
                      'epochs': num_epochs,
                      'train losses': losses_train}
    torch.save(resnet_content, mod_path + 'resnet.pt')
    torch.save(optimizer.state_dict(), mod_path + 'resnet_optim.pt')

    
    return resnet

def train_simclr_classifier(data, resnet, linear_classifier, linear_optimizer, num_epochs_linear, mod_path, device, save_plot = True):

    dataloader_training_dataset_mutated, dataloader_training_dataset, dataloader_testing_dataset = data

    # Remove the last few layers and check to ensure you have the right distribution
    resnet.fc = nn.Sequential(*list(resnet.fc.children())[:-3])

    resnet.eval()

    # moving it to device
    linear_classifier.to(device)
    
    # Defining data structures to store train and test info for linear classifier
    losses_train_linear = []
    acc_train_linear = []
    losses_test_linear = []
    acc_test_linear = []
    best_model_weights = linear_classifier.state_dict()
    starting_epoch = 0
    
    # a variable to keep track of the maximum test accuracy, will be useful to store 
    # model parameters with the best test accuracy
    max_test_acc = 0
    
    if os.path.exists(mod_path + 'linear.pt'):
        model_info = torch.load(mod_path + 'linear.pt')
        best_model_weights = model_info['weights']
        linear_classifier.load_state_dict(best_model_weights)
        starting_epoch = model_info['epoch']
        losses_train_linear = model_info['losses train']
        losses_test_linear = model_info['losses test']
        acc_train_linear = model_info['acc train']
        acc_test_linear = model_info['acc test']
        max_test_acc = model_info['max test acc']
        print(f'Loading {starting_epoch} epochs of training for resnet with current max accuracy of {max_test_acc}')
    
    
    # Run a for loop for training the linear classifier
    for epoch in range(starting_epoch, num_epochs_linear):
    
        # run linear classifier in train mode
        linear_classifier.train()
    
        # a list to store losses for each batch in an epoch
        epoch_losses_train_linear = []
        epoch_acc_train_num_linear = 0.0
        epoch_acc_train_den_linear = 0.0
    
        # for loop for running through each batch
        for (_, sample_batched) in enumerate(tqdm(dataloader_training_dataset, desc = f'Epoch {epoch + 1} of {num_epochs_linear}')):
    
            # get x and y from the batch
            x = sample_batched['image']
            y_actual = sample_batched['label']
    
            # move them to the device
            x = x.to(device)
            y_actual  = y_actual.to(device)
    
            # get output from resnet architecture
            y_intermediate = resnet(x)
    
            # zero the grad values
            linear_optimizer.zero_grad()
    
            # run y_intermediate through the linear classifier
            y_predicted = linear_classifier(y_intermediate)
    
            # get the cross entropy loss value
            loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
    
            # add the obtained loss value to this list
            epoch_losses_train_linear.append(loss.data.item())
            
            # perform backprop through the loss value
            loss.backward()
    
            # call the linear_optimizer step function
            linear_optimizer.step()
    
            # get predictions and actual values to cpu  
            pred = np.argmax(y_predicted.cpu().data, axis=1)
            actual = y_actual.cpu().data
    
            #update the numerators and denominators of accuracy
            epoch_acc_train_num_linear += (actual == pred).sum().item()
            epoch_acc_train_den_linear += len(actual)
    
            x = None
            y_intermediate = None
            y_predicted = None
            sample_batched = None
    
        # update losses and acc lists    
        losses_train_linear.append(get_mean_of_list(epoch_losses_train_linear))
        acc_train_linear.append(epoch_acc_train_num_linear / epoch_acc_train_den_linear)
        
        # run linear classifier in eval mode
        linear_classifier.eval()
    
        # essential variables to keep track of losses and acc
        epoch_losses_test_linear = []
        epoch_acc_test_num_linear = 0.0
        epoch_acc_test_den_linear = 0.0
    
        # run a for loop through each batch
        for (_, sample_batched) in enumerate(dataloader_testing_dataset):
            x = sample_batched['image']
            y_actual = sample_batched['label']
    
            x = x.to(device)
            y_actual  = y_actual.to(device)
    
            y_intermediate = resnet(x)
    
            y_predicted = linear_classifier(y_intermediate)
            loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
            epoch_losses_test_linear.append(loss.data.item())
    
            pred = np.argmax(y_predicted.cpu().data, axis=1)
            actual = y_actual.cpu().data
            epoch_acc_test_num_linear += (actual == pred).sum().item()
            epoch_acc_test_den_linear += len(actual)
    
        # calculate test_acc
        test_acc = epoch_acc_test_num_linear / epoch_acc_test_den_linear
    
        losses_test_linear.append(get_mean_of_list(epoch_losses_test_linear))
        acc_test_linear.append(epoch_acc_test_num_linear / epoch_acc_test_den_linear)
    
        if test_acc >= max_test_acc:
    
            # save the model only when test_acc exceeds the current max_test_acc
            max_test_acc = test_acc
            best_model_weights = copy.deepcopy(linear_classifier.state_dict())
            
            print(f'Best accuracy: {max_test_acc}')
    
        model_info = {
            'weights': best_model_weights,
            'epoch': epoch + 1,
            'losses train': losses_train_linear,
            'losses test': losses_test_linear,
            'acc train': acc_train_linear,
            'acc test': acc_test_linear,
            'max test acc': max_test_acc
    
        }
        torch.save(model_info, mod_path + 'linear.pt')
    
    # plotting losses and accuracies
    if save_plot:
    
        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(losses_train_linear)
        plt.plot(losses_test_linear)
        plt.legend(['Training Losses', 'Testing Losses'])
        plt.savefig(mod_path + 'linear_losses.png')
        plt.close()
    
        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(acc_train_linear)
        plt.plot(acc_test_linear)
        plt.legend(['Training Accuracy', 'Testing Accuracy'])
        plt.savefig(mod_path + 'linear_accuracy.png')
        plt.close()
    
    # Save final results
    print(f'Highest Accuracy: {max_test_acc}')
    
    results_fp = '/'.join(mod_path.split('/')[:-2])
    results_p = f'{results_fp}/Results.txt'
    
    with open(results_p, 'a') as f:
        f.write(f'{mod_path} Highest Accuracy: {max_test_acc}\n')

    return linear_classifier

def prep_data(train_batch_size, test_batch_size, root_folder, percentage_supervised, transform, mapping, inverse_mapping, ft_aug = False):

    train_names = sorted(os.listdir(root_folder + '/train'))
    test_names = sorted(os.listdir(root_folder + '/test'))

    # setting random seed to ensure the same 10% labelled data is used when training the linear classifier
    random.seed(0)

    names_train_x_percent = random.sample(train_names, int(len(train_names)*(percentage_supervised/100)))
    names_train = random.sample(train_names, len(train_names))
    names_test = random.sample(test_names, len(test_names))


    # getting labels based on filenames, note that the filenames themselves contain classnames
    # also note that these labels won't be used to actually train the base model
    # these are just for visualization purposes
    labels_train = [mapping[x.split('_')[0]] for x in names_train]
    labels_test = [mapping[x.split('_')[0]] for x in names_test]

    # these 10 percent labels will be used for training the linear classifer
    labels_train_x_percent = [mapping[x.split('_')[0]] for x in names_train_x_percent]

    # datasets
    training_dataset_mutated = MyDataset(root_folder + '/train', names_train, labels_train, transform, mutation = True)
    training_dataset = MyDataset(root_folder + '/train', names_train_x_percent, labels_train_x_percent, transform, train = ft_aug, mutation = False)
    testing_dataset = MyDataset(root_folder + '/test', names_test, labels_test, transform, mutation = False)
    
    # dataloaders
    dataloader_training_dataset_mutated = DataLoader(training_dataset_mutated, 
                                                    batch_size=train_batch_size, 
                                                    shuffle=True, 
                                                    num_workers=0)
    dataloader_training_dataset = DataLoader(training_dataset, 
                                            batch_size=train_batch_size, 
                                            shuffle=True, 
                                            num_workers=0)
    dataloader_testing_dataset = DataLoader(testing_dataset, 
                                            batch_size=test_batch_size, 
                                            shuffle=True, 
                                            num_workers=0)

    return dataloader_training_dataset_mutated, dataloader_training_dataset, dataloader_testing_dataset

def train_test_simclr(args):

    ft_aug, train_batch_size, test_batch_size, percentage_supervised, pre_trained, fcs, tau, resnet_epochs, fine_tuned_epochs, res_s, optim_s, dataset_n, model_folder, root_folder, mapping, inverse_mapping, transform, transform_name = args

    # device is set to cuda if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tau_s = str(tau).replace('.','p')

    #first round path creation
    res_folder = f'resnet_{res_s}/'
    mod_fp = model_folder + res_folder
    if not os.path.exists(mod_fp):
        os.mkdir(mod_fp)
    # second round path creation
    mod_fp = mod_fp + f'{dataset_n}/'
    if not os.path.exists(mod_fp):
        os.mkdir(mod_fp)
      
    mod_path = f'{mod_fp}ftaug_{ft_aug}_trans_{transform_name}_batch_{train_batch_size}_fcs_{fcs[0]}_{fcs[1]}_{fcs[2]}_tau_{tau_s}_optim_{optim_s}_pretrn_{pre_trained}_sup_{percentage_supervised}_eps_{resnet_epochs}_{fine_tuned_epochs}/'

    if os.path.exists(mod_fp + 'Results.txt'):
        with open(mod_fp+'Results.txt', 'r') as f:
            text = f.read()
        if mod_path in text:
            print('Experiment Completed')
            return

    if not os.path.exists(mod_path):
        os.mkdir(mod_path)

    dataloader_training_dataset_mutated, dataloader_training_dataset, dataloader_testing_dataset = prep_data(train_batch_size, test_batch_size, root_folder, percentage_supervised, transform, mapping, inverse_mapping, ft_aug)

    data = [dataloader_training_dataset_mutated, dataloader_training_dataset, dataloader_testing_dataset]

    # defining our deep learning architecture
    resnet = get_simclr_resnet18_fc3(fcs, pre_trained)

    # using SGD optimizer
    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    print('Resnet Training')

    resnet = train_simclr_resnet(data, resnet, resnet_epochs, optimizer, tau, mod_path, device)

    # getting our linear classifier
    linear_classifier = LinearNet(len(inverse_mapping), fcs)

    # using SGD as a linear optimizer
    linear_optimizer = optim.SGD(linear_classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)

    print('Linear Tuning')

    linear_classifier = train_simclr_classifier(data, resnet, linear_classifier, linear_optimizer, fine_tuned_epochs, mod_path, device)

    return mod_path

def get_mapping(data_folder):

    fs = os.listdir(f'{data_folder}/train')
    classes = [m.split('_')[0] for m in fs]
    classes = list(set(classes))
    classes.sort()
    
    mapping = {}
    for i, m in enumerate(classes):
      mapping[m] = i

    return mapping, classes

def clean_images(fps):
  # given a list of image folders, clean any bad images in them
    for fp in fps:
        bad_ims = []
        good_ims = []
        for f in os.listdir(fp):
            try:
                img= plt.imread(fp + f)
                good_ims.append(f)
            except:
                os.remove(fp + f)
                bad_ims.append(f)

def test_trained_model(args, display = False, figsize = (20,20), save_fig = False):

    ft_aug, train_batch_size, test_batch_size, percentage_supervised, pre_trained, fcs, tau, resnet_epochs, fine_tuned_epochs, res_s, optim_s, dataset_n, model_folder, root_folder, mapping, inverse_mapping, transform, transform_name = args
    
    # get model path
    tau_s = str(tau).replace('.','p')
    res_folder = f'resnet_{res_s}/'
    mod_fp = model_folder + res_folder
    mod_fp = mod_fp + f'{dataset_n}/'
    mod_path = f'{mod_fp}ftaug_{ft_aug}_trans_{transform_name}_batch_{train_batch_size}_fcs_{fcs[0]}_{fcs[1]}_{fcs[2]}_tau_{tau_s}_optim_{optim_s}_pretrn_{pre_trained}_sup_{percentage_supervised}_eps_{resnet_epochs}_{fine_tuned_epochs}/'

    # device is set to cuda if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the resnet
    resnet = get_simclr_resnet18_fc3(fcs, False)
    resnet_info = torch.load(mod_path + 'resnet.pt')
    resnet.load_state_dict(resnet_info['weights'])

    # load the linear model
    linear_classifier = LinearNet(len(inverse_mapping), fcs)
    linear_info = torch.load(mod_path + 'linear.pt')
    linear_classifier.load_state_dict(linear_info['weights'])

    resnet.to(device)
    linear_classifier.to(device)

    # Remove the last few layers and check to ensure you have the right distribution
    resnet.fc = nn.Sequential(*list(resnet.fc.children())[:-3])

    resnet.eval()
    # run linear classifier in eval mode
    linear_classifier.eval()

    dataloader_training_dataset_mutated, dataloader_training_dataset, dataloader_testing_dataset = prep_data(train_batch_size, test_batch_size, root_folder, percentage_supervised, transform, mapping, inverse_mapping, ft_aug)

    # essential variables to keep track of losses and acc
    epoch_losses_test_linear = []
    epoch_acc_test_num_linear = 0.0
    epoch_acc_test_den_linear = 0.0
    preds = []
    actuals = []

    # run a for loop through each batch
    for (_, sample_batched) in enumerate(dataloader_testing_dataset):
        x = sample_batched['image']
        y_actual = sample_batched['label']

        x = x.to(device)
        y_actual  = y_actual.to(device)

        y_intermediate = resnet(x)

        y_predicted = linear_classifier(y_intermediate)
        loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
        epoch_losses_test_linear.append(loss.data.item())

        pred = np.argmax(y_predicted.cpu().data, axis=1)
        preds.extend(pred)
        actual = y_actual.cpu().data
        actuals.extend(actual)
        epoch_acc_test_num_linear += (actual == pred).sum().item()
        epoch_acc_test_den_linear += len(actual)

    # calculate test_acc
    test_acc = epoch_acc_test_num_linear / epoch_acc_test_den_linear

    acc = round(test_acc,4)

    print(f'Model Accuracy: {acc}')

    if display:
        cm_map = {}
        for k,v in mapping.items():
            cm_map[v] = k
            
        preds = [cm_map[p.item()] for p in preds]
        actuals = [cm_map[a.item()] for a in actuals]

        cf_matrix = confusion_matrix(actuals, preds)
        cf_matrix = [np.round(r/sum(r),2) for r in cf_matrix]
        plt.figure(figsize=figsize)

        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
        ax.set_xlabel('Predicted Category')
        ax.set_ylabel('Actual Category ');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(inverse_mapping)
        ax.yaxis.set_ticklabels(inverse_mapping)

        ax.tick_params(axis="x", labelrotation=-90)
        ax.tick_params(axis="y", labelrotation=0)

        if save_fig:
            plt_save = mod_path + f'cm_{figsize[0]}_{figsize[1]}.png'
            plt.savefig(plt_save, bbox_inches="tight")

        ## Display the visualization of the Confusion Matrix.
        plt.show()
        return cf_matrix

    return

def semi_train_test(base_folder, new_folder, test_p = 30, max_samples = 1000000000):
    '''
    IN:
      - base_folder: folder of images with folder names corresponding to class names
      - new_folder: folder where train and test split will be placed
      - test_p: percentage as integer that should be placed in the test folder
      - max_samples: integer value for the max samples per class. If a class has
                     more examples than this, a random selection up to this number
                     will be used
    OUT: no variables returned, creates train and test folder in new_folder, 
         with image chips renamed to include the class name for use in 
         semi-supervised learning
    '''

    # create new data folder with train and test sub-folders
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    train_folder = new_folder + 'train/'
    test_folder = new_folder + 'test/'

    if not os.path.exists(test_folder):
        os.mkdir(train_folder)
        os.mkdir(test_folder)
    
    # get list of class names
    cls_fs = os.listdir(base_folder)

    # process on a class by class basis 
    for cls_fp in tqdm(cls_fs):

        # get a list of images within this class, shuffle them
        fp = base_folder + cls_fp + '/'
        images = os.listdir(fp)
        num_samples = len(images)
        np.random.shuffle(images)

        # split using the percentage
        if num_samples < max_samples:
            test_index = int((num_samples)*(test_p)/100)
            test_samples = images[:test_index]
            train_samples = images[test_index:]
        # unless the class is too big
        else:
            test_index = int(max_samples*(test_p)/100)
            test_samples = images[:test_index]
            train_samples = images[test_index:max_samples]
        
        # Move the selected images to the correct new folders
        for t in test_samples:
            src = fp + t
            dst = test_folder + f'{cls_fp}_{t}'
            shutil.copy2(src,dst)
        for t in train_samples:
            src = fp + t
            dst = train_folder + f'{cls_fp}_{t}'
            shutil.copy2(src,dst)

    return 

def copy_resnet_percentages(args, percentages_supervised, base_percentage = 1):
    '''
    For experiments where you keep model parameters the same except for supervision levels, 
    this function can be used to copy the resnet portion of the models - since it is trained
    on the whole dataset in an unsupervised fashion, you only need to train unique linear networks

    - args - experimental args, used to make folder paths
    - percentages_supervises - all supervision levels which will be used
    - base_percentage - the experiment already completed
    '''
    ft_aug, train_batch_size, test_batch_size, percentage_supervised, pre_trained, fcs, tau, resnet_epochs, fine_tuned_epochs, res_s, optim_s, dataset_n, model_folder, root_folder, mapping, inverse_mapping, transform, transform_name = args
    
    percentage_supervised = base_percentage

    # device is set to cuda if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tau_s = str(tau).replace('.','p')

    #first round path creation
    res_folder = f'resnet_{res_s}/'
    mod_fp = model_folder + res_folder
    mod_fp = mod_fp + f'{dataset_n}/'
      
    mod_path = f'{mod_fp}ftaug_{ft_aug}_trans_{transform_name}_batch_{train_batch_size}_fcs_{fcs[0]}_{fcs[1]}_{fcs[2]}_tau_{tau_s}_optim_{optim_s}_pretrn_{pre_trained}_sup_{percentage_supervised}_eps_{resnet_epochs}_{fine_tuned_epochs}/'

    for p in percentages_supervised:
        if p != base_percentage:
            new_mod_path = mod_path.replace(f'sup_{base_percentage}', f'sup_{p}')
            
            if os.path.exists(new_mod_path):
                shutil.rmtree(new_mod_path)
            os.mkdir(new_mod_path)
            src1 = mod_path + 'resnet.pt'
            src2 = mod_path + 'resnet_optim.pt'
            src3 = mod_path + 'resnet_losses.png'

            dst1 = new_mod_path + 'resnet.pt'
            dst2 = new_mod_path + 'resnet_optim.pt'
            dst3 = new_mod_path + 'resnet_losses.png'

            shutil.copy2(src1, dst1)
            shutil.copy2(src2, dst2)
            shutil.copy2(src3, dst3)

    return
