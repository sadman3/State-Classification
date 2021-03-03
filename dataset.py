import torch
from torchvision import transforms
import torchvision.transforms as transforms
import torch.utils.data as data
import os, glob
import pickle, json
import numpy as np
import random, collections
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------

class StatesDataset(data.Dataset):
    """
        Dataloader to generate dataset
    """

    def __init__(self, data_dir="data", transforms=None, args=None):
        '''
            function to initialize dataset.
        '''

        self.transform = transformation_functions(args, args.mode)

        # you can change this in however manner you want to run your experiments.
        if args.mode == "train":
            dataset_split_path = os.path.join(data_dir, "train")
        elif args.mode == "valid":
            dataset_split_path = os.path.join(data_dir, "valid")
        elif args.mode == "test":
            dataset_split_path = os.path.join(data_dir, "test")

        dataset_class_folders = os.listdir(dataset_split_path)

        self.id_to_path = {}
        self.id_to_class = {}
        self.classes_to_count = {}
        iid = 0
        for class_folder in dataset_class_folders:
            img_paths = glob.glob(os.path.join(dataset_split_path, class_folder, '*.jpg'))
            self.classes_to_count[class_folder] = len(img_paths)
            for path in img_paths:
                self.id_to_path[iid] = path
                self.id_to_class[iid] = class_folder
                iid += 1

        self.class_to_idx, self.idx_to_class = create_class_indice(self.classes_to_count)

        # get ids of all samples in this split of the dataset
        self.ids = list(self.id_to_path.keys())

        self.num_classes = len(self.classes_to_count)
        self.data_size = sum(self.classes_to_count.values())
        self.instance_size = args.crop_size * args.crop_size * 3

        print ("Data loader in {} mode!".format(args.mode))
        print ("Number of classes: {}.".format(self.num_classes))
        print ("Data size: {}.".format(self.data_size))
        print (" ---------- --------- ---------- \n")

    def shuffle(self):
        random.shuffle(self.ids)

    def get_data_size(self):
        return self.data_size

    def get_num_classes(self):
        return self.num_classes

    def get_instance_size(self):
        return self.instance_size

    def __getitem__(self, iid):
        """
            Returns image, class index
        """

        iid = self.ids[iid]
        
        path = self.id_to_path[iid]

        class_name = self.id_to_class[iid]
        class_idx = self.class_to_idx[class_name]
        # load image
        image_input = Image.open(self.id_to_path[iid]).convert('RGB')
        if self.transform is not None:
            image_input = self.transform(image_input)

        # vectorize the image (you should remove this line for your code)
        #image_input = image_input.reshape((-1,))

        return image_input, class_idx, path

    def __len__(self):
        return len(self.ids)

# -------------------------------------------

def collate_fn(batch):
    '''
        collate function to gather data into batches
    '''

    image_input, class_idx, path  = zip(*batch)

    image_input = torch.stack(image_input, 0)
    class_idxs = torch.tensor(class_idx)

    return image_input, class_idxs, path

# -------------------------------------------

def get_loader(data_dir, batch_size, shuffle, num_workers, drop_last=False, args=None):
    '''
        data loader function
    '''

    dataset = StatesDataset(data_dir=data_dir, args=args)
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                              drop_last=drop_last, collate_fn=collate_fn, pin_memory=True)
    return data_loader, dataset

# -------------------------------------------

"""
   Auxilary data functions
"""

def create_class_indice(names):

    idx_to_class = {}
    class_to_idx = {}
    idx = 0
    for name in sorted(names):
        class_to_idx[name] = idx
        idx_to_class[idx] = name
        idx += 1

    return class_to_idx, idx_to_class

# you can add new augmentations to this function
def transformation_functions(args, mode="train"):
    '''
        apply transfomrations to input images.
        the type of transformation (augmentations) used is different at train and evaluation time.
        mode: can be used to explicitly specify separate operations for train and evaluation mode (e.g. augmentation)
    '''

    transforms_list = [transforms.Resize((args.image_size))]
    if mode == 'train':
        transforms_list.append(transforms.RandomRotation(degrees=15))
        transforms_list.append(transforms.ColorJitter(hue=0.5,saturation=0.5))
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.GaussianBlur(19))
    
    transforms_list.append(transforms.CenterCrop(args.crop_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))
    

    return transforms.Compose(transforms_list) 

# -------------------------------------------
