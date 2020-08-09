import cv2
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class FlightDataset(Dataset):
    """
    Custom class to load flight dataset
    """
    def __init__(self, 
               train_transform=None, 
               test_transform=None,
               **kwargs):
        super().__init__(**kwargs)

        self.DATASET_INFO_FILE = 'dataset_info.csv'
        _df = pd.read_csv(self.DATASET_INFO_FILE)
        self.df = _df[_df['valid']].reset_index(drop=True)
        # self.df = self.df[self.df['classes'] != 'Flying Birds'].reset_index(drop=True)

        self.train = True
        self.train_transform = train_transform 
        self.test_transform = test_transform
        self.n_images = len(self.df)

        class_categories = self.df['classes'].astype('category')
        self.df['target'] = class_categories.cat.codes
        self.mapper = dict(enumerate(class_categories.cat.categories))
        w = self.df['classes'].value_counts()
        self.weights = [1] * len(w)
        
        sample_weight = self.df['target'].value_counts()
        self.weights = np.power(10, 1 - (sample_weight/sample_weight.max()))
        self.df['sample_weights'] = self.df['target'].map(self.weights)
        
    def __len__(self):
        return self.n_images
    
    def set_train(self):
        self.train = True
    
    def set_eval(self):
        self.train = False
    
    def __getitem__(self, index):
        img = cv2.imread(self.df['train_paths'].iloc[index], cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.df['target'].iloc[index]

        if self.train:
            if self.train_transform:
                transform = self.train_transform
        else:
            if self.test_transform:
                transform = self.test_transform

        sample = {}
        # Image augmentation
        transformed = transform(image=img)
        img = transformed['image']

        sample['img'] = img
        sample['target'] = target

        return sample