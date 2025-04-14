import torch
from torch.utils.data import Dataset

class InferenceDataset(Dataset):
    def __init__(self, tabular_data, image_features, text_features):
        self.tabular_data = torch.FloatTensor(tabular_data.values)
        self.image_features = torch.FloatTensor(image_features.values)
        self.text_features = torch.FloatTensor(text_features.values)

    def __len__(self):
        return len(self.tabular_data)
    
    def __getitem__(self, index):
        return self.tabular_data[index], self.image_features[index], self.text_features[index]    
    

