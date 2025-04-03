import torch
#install required packages
import os
import torch
os.environ['TORCH'] = torch.__version__
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F



class DeltaDataset(Dataset):
    def __init__(self, data, dim_embedding, inv = False):
        self.data = data
        self.dim_embedding = dim_embedding
        self.inv = inv

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.inv: 
            return {
                'id': sample['id'],
                'wild_type': torch.tensor(sample['mut_type'], dtype=torch.float32),    #inverto mut con wild 
                'mut_type': torch.tensor(sample['wild_type'], dtype=torch.float32),    #inverto mut con wild             
                'length': torch.tensor(sample['length'], dtype=torch.float32),
                }

        else:
            return {
                'id': sample['id'],
                'wild_type': torch.tensor(sample['wild_type'], dtype=torch.float32),
                'mut_type': torch.tensor(sample['mut_type'],dtype=torch.float32),
                'length': torch.tensor(sample['length'], dtype=torch.float32),
                }



def collate_fn(batch):
    max_len = max(sample['wild_type'].shape[0] for sample in batch)  
    max_features = max(sample['wild_type'].shape[1] for sample in batch)  # Max feature size

    padded_batch = {
        'id': [],
        'wild_type': [],
        'mut_type': [],
        'length': [],
        }
    for sample in batch:
        wild_type_padded = F.pad(sample['wild_type'], (0, max_features - sample['wild_type'].shape[1], 
                                                       0, max_len - sample['wild_type'].shape[0]))
        mut_type_padded = F.pad(sample['mut_type'], (0, max_features - sample['mut_type'].shape[1], 
                                                     0, max_len - sample['mut_type'].shape[0]))

        padded_batch['id'].append(sample['id'])  
        padded_batch['wild_type'].append(wild_type_padded)  
        padded_batch['mut_type'].append(mut_type_padded)  
        padded_batch['length'].append(sample['length'])#append(torch.tensor(sample['length'], dtype=torch.float32))  

    # Convert list of tensors into a single batch tensor
    padded_batch['wild_type'] = torch.stack(padded_batch['wild_type'])  # Shape: (batch_size, max_len, max_features)
    padded_batch['mut_type'] = torch.stack(padded_batch['mut_type'])  
    padded_batch['length'] = torch.stack(padded_batch['length'])  
    
    return padded_batch




def dataloader_generation_pred(dataset_test, batch_size = 128, dataloader_shuffle = True, inv= False):
    
    dim_embedding = 1280
    dataset_test = DeltaDataset(dataset_test, dim_embedding, inv = inv)
    # Creazione DataLoader
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=dataloader_shuffle, collate_fn=collate_fn)

    return dataloader_test



