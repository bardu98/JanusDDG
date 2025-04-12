import torch


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

