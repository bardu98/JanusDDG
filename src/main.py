import argparse
from utils import *
import random
import torch
import numpy as np
#import data_class

# Config parser
parser = argparse.ArgumentParser(description="Dataset to process.")
parser.add_argument("df_path", type=str, help="Path of the dataset to process")
args = parser.parse_args()
print(f"Dataset processed: {args.df_path}")

# preprocessing with ESM2 650ML param
df_preprocessed = process_data('./data/' + args.df_path)

def set_seed(seed):
    random.seed(seed)  # Python random
    np.random.seed(seed)  # Numpy random
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (un singolo dispositivo)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (tutti i dispositivi, se usi multi-GPU)
    torch.backends.cudnn.deterministic = True  # Comportamento deterministico di cuDNN
    torch.backends.cudnn.benchmark = False  # Evita che cuDNN ottimizzi dinamicamente (influisce su riproducibilità)

# Imposta il seed
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transf_parameters={'input_dim':1280, 'num_heads':8,
                    'dropout_rate':0.,}

best_model=torch.load('./models/DeltaDelta_BELLO/JanusDDG_28epochs_finetuned_zeros_MODELLO_FINALE.pth')#torch.load('JanusDDG_300epochs_FINALE.pth')#torch.load('../../DeltaDelta_BELLO/JanusDDG_28epochs_finetuned_zeros_MODELLO_FINALE.pth')#('../../DeltaDelta_BELLO/JanusDDG_300epochs_plus25_hydra_slim.pth')
best_model.eval()

dataloader_test_dir = dataloader_generation_pred(dataset_test=df_preprocessed,  batch_size = 1, dataloader_shuffle = False, inv= False)
dataloader_test_inv = dataloader_generation_pred(dataset_test=df_preprocessed,  batch_size = 1, dataloader_shuffle = False, inv= True)


#for x in range(10):
all_predictions_test_dir = model_performance_test(best_model,dataloader_test_dir)
all_predictions_test_inv = model_performance_test(best_model,dataloader_test_inv)


df_output = pd.read_csv('./data/' + args.df_path)
df_output['DDG_JanusDDG'] = pd.Series(torch.cat(all_predictions_test_dir, dim=0).cpu()).values
df_output.to_csv(f'./results/Result_{args.df_path}', index=False)

print(f'\n ________Processed: {args.df_path}__________ \n')


#IF DDG COLUMN EXISTS 
try:
    metrics(df_output['DDG_JanusDDG'], pd.Series(torch.cat(all_predictions_test_inv, dim=0).cpu()), df_output['DDG'])
except:
    print('No DDG Column in dataset')




