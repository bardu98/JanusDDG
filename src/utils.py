import numpy as np
import pandas as pd
import random
import torch
import esm
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import random
from sklearn.metrics import root_mean_squared_error,mean_absolute_error, accuracy_score,mean_squared_error
from torch.utils.data import DataLoader, Dataset
import argparse
import os
from data_class import DeltaDataset


#################
# ESM2 Function #
#################

def Esm2_embedding(seq, model_esm = model_esm, batch_converter_esm = batch_converter_esm):

    model_esm, alphabet_esm = esm.pretrained.esm2_t33_650M_UR50D()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_esm = model_esm.to(device)
    batch_converter_esm = alphabet_esm.get_batch_converter()
    model_esm.eval()
    
    sequences = [("protein", seq),]
    
    batch_labels, batch_strs, batch_tokens = batch_converter_esm(sequences)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33])  # Usa l'ultimo layer
        token_representations = results["representations"][33]
    
    # Remove the special tokens 
    embedding = token_representations[0, 1:-1].cpu().numpy()
    return embedding  #Output: L_SEQ X D_EMB


#############################
# DATA PROCESSING FUNCTIONS #
#############################

def old_aa(row, mut_col_name='MTS'):

    #get the list of old AA for all MTS 
    
    old_aa_list = []
    mutations = row[mut_col_name].split('_')
    for mut in mutations:
        old_aa_list.append(mut[0])
    return old_aa_list

def position_aa(row,mut_col_name='MTS'):

    #get the list of positions AA for all MTS 

    pos_aa_list = []
    mutations = row[mut_col_name].split('_')
    for mut in mutations:
        try:
            pos_aa_list.append(int(mut[1:-1]))
        except:
            pos_aa_list.append(int(mut[1:-2]))  #c'era un errore in un dataset 
    
    return pos_aa_list

def new_aa(row,mut_col_name='MTS'):

    #get the list of NEW AA for all MTS     
    
    new_aa_list = []
    mutations = row[mut_col_name].split('_')
    for mut in mutations:
        new_aa_list.append(mut[-1])
    return new_aa_list


def Create_mut_sequence_multiple(sequence_wild, position_real, old_AA, new_AA, debug = True):

    if debug:
        for i,pos in enumerate(position_real):
            assert sequence_wild[pos] == old_AA[i]
        
    mut_sequence = sequence_wild
    mut_sequence = list(mut_sequence)
    for i,pos in enumerate(position_real):
        mut_sequence[pos] = new_AA[i]
        
    mut_sequence= ''.join(mut_sequence)

    return mut_sequence


def dataset_builder(dataset_mutations, dataset_sequences,debug=True):
    
    dataset = [] 
    lista_proteine = set(dataset_sequences['ID'])
    
    for index, item in dataset_mutations.iterrows():
        
        sample_protein = {}

        id = item['ID']
        
        if(id in lista_proteine):

            #Info sulla mutazione
            sample_protein['id'] = id
            
            position_real = [pos for pos in item['Pos_AA']]  #la posizione si conta da 0 invece nel dataset CASTRENSE da 1 AGGIUNGI -1

            num_mutations = len(position_real)

            old_AA = item['Old_AA']
            new_AA = item['New_AA']
            sequence_original = dataset_sequences[dataset_sequences['ID']==id]['Sequence'].item()
            
            #Embedding Wild ESM2
            sequence_wild = sequence_original
            #Embedding Mut ESM2
            try:
                mut_sequence = Create_mut_sequence_multiple(sequence_original, position_real, old_AA, new_AA,debug=debug)
            except:
                print(f'Errore:{id}')
                continue
            
            sample_protein['wild_type'] = Esm2_embedding(sequence_wild)
            sample_protein['mut_type'] = Esm2_embedding(mut_sequence)
            
            #insert true lenght
            sample_protein['length'] = len(sequence_wild)

            #inserisco posizione della mutazione
            sample_protein['pos_mut'] = position_real
            
            assert sample_protein['length'] == sample_protein['wild_type'].shape[0]
            assert sample_protein['length'] == sample_protein['mut_type'].shape[0]

            assert sample_protein['length'] < 3700

            dataset.append(sample_protein)
        else:
            print(f'{id} not in data')

    return dataset


def process_data(path_df):
    
    df = pd.read_csv(path_df)
    
    dataset_mutations = df[['ID','MTS']].copy()
    dataset_sequences = df.loc[:,['ID','Sequence']].copy()
    dataset_sequences = dataset_sequences.drop_duplicates(subset='ID')
    
    dataset_mutations['Old_AA'] = dataset_mutations.apply(old_aa, axis = 1)
    dataset_mutations['Pos_AA'] = dataset_mutations.apply(position_aa, axis = 1)
    dataset_mutations['New_AA'] = dataset_mutations.apply(new_aa, axis = 1)
    dataset_mutations['Pos_AA'] = dataset_mutations['Pos_AA'].map(lambda x: [i-1 for i in x])
    
    dataset_processed = dataset_builder(dataset_mutations, dataset_sequences, debug=False)
    
    return dataset_processed



################################################################
##### PREDICT FUNC #########################################
################################################################


def set_seed(seed):
    random.seed(seed)  # Python random
    np.random.seed(seed)  # Numpy random
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (un singolo dispositivo)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (tutti i dispositivi, se usi multi-GPU)
    torch.backends.cudnn.deterministic = True  # Comportamento deterministico di cuDNN
    torch.backends.cudnn.benchmark = False  # Evita che cuDNN ottimizzi dinamicamente (influisce su riproducibilità)



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




def output_model_from_batch(batch, model, device):

    '''Dato un modello pytorch e batch restituisce: output_modello, True labels'''
    
    x_wild = batch['wild_type'].float().to(device)
    x_mut = batch['mut_type'].float().to(device)
    length = batch['length'].to(device)
    output_ddg = model(x_wild, x_mut, length)
    
    return output_ddg


def dataloader_generation_pred(dataset_test, batch_size = 128, dataloader_shuffle = True, inv= False):
    
    dim_embedding = 1280
    dataset_test = DeltaDataset(dataset_test, dim_embedding, inv = inv)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=dataloader_shuffle, collate_fn=collate_fn)

    return dataloader_test


def model_performance_test(model, dataloader_test):

    model.eval()
    all_predictions_test = []
    
    with torch.no_grad():
       
        for i, batch in enumerate(dataloader_test):

            predictions_test=output_model_from_batch(batch, model, device)
            all_predictions_test.append(predictions_test)
    
    return all_predictions_test



def metrics(pred_dir=None, pred_inv=None, true_dir=None):

    if pred_dir is not None :
        #Dirette
        true_binary_dir = (true_dir > 0).astype(int)
        pred_binary_dir = (pred_dir > 0).astype(int)
        
        print(f'Pearson test dirette: {pearsonr(true_dir,pred_dir)[0]}')   
        print(f'Spearmanr test dirette: {spearmanr(true_dir,pred_dir)[0]}')    
        print(f'RMSE dirette: {root_mean_squared_error(true_dir,pred_dir)}')
        print(f'MAE dirette: {mean_absolute_error(true_dir,pred_dir)}')
        print(f'ACC dirette: {accuracy_score(true_binary_dir,pred_binary_dir)}')
        print(f'MSE dirette: {mean_squared_error(true_dir,pred_dir)}\n')


    
    if pred_inv is not None: 
        #Inverse
        true_binary_inv = (true_dir < 0).astype(int)
        pred_binary_inv = (pred_inv > 0).astype(int)
        
        print(f'Pearson test inverse: {pearsonr(-true_dir,pred_inv)[0]}')   
        print(f'Spearmanr test inverse: {spearmanr(-true_dir,pred_inv)[0]}')    
        print(f'RMSE inverse: {root_mean_squared_error(-true_dir,pred_inv)}')
        print(f'MAE inverse: {mean_absolute_error(-true_dir,pred_inv)}')
        print(f'ACC inverse: {accuracy_score(true_binary_inv,pred_binary_inv)}')
        print(f'MSE inverse: {mean_squared_error(-true_dir,pred_inv)}\n')
    
    if (pred_dir is not None) and (pred_inv is not None):
        #Tot
        print(f'Pearson test tot: {pearsonr(pd.concat([true_dir,-true_dir],axis=0),pd.concat([pred_dir,pred_inv],axis=0))[0]}')   
        print(f'Spearmanr test tot: {spearmanr(pd.concat([true_dir,-true_dir],axis=0),pd.concat([pred_dir,pred_inv],axis=0))[0]}')    
        print(f'RMSE tot: {root_mean_squared_error(pd.concat([true_dir,-true_dir],axis=0),pd.concat([pred_dir,pred_inv],axis=0))}')
        print(f'MAE tot: {mean_absolute_error(pd.concat([true_dir,-true_dir],axis=0),pd.concat([pred_dir,pred_inv],axis=0))}\n')
        print(f'ACC tot: {accuracy_score(pd.concat([true_binary_dir,true_binary_inv],axis=0),pd.concat([pred_binary_dir,pred_binary_inv],axis=0))}\n')
        print(f'MSE tot: {mean_squared_error(pd.concat([true_dir,-true_dir],axis=0),pd.concat([pred_dir,pred_inv],axis=0))}\n')
        
        print(f'PCC d-r: {pearsonr(pred_dir,pred_inv)}\n')
        print(f'anti-symmetry bias: {np.mean(pred_dir + pred_inv)}\n-----------------------\n')



######################
#MAIN.PY FUNCTIONS
#######################

MODELS_DIR = './models'       # Directory for trained models
RESULTS_DIR = './results'     # Directory to save prediction results
DATA_DIR = './data'           # Directory containing input data


def parse_arguments():
    """Parse command line arguments for input dataset path"""
    parser = argparse.ArgumentParser(description="Dataset processing and prediction")
    parser.add_argument("df_path", type=str, help="Path to the input dataset")
    return parser.parse_args()



def load_model(model_name, device):
    """Load pretrained model from MODELS_DIR"""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")

    # Load model and set to evaluation mode
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def process_and_predict(df_path, model, device):
    """
    Process input data and generate predictions
    Returns tuple of (direct_predictions, inverse_predictions)
    """
    # Load and preprocess data
    df_preprocessed = process_data(df_path)

    # Create dataloaders for both directions
    dataloader_test_dir = dataloader_generation_pred(
        dataset_test=df_preprocessed,
        batch_size=1,
        dataloader_shuffle=False,
        inv=False
    )

    dataloader_test_inv = dataloader_generation_pred(
        dataset_test=df_preprocessed,
        batch_size=1,
        dataloader_shuffle=False,
        inv=True
    )

    # Generate predictions and convert to pandas Series
    predictions_dir = pd.Series(torch.cat(model_performance_test(model, dataloader_test_dir), dim=0).cpu().numpy())
    predictions_inv = pd.Series(torch.cat(model_performance_test(model, dataloader_test_inv), dim=0).cpu().numpy())

    return predictions_dir, predictions_inv

def save_results(input_path, predictions, results_dir=RESULTS_DIR):
    """Save predictions to CSV in RESULTS_DIR"""
    df_output = pd.read_csv(input_path)
    df_output['DDG_JanusDDG'] = predictions  # Add prediction column

    # Ensure output directory exists
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"Result_{os.path.basename(input_path)}")

    df_output.to_csv(output_path, index=False)
    return output_path




