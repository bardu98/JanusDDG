################################################################
#####  PROCESS FUNC ################################################
##################################################################

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



###########################
#ESM2
import esm
import torch
model_esm, alphabet_esm = esm.pretrained.esm2_t33_650M_UR50D()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_esm = model_esm.to(device)
batch_converter_esm = alphabet_esm.get_batch_converter()
model_esm.eval()

def Esm2_embedding(seq, model_esm = model_esm, batch_converter_esm = batch_converter_esm):
    sequences = [("protein", seq),]
    
    batch_labels, batch_strs, batch_tokens = batch_converter_esm(sequences)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33])  # Usa l'ultimo layer
        token_representations = results["representations"][33]
    
    # Remove the special tokens 
    embedding = token_representations[0, 1:-1].cpu().numpy()
    return embedding  #Output: L_SEQ X D_EMB
###########################




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



def dataset_builder(dataset_mutations, dataset_sequences,):
    
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
                mut_sequence = Create_mut_sequence_multiple(sequence_original, position_real, old_AA, new_AA)
            except:
                print(f'Errore:{id}')
                continue
            
            sample_protein['wild_type'] = Esm2_embedding(sequence_wild)
            sample_protein['mut_type'] = Esm2_embedding(mut_sequence)
            
            #insert true lenght
            sample_protein['length'] = len(sequence_wild)

            #inserisco posizione della mutazione
            sample_protein['pos_mut'] = position_real
            sample_protein['ddg'] = item['DDG']
            sample_protein['Sequence'] = item['Sequence']
            
            
            assert sample_protein['length'] == sample_protein['wild_type'].shape[0]
            assert sample_protein['length'] == sample_protein['mut_type'].shape[0]

            assert sample_protein['length'] < 3700

            dataset.append(sample_protein)
        else:
            print(f'{id} not in data')

    return dataset











import sys
import numpy as np
from Bio import PDB
import pandas as pd
import seaborn as sns
import os
import pickle
import random
import torch
import esm
import re
import math
from Bio import SeqIO

from utils import old_aa, position_aa, new_aa,Esm2_embedding ,Create_mut_sequence_multiple, dataset_builder
import argparse



def process_data(path_df):
    
    df = pd.read_csv(path_df)
    
    dataset_mutations = df[['ID','MTS']].copy()
    dataset_sequences = df.loc[:,['ID','Sequence']].copy()
    dataset_sequences = dataset_sequences.drop_duplicates(subset='ID')
    
    dataset_mutations['Old_AA'] = dataset_mutations.apply(old_aa, axis = 1)
    dataset_mutations['Pos_AA'] = dataset_mutations.apply(position_aa, axis = 1)
    dataset_mutations['New_AA'] = dataset_mutations.apply(new_aa, axis = 1)
    dataset_mutations['Pos_AA'] = dataset_mutations['Pos_AA'].map(lambda x: [i-1 for i in x])
    
    dataset_processed = dataset_builder(dataset_mutations, dataset_sequences,)
    
    return dataset_processed

################################################################
##### END PROCESS FUNC #########################################
################################################################



################################################################
##### PREDICT FUNC #########################################
################################################################
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import seaborn as sns

from torch_geometric.utils import to_networkx
#install required packages
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
# Helper function for visualization.
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Dataset
import torch_geometric.utils as pyg_utils
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,GATv2Conv
from torch_geometric.nn.models import GCN, GAT
from torch.nn import Linear

from torch_geometric.utils import degree

import torch.nn as nn
from torch_geometric.utils import softmax
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import random
from sklearn.metrics import root_mean_squared_error,mean_absolute_error
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import copy




from random import sample



class Cross_Attention_DDG(nn.Module):
    
    def __init__(self, base_module, cross_att=False, dual_cross_att= False,**transf_parameters):
        super().__init__()
        self.base_ddg = base_module(**transf_parameters, cross_att=cross_att, dual_cross_att= dual_cross_att).to(device)
    
    def forward(self, x_wild, x_mut, length, train = True):

        delta_x = x_wild - x_mut
        output_TCA = self.base_ddg(delta_x, x_wild, length)

        # inv Janus
        delta_x_inv = x_mut -x_wild
        output_TCA_inv = self.base_ddg(delta_x_inv, x_mut, length)
        
        return (output_TCA - output_TCA_inv)/2


def output_model_from_batch(batch, model, device,train=True):

    '''Dato un modello pytorch e batch restituisce: output_modello, True labels'''
    
    x_wild = batch['wild_type'].float().to(device)
    x_mut = batch['mut_type'].float().to(device)
    length = batch['length'].to(device)
    output_ddg = model(x_wild, x_mut, length, train = train)
    
    return output_ddg

import torch
import torch.nn as nn


def apply_masked_pooling(position_attn_output, padding_mask):

    # Convert mask to float for element-wise multiplication
    padding_mask = padding_mask.float()

    # Global Average Pooling (GAP) - Exclude padded tokens
    # Sum only over valid positions (padding_mask is False for valid positions)
    sum_output = torch.sum(position_attn_output * (1 - padding_mask.unsqueeze(-1)), dim=1)  # (batch_size, feature_dim)
    valid_count = torch.sum((1 - padding_mask).float(), dim=1)  # (batch_size,)
    gap = sum_output / valid_count.unsqueeze(-1)  # Divide by number of valid positions

    # Global Max Pooling (GMP) - Exclude padded tokens
    # Set padded positions to -inf so they don't affect the max computation
    position_attn_output_masked = position_attn_output * (1 - padding_mask.unsqueeze(-1)) + (padding_mask.unsqueeze(-1) * (- 1e10))
    gmp, _ = torch.max(position_attn_output_masked, dim=1)  # (batch_size, feature_dim)

    return gap, gmp


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=3700):    

        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, embedding_dim)
        self.register_buffer('pe', pe)  # Salvato come tensore fisso (non parametro)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerRegression(nn.Module):
    def __init__(self, input_dim=1280, num_heads=8, dropout_rate=0., num_experts=1, f_activation = nn.ReLU(), kernel_size=20, cross_att = True,
                dual_cross_att=True):
        
        super(TransformerRegression, self).__init__()
        self.cross_att = cross_att
        self.dual_cross_att = dual_cross_att
        
        print(f'Cross Attention: {cross_att}')
        print(f'Dual Cross Attention: {dual_cross_att}')

        self.embedding_dim = input_dim
        self.act = f_activation                                       
        self.max_len = 3700
        out_channels = 128  #num filtri conv 1D                  
        kernel_size = 20
        padding = 0
        
        self.conv1d = nn.Conv1d(in_channels=self.embedding_dim, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size, 
                                             padding=padding) 
        
        self.conv1d_wild = nn.Conv1d(in_channels=self.embedding_dim, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size, 
                                             padding=padding)

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Cross-attention layers
        self.positional_encoding = SinusoidalPositionalEncoding(out_channels, 3700) 
        self.speach_att_type = True
        self.multihead_attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, dropout=dropout_rate, batch_first=True )
        self.inverse_attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, dropout=dropout_rate, batch_first =True)
        
        if cross_att:
            # Router (learns which expert to choose per token)
            if dual_cross_att:
                dim_position_wise_FFN = out_channels*2
            else:
                dim_position_wise_FFN = out_channels


        else:
            dim_position_wise_FFN = out_channels
        
        self.norm3 = nn.LayerNorm(dim_position_wise_FFN)
        self.norm4 = nn.LayerNorm(dim_position_wise_FFN)        
        self.router = nn.Linear(dim_position_wise_FFN, num_experts) #dim_position_wise_FFN*2
        # Mixture of Experts (Switch FFN)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim_position_wise_FFN, 512),
            self.act,
            nn.Linear(512, dim_position_wise_FFN)
        ) for _ in range(num_experts)])
        # self.experts = nn.Sequential(
        #     nn.Linear(dim_position_wise_FFN, 512),
        #     self.act,
        #     nn.Linear(512, dim_position_wise_FFN)
        #     )
        

        self.Linear_ddg = nn.Linear(dim_position_wise_FFN*2, 1)

    def create_padding_mask(self, length, seq_len, batch_size):
        """
        Create a padding mask for multihead attention.
        length: Tensor of shape (batch_size,) containing the actual lengths of the sequences.
        seq_len: The maximum sequence length.
        batch_size: The number of sequences in the batch.
        
        Returns a padding mask of shape (batch_size, seq_len).
        """
        mask = torch.arange(seq_len, device=length.device).unsqueeze(0) >= length.unsqueeze(1)
        return mask

    def forward(self, delta_w_m, x_wild, length):
            # Add positional encoding
            
            delta_w_m = delta_w_m.transpose(1, 2)  # (batch_size, feature_dim, seq_len) -> (seq_len, batch_size, feature_dim)
            C_delta_w_m = self.conv1d(delta_w_m)
            # C_delta_w_m = self.act(C_delta_w_m)  #CASTRENSE USA RELU IO NON AVEVO MESSO NULLA 
            C_delta_w_m = C_delta_w_m.transpose(1, 2)  # (seq_len, batch_size, feature_dim) -> (batch_size, seq_len, feature_dim)
            C_delta_w_m = self.positional_encoding(C_delta_w_m)
            
            x_wild = x_wild.transpose(1, 2)  # (batch_size, feature_dim, seq_len) -> (seq_len, batch_size, feature_dim)
            C_x_wild = self.conv1d_wild(x_wild)
            # C_x_wild = self.act(C_x_wild)  #CASTRENSE USA RELU IO NON AVEVO MESSO NULLA 
            C_x_wild = C_x_wild.transpose(1, 2)  # (seq_len, batch_size, feature_dim) -> (batch_size, seq_len, feature_dim)
            C_x_wild = self.positional_encoding(C_x_wild)            
            
            batch_size, seq_len, feature_dim = C_x_wild.size()

            padding_mask = self.create_padding_mask(length, seq_len, batch_size)        

            if self.cross_att :
                if self.dual_cross_att:
                    
                    if self.speach_att_type:
                        print('ATTENTION TYPE: Dual cross Attention\n q = wild , k = delta, v = delta and q = delta , k = wild, v = wild \n ----------------------------------')
                        self.speach_att_type = False
                        
                    direct_attn_output, _ = self.multihead_attention(C_x_wild, C_delta_w_m, C_delta_w_m, key_padding_mask=padding_mask)
                    direct_attn_output += C_delta_w_m 
                    direct_attn_output = self.norm1(direct_attn_output)                        
                    
                    inverse_attn_output, _ = self.inverse_attention(C_delta_w_m, C_x_wild, C_x_wild, key_padding_mask=padding_mask)                   
                    inverse_attn_output += C_x_wild  
                    inverse_attn_output = self.norm2(inverse_attn_output)
                    
                    attn_output = torch.cat([direct_attn_output, inverse_attn_output], dim=-1)
                    #combined_output = self.norm3(combined_output)

                else:
                    if self.speach_att_type:
                        print('ATTENTION TYPE: Cross Attention \n q = wild , k = delta, v = delta  \n ----------------------------------')
                        self.speach_att_type = False

                    attn_output, _ = self.multihead_attention(C_x_wild, C_delta_w_m, C_delta_w_m, key_padding_mask=padding_mask)
                    attn_output += C_delta_w_m 
                    attn_output = self.norm1(attn_output) 
            
            else:
                if self.speach_att_type:
                    print('ATTENTION TYPE: Self Attention \n q = delta , k = delta, v = delta  \n ----------------------------------')
                    self.speach_att_type = False
                
                attn_output, _ = self.multihead_attention(C_delta_w_m, C_delta_w_m, C_delta_w_m, key_padding_mask=padding_mask)
                attn_output += C_delta_w_m
                attn_output = self.norm1(attn_output)


            ########
            # Route tokens to experts
            routing_logits = self.router(attn_output)  # Shape: [batch, seq_len, num_experts]
            routing_weights = F.softmax(routing_logits, dim=-1)  # Probability distribution over experts
            expert_indices = torch.argmax(routing_weights, dim=-1)  # Choose the most probable expert for each token
            
            # Apply selected expert
            batch_size, seq_len, embed_dim = attn_output.shape
            output = torch.zeros_like(attn_output)
            for i in range(self.num_experts):
                mask = (expert_indices == i).unsqueeze(-1).float()  # Mask for tokens assigned to expert i
                expert_out = self.experts[i](attn_output) * mask  # Apply expert only to selected tokens
                output += expert_out  # Aggregate expert outputs
            ############ù

            # output = self.experts(attn_output)

            position_attn_output = attn_output + output

            position_attn_output = self.norm3(position_attn_output)
    
            gap, gmp = apply_masked_pooling(position_attn_output, padding_mask)
    
            # Concatenate GAP and GMP
            pooled_output = torch.cat([gap, gmp], dim=-1)  # (batch_size, 2 * feature_dim)
    
            # Pass through FFNN to predict DDG
            x = self.Linear_ddg(pooled_output)        
            
            return x.squeeze(-1)






def model_performance_test(model, dataloader_test):

    model.eval()
    all_predictions_test = []
    
    with torch.no_grad():
       
        for i, batch in enumerate(dataloader_test):

            predictions_test=output_model_from_batch(batch, model, device, train=False)
            all_predictions_test.append(predictions_test)
    
    return all_predictions_test



##############    metrics


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




