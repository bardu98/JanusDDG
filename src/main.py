import os
import random
import torch
import numpy as np
import pandas as pd
from utils import process_data,  dataloader_generation_pred, model_performance_test, metrics, parse_arguments, load_model, process_and_predict, save_results
from model import Cross_Attention_DDG, apply_masked_pooling, SinusoidalPositionalEncoding, TransformerRegression
import esm

def main():
    """Main execution pipeline"""
    # Initial configuration
    args = parse_arguments()
    #set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing dataset: {args.df_path}")
    print(f"Using device: {device}")

    try:

        model_esm, alphabet_esm = esm.pretrained.esm2_t33_650M_UR50D()
        model_esm = model_esm.to(device)
        batch_converter_esm = alphabet_esm.get_batch_converter()
        model_esm.eval()
        
        # Load pretrained model
        model = load_model('JanusDDG_fine_tuned.pth',device)

        # Process data and make predictions
        pred_dir, pred_inv = process_and_predict(args.df_path, model,model_esm,batch_converter_esm, device)

        # Save results
        output_file = save_results(args.df_path, pred_dir)
        print(f"Results saved to: {output_file}")

        # Calculate metrics if ground truth exists
        df = pd.read_csv(args.df_path)
        if 'DDG' in df.columns:
            metrics_df = metrics(pred_dir, pred_inv, df['DDG'])
            print("\nEvaluation Metrics:")
            print(metrics_df)

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
