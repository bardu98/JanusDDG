import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
from utils import *

# Constants for configuration (uppercase naming convention)
MODELS_DIR = './models'       # Directory for trained models
RESULTS_DIR = './results'     # Directory to save prediction results
DATA_DIR = './data'           # Directory containing input data
SEED = 42                     # Random seed for reproducibility
TRANSF_PARAMETERS = {         # Transformer model parameters
    'input_dim': 1280,
    'num_heads': 8,
    'dropout_rate': 0.0
}

def parse_arguments():
    """Parse command line arguments for input dataset path"""
    parser = argparse.ArgumentParser(description="Dataset processing and prediction")
    parser.add_argument("df_path", type=str, help="Path to the input dataset")
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility across all libraries"""
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():  # PyTorch GPU settings
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # Deterministic CuDNN
        torch.backends.cudnn.benchmark = False  # Disable dynamic optimization

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
    full_data_path = os.path.join(DATA_DIR, df_path)
    df_preprocessed = process_data(full_data_path)

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

    # Generate predictions
    predictions_dir = model_performance_test(model, dataloader_test_dir)
    predictions_inv = model_performance_test(model, dataloader_test_inv)

    return (torch.cat(predictions_dir, dim=0).cpu().numpy(),
            torch.cat(predictions_inv, dim=0).cpu().numpy())

def save_results(input_path, predictions, results_dir=RESULTS_DIR):
    """Save predictions to CSV in RESULTS_DIR"""
    df_output = pd.read_csv(input_path)
    df_output['DDG_JanusDDG'] = predictions  # Add prediction column

    # Ensure output directory exists
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"Result_{os.path.basename(input_path)}")

    df_output.to_csv(output_path, index=False)
    return output_path

def main():
    """Main execution pipeline"""
    # Initial configuration
    args = parse_arguments()
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing dataset: {args.df_path}")

    try:
        # Load pretrained model
        model = load_model("JanusDDG.pth", device)

        # Process data and make predictions
        pred_dir, pred_inv = process_and_predict(args.df_path, model, device)

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
