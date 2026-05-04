"""
Train custom CNN regression model from scratch.
python train_custom.py
"""

from reg_model    import CustomCNNReg
from train_engine import run_training

cfg = {
    "run_name"     : "custom_cnn",
    "epochs"       : 40,      # longer — no pretrained features
    "batch_size"   : 16,
    "lr"           : 3e-4,    
    "lr_head"      : 3e-4,
    "weight_decay" : 3e-4,    
    "patience"     : 20,      
    "num_workers"  : 4,
}

if __name__ == "__main__":
    model = CustomCNNReg(dropout=0.5)
    run_training(cfg, model)
