

from reg_model    import ResNet34Reg
from train_engine import run_training

cfg = {
    "run_name"     : "resnet34",
    "epochs"       : 30,
    "batch_size"   : 16,
    "lr"           : 5e-5,    
    "lr_head"      : 5e-4,   
    "weight_decay" : 5e-4,    
    "patience"     : 15,
    "freeze_epochs": 5,      
    "num_workers"  : 4,
}

if __name__ == "__main__":
    model = ResNet34Reg(freeze_encoder=False, dropout=0.5)
    run_training(cfg, model)
