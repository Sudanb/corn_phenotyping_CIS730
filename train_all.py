from reg_model    import EfficientNetReg, ResNet34Reg, CustomCNNReg
from train_engine import run_training

if __name__ == "__main__":

    print("\n=== [1/3] EfficientNet-B3 ===")
    cfg = {
        "run_name"     : "efficientnet_b3",
        "epochs"       : 30,
        "batch_size"   : 16,
        "lr"           : 1e-6,    
        "lr_head"      : 5e-4,
        "weight_decay" : 5e-4,
        "patience"     : 15,
        "freeze_epochs": 5,
        "num_workers"  : 4,
    }
    run_training(cfg, EfficientNetReg(freeze_encoder=False, dropout=0.5))

    print("\n=== [2/3] ResNet-34 ===")
    cfg = {
        "run_name"     : "resnet34",
        "epochs"       : 30,
        "batch_size"   : 16,
        "lr"           : 1e-6,    # very slow encoder fine-tuning
        "lr_head"      : 5e-4,
        "weight_decay" : 5e-4,
        "patience"     : 15,
        "freeze_epochs": 5,
        "num_workers"  : 4,
    }
    run_training(cfg, ResNet34Reg(freeze_encoder=False, dropout=0.5))

    print("\n=== [3/3] Custom CNN ===")
    cfg = {
        "run_name"     : "custom_cnn",
        "epochs"       : 40,
        "batch_size"   : 16,
        "lr"           : 3e-4,
        "lr_head"      : 3e-4,
        "weight_decay" : 3e-4,
        "patience"     : 20,
        "num_workers"  : 4,
    }
    run_training(cfg, CustomCNNReg(dropout=0.5))

    print("\n=== All training complete ===")
