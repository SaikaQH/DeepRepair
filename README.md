# DeepRepair


### Installation
- Clone this repo:
```
git clone https://github.com/SaikaQH/DeepRepair.git
cd DeepRepair
```

### Steps
- Prepare image dataset CIFAR-10 and CIFAR-10-C
    CIFAR-10-C: https://zenodo.org/record/2535967
    
- Prepare pre-trained model, and generate failure cases via pre-trained model.

- Preprocess failure cases
```
python preprocess_repair_data.py
```

- Apply style transfer on test cases, with will be used in DeepRepair
```
python style_transfer.py
```

- Repair target DNN model with DeepRepair
```
python repairing.py --gpu {gpu} --model {resnet18, densenet, wrn}
```
