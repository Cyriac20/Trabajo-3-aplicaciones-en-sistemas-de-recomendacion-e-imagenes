import torch
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes, fine_tune=False):
    # Cargar modelo preentrenado (ResNet18 o MobileNetV2)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Congelar capas si no hacemos fine-tuning completo
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
            
    # Reemplazar la capa final (cabeza)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
