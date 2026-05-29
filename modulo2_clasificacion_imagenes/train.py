import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Importamos la función de creación de modelo que definimos en model.py
from model import build_model

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_data_loaders(data_dir, batch_size=32, train_split=0.8):
    """
    Prepara los dataloaders para entrenamiento y validación.
    Se asume que los datos están organizados en subcarpetas por clase (ImageFolder).
    """
    # Transformaciones con Data Augmentation para entrenamiento
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Medias de ImageNet
    ])
    
    # Transformaciones para validación (sin augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Cargar dataset completo
    try:
        full_dataset = datasets.ImageFolder(root=data_dir)
        classes = full_dataset.classes
    except Exception as e:
        print(f"Error cargando los datos desde {data_dir}. Asegúrate de haber extraído el dataset de Kaggle ahí.")
        raise e

    # Dividir en entrenamiento y validación
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_dataset = DatasetWrapper(train_dataset, transform=train_transforms)
    val_dataset = DatasetWrapper(val_dataset, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, classes

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # 1. Preparar Datos
    train_loader, val_loader, classes = get_data_loaders(data_dir, batch_size)
    num_classes = len(classes)
    print(f"Clases detectadas ({num_classes}): {classes}")
    
    # 2. Inicializar Modelo (Transfer Learning)
    model = build_model(num_classes=num_classes, fine_tune=False)
    model = model.to(device)
    
    # 3. Función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    # Solo optimizamos los parámetros de la capa final que sí requieren gradiente
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Variables para guardar la mejor versión
    best_acc = 0.0
    
    # 4. Ciclo de Entrenamiento
    for epoch in range(num_epochs):
        print(f"\\nÉpoca {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # --- Entrenamiento ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # --- Validación ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Guardar el mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'modelo_cnn.pth')
            print(f"Nuevo mejor modelo guardado con precisión de {best_acc:.4f}.")
            
    # 5. Evaluación Final y Métricas
    print("\\n--- Evaluación Final del Mejor Modelo ---")
    model.load_state_dict(torch.load('modelo_cnn.pth'))
    model.eval()
    
    # Métricas sklearn
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    
    # Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión')
    plt.savefig('matriz_confusion.png')
    print("Matriz de confusión guardada como 'matriz_confusion.png'")
    
if __name__ == '__main__':
    # Ruta relativa donde están guardados los datos tras descomprimir.
    DATA_DIRECTORY = '../data/driver_behavior_dataset/Multi-Class Driver Behavior Image Dataset/' 
    
    if os.path.exists(DATA_DIRECTORY):
        train_model(data_dir=DATA_DIRECTORY, num_epochs=10, batch_size=32)
    else:
        print(f"El directorio '{DATA_DIRECTORY}' no existe.")
