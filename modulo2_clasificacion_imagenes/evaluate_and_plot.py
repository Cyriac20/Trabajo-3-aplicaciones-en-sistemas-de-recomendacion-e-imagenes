import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import build_model
import numpy as np

def visualize_predictions(data_dir, model_path='modelo_cnn.pth', num_images=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mismas transformaciones de validación que en train.py
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=val_transforms)
    classes = dataset.classes
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Cargar modelo
    model = build_model(num_classes=len(classes), fine_tune=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    correct_images = []
    incorrect_images = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(inputs.size(0)):
                img = inputs[i].cpu()
                pred = preds[i].item()
                true = labels[i].item()
                
                # Desnormalizar la imagen para mostrarla correctamente
                img = img.numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                if pred == true and len(correct_images) < num_images:
                    correct_images.append((img, classes[pred], classes[true]))
                elif pred != true and len(incorrect_images) < num_images:
                    incorrect_images.append((img, classes[pred], classes[true]))
                    
            if len(correct_images) >= num_images and len(incorrect_images) >= num_images:
                break
                
    # Plotear imágenes correctas
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Predicciones Correctas', fontsize=16)
    for idx, (img, pred_cls, true_cls) in enumerate(correct_images):
        if idx >= 6: break
        ax = axes[idx//3, idx%3]
        ax.imshow(img)
        ax.set_title(f'Pred/Real:\\n{pred_cls}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('predicciones_correctas.png')
    
    # Plotear imágenes incorrectas
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Predicciones Erróneas', fontsize=16)
    for idx, (img, pred_cls, true_cls) in enumerate(incorrect_images):
        if idx >= 6: break
        ax = axes[idx//3, idx%3]
        ax.imshow(img)
        ax.set_title(f'Pred: {pred_cls}\\nReal: {true_cls}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('predicciones_erroneas.png')
    
    print("Las imágenes de predicciones correctas y erróneas se han guardado.")

if __name__ == '__main__':
    DATA_DIRECTORY = '../data/driver_behavior_dataset/Multi-Class Driver Behavior Image Dataset/'
    try:
        visualize_predictions(DATA_DIRECTORY)
    except FileNotFoundError:
         print("Asegúrate de haber entrenado el modelo primero para tener 'modelo_cnn.pth'")
