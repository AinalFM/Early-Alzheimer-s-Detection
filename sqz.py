# This code shows how to do alzheimer detection using SqueezetNet pre-trained model from Torchvision

import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix
from imblearn.combine import SMOTEENN
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import psutil

# Definisikan transformasi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path ke folder gambar
data_path = 'alz-dataset/dataset'

# Muat data menggunakan ImageFolder
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Hitung jumlah gambar per kelas
class_names = dataset.classes
class_count = {class_name: 0 for class_name in class_names}

for _, label in dataset:
    class_name = class_names[label]
    class_count[class_name] += 1

print("Jumlah gambar per kelas:")
for class_name, count in class_count.items():
    print(f'{class_name}: {count} gambar')

# Hitung rasio minor-to-major
min_count = min(class_count.values())
max_count = max(class_count.values())
minor_to_major_ratio = min_count / max_count

print(f'\nRasio Minor-to-Major: {minor_to_major_ratio:.2f}')

# Periksa apakah dataset imbalanced
if minor_to_major_ratio < 0.5:
    print("Dataset bersifat imbalanced.")
else:
    print("Dataset bersifat balanced.")

# Balancing Data
# 1. SMOTE-ENN
# Ekstrak data dan label dari dataset
X = np.array([data[0].numpy() for data in dataset])
y = np.array([target for _, target in dataset])

# Flatten data untuk digunakan dengan SMOTE-ENN
n_samples, n_channels, height, width = X.shape
X_flattened = X.reshape((n_samples, -1))

# Inisialisasi SMOTE-ENN
smote_enn = SMOTEENN(random_state=42)

# Resample data menggunakan SMOTE-ENN
X_resampled, y_resampled = smote_enn.fit_resample(X_flattened, y)

# Reshape kembali data ke bentuk asli
X_resampled = X_resampled.reshape((-1, n_channels, height, width))

# Konversi kembali ke format tensor PyTorch
X_resampled = torch.from_numpy(X_resampled).float()
y_resampled = torch.from_numpy(y_resampled).long()

# Buat dataset baru dari data yang sudah diseimbangkan
class BalancedDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
#             # Convert tensor to PIL image for transformations
#             image = transforms.ToPILImage()(image)
#             image = self.transform(image)
        
        
        return image, label

balanced_dataset = BalancedDataset(X_resampled, y_resampled)

# Menampilkan jumlah data per kelas setelah balancing
counter = Counter(y_resampled.numpy())
print("\nJumlah gambar per kelas setelah balancing:")
for class_name, count in zip(class_names, counter.values()):
    print(f'{class_name}: {count} gambar')

# Visualisasi distribusi kelas setelah balancing
plt.figure(figsize=(10, 5))
plt.bar(class_names, counter.values())
plt.xlabel('Kelas')
plt.ylabel('Jumlah Gambar')
plt.title('Distribusi Kelas Setelah Balancing')
plt.show()
plt.savefig('Distribusi Kelas Setelah Balancing pada SqueezeNet.png')
plt.close()

# Hitung rasio minor-to-major setelah dibalancing
min_count_balanced = min(counter.values())
max_count_balanced = max(counter.values())
minor_to_major_ratio_balanced = min_count_balanced / max_count_balanced

print(f'\nRasio Minor-to-Major setelah balancing: {minor_to_major_ratio_balanced:.2f}')

# Periksa apakah dataset sudah balanced
if minor_to_major_ratio_balanced < 0.5:
    print("Dataset masih bersifat imbalanced setelah balancing.")
else:
    print("Dataset bersifat balanced setelah balancing.")

# Tentukan rasio untuk split dataset
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Hitung ukuran dataset berdasarkan rasio
train_size = int(train_ratio * len(balanced_dataset))
val_size = int(val_ratio * len(balanced_dataset))
test_size = len(balanced_dataset) - train_size - val_size

# Split dataset menjadi data training, validation, dan testing
train_set, val_set, test_set = random_split(balanced_dataset, [train_size, val_size, test_size])

# Buat DataLoader untuk masing-masing dataset
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

print(f"\nJumlah data training: {len(train_set)}")
print(f"Jumlah data validation: {len(val_set)}")
print(f"Jumlah data testing: {len(test_set)}")

# Load pre-trained SqueezeNet model
model = torchvision.models.squeezenet1_1(weights='IMAGENET1K_V1')

num_classes = 4
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1))
)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Menampilkan summary model menggunakan torchsummary
from torchsummary import summary
summary(model, input_size=(3, 224, 224))

# Memory usage
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_MB = memory_info.rss / 1024 ** 2
    print(f"Memory Usage: {memory_usage_MB:.2f} MB")
    
# Menampilkan penggunaan memori sebelum dan setelah training
print("Memory Usage Before Training:")
print_memory_usage()

start_time = time.time()
train_losses_history = []
val_losses_history = []
train_auc_history = []
val_auc_history = []
test_losses_history = []
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    train_predictions = []
    train_true_labels = []
    train_probabilities = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        train_predictions.extend(predicted.cpu().numpy())
        train_true_labels.extend(labels.cpu().numpy())
        train_probabilities.extend(probabilities.detach().cpu().numpy())
    
    train_true_labels = np.array(train_true_labels)
    train_probabilities = np.array(train_probabilities)
    
    # Menambahkan dummy labels dan probabilities untuk memastikan semua kelas terwakili
    for class_idx in range(num_classes):
        if class_idx not in train_true_labels:
            train_true_labels = np.append(train_true_labels, class_idx)
            dummy_prob = np.zeros(num_classes)
            dummy_prob[class_idx] = 1
            train_probabilities = np.vstack([train_probabilities, dummy_prob])
    
    # Check for NaN values
    if np.any(np.isnan(train_true_labels)) or np.any(np.isnan(train_predictions)) or np.any(np.isnan(train_probabilities)):
        print("NaN values found in training data")
        continue
    
    train_acc = accuracy_score(train_true_labels, train_predictions)
    train_f1 = f1_score(train_true_labels, train_predictions, average='macro')
    train_precision = precision_score(train_true_labels, train_predictions, average='macro', zero_division=0)
    train_recall = recall_score(train_true_labels, train_predictions, average='macro', zero_division=0)
    train_auc = roc_auc_score(train_true_labels, train_probabilities, multi_class='ovr')
    
    train_losses_history.append(np.mean(train_losses))
    train_auc_history.append(train_auc)

    model.eval()
    val_losses = []
    val_predictions = []
    val_true_labels = []
    val_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            val_predictions.extend(predicted.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())
            val_probabilities.extend(probabilities.detach().cpu().numpy())
    
    val_true_labels = np.array(val_true_labels)
    val_probabilities = np.array(val_probabilities)
    
    # Menambahkan dummy labels dan probabilities untuk memastikan semua kelas terwakili
    for class_idx in range(num_classes):
        if class_idx not in val_true_labels:
            val_true_labels = np.append(val_true_labels, class_idx)
            dummy_prob = np.zeros(num_classes)
            dummy_prob[class_idx] = 1
            val_probabilities = np.vstack([val_probabilities, dummy_prob])
    
    # Check for NaN values
    if np.any(np.isnan(val_true_labels)) or np.any(np.isnan(val_predictions)) or np.any(np.isnan(val_probabilities)):
        print("NaN values found in validation data")
        continue
    
    val_acc = accuracy_score(val_true_labels, val_predictions)
    val_f1 = f1_score(val_true_labels, val_predictions, average='macro')
    val_precision = precision_score(val_true_labels, val_predictions, average='macro', zero_division=0)
    val_recall = recall_score(val_true_labels, val_predictions, average='macro', zero_division=0)
    val_auc = roc_auc_score(val_true_labels, val_probabilities, multi_class='ovr')
    
    val_losses_history.append(np.mean(val_losses))
    val_auc_history.append(val_auc)
    
    # Print epoch metrics
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Training Loss: {np.mean(train_losses):.4f}")
    print(f"Validation Loss: {np.mean(val_losses):.4f}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Training AUC: {train_auc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

# Hitung waktu training
end_time = time.time()
total_training_time = end_time - start_time
print(f"Total Training Time: {total_training_time:.2f} seconds")

# Menampilkan penggunaan memori setelah training
print("Memory Usage After Training:")
print_memory_usage()

# Plot training & validation loss history
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses_history, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()
plt.savefig('Training & Validation Loss pada SqueezeNet.png')
plt.close()

# Plot training & validation AUC history
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_auc_history, label='Training AUC')
plt.plot(range(1, num_epochs+1), val_auc_history, label='Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Training & Validation AUC')
plt.legend()
plt.show()
plt.savefig('Training & Validation AUC pada SqueezeNet.png')
plt.close()

# Evaluate on test set
model.eval()
test_losses = []
test_predictions = []
test_true_labels = []
test_probabilities = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        test_losses.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        test_predictions.extend(predicted.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())
        test_probabilities.extend(probabilities.detach().cpu().numpy())

test_true_labels = np.array(test_true_labels)
test_probabilities = np.array(test_probabilities)

# Menambahkan dummy labels dan probabilities untuk memastikan semua kelas terwakili
for class_idx in range(num_classes):
    if class_idx not in test_true_labels:
        test_true_labels = np.append(test_true_labels, class_idx)
        dummy_prob = np.zeros(num_classes)
        dummy_prob[class_idx] = 1
        test_probabilities = np.vstack([test_probabilities, dummy_prob])

# Check for NaN values
if np.any(np.isnan(test_true_labels)) or np.any(np.isnan(test_predictions)) or np.any(np.isnan(test_probabilities)):
    print("NaN values found in test data")
else:
    test_acc = accuracy_score(test_true_labels, test_predictions)
    test_f1 = f1_score(test_true_labels, test_predictions, average='macro')
    test_precision = precision_score(test_true_labels, test_predictions, average='macro', zero_division=0)
    test_recall = recall_score(test_true_labels, test_predictions, average='macro', zero_division=0)
    test_auc = roc_auc_score(test_true_labels, test_probabilities, multi_class='ovr')

    print(f"\nTest Loss: {np.mean(test_losses):.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

# Menampilkan confusion matrix pada training set
train_confusion_matrix = confusion_matrix(train_true_labels, train_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(train_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Training Set)')
plt.show()
plt.savefig('Confusion Matrix (Training Set) pada SqueezeNet.png')
plt.close()

# Menampilkan confusion matrix pada data training
test_confusion_matrix = confusion_matrix(test_true_labels, test_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Testing Set)')
plt.show()
plt.savefig('Confusion Matrix (Testing Set) pada SqueezeNet.png')
plt.close()

# Prediksi pada satu gambar contoh
def predict_single_image(image_path, model, transform, device, class_names):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_prob = probabilities[0][predicted_class].item()
    return class_names[predicted_class], predicted_prob

# Path ke gambar contoh
image_path = 'alz-dataset/dataset/VeryMildDemented/verymildDem1071.jpg'  # Ganti dengan path ke gambar contoh Anda
predicted_class, predicted_prob = predict_single_image(image_path, model, transform, device, class_names)
print(f"Predicted class: {predicted_class}, Probability: {predicted_prob:.4f}")

# Simpan model
model_path = 'sqznet_model.pth'
torch.save({'model_state_dict': model.state_dict()}, model_path)
# torch.save(model.state_dict(), model_path)
print(f'Model disimpan di {model_path}')
