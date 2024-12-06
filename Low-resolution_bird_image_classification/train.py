import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from timm import create_model 

# Custom dataset 
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, down_transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.down_transform = down_transform
        
        self.unique_labels = self.data['label'].unique()
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.data['label_index'] = self.data['label'].map(self.label_to_index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        up_scale_path = self.data.iloc[idx, 1]

        label_index = self.data.iloc[idx, 3] 
        img = Image.open(img_path).convert("RGB")
        up_scale_img = Image.open(up_scale_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
            up_scale_img = self.down_transform(up_scale_img)

        return img, up_scale_img, label_index

# CSV directory
csv_file = "./train.csv"

# Data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.05, 0.1), ratio=(0.3, 3.3), value='random', inplace=False)
])

transform_down_sample = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3, interpolation=3),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.05, 0.1), ratio=(0.3, 3.3), value='random', inplace=False)
])


# Dataloader
dataset = CustomDataset(csv_file=csv_file, transform=transform, down_transform=transform_down_sample)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=12)

# Vision Transformer
class ViTClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViTClassifier, self).__init__()
        self.vit = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.vit(x)

num_classes = len(dataset.unique_labels)
model = ViTClassifier(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

# StepLR 
scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
num_epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if __name__ == "__main__":
    best_loss = float('inf') 
    best_model_path = "best_model.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_pred_loss = 0.0
        running_consistency_loss = 0.0

        for images, up_scale_images, labels in dataloader:
            images, up_scale_images, labels = images.to(device), up_scale_images.to(device), labels.to(device)

            outputs = model(images)
            outputs_up_scales = model(up_scale_images)

            pred_loss = criterion(outputs, labels)
            pred_loss += criterion(outputs_up_scales, labels)

            log_probs = nn.functional.log_softmax(outputs, dim=1)
            soft_probs_up_scales = nn.functional.softmax(outputs_up_scales, dim=1)
            consistency_loss = kl_loss(log_probs, soft_probs_up_scales)
            
            loss = pred_loss + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_pred_loss += pred_loss.item()
            running_consistency_loss += consistency_loss.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_pred_loss = running_pred_loss / len(dataloader)
        epoch_consistency_loss = running_consistency_loss / len(dataloader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {epoch_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Prediction Loss: {epoch_pred_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Consistency Loss: {epoch_consistency_loss:.4f}")

        # Best model save
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)

            # Learning rate print
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Best model updated at epoch {epoch + 1} with loss {best_loss:.4f}, Learning Rate: {current_lr:.6f}")

        scheduler.step()

    # Save label
    index_to_label = {idx: label for label, idx in dataset.label_to_index.items()}
    torch.save(index_to_label, "label_mapping.pth")
    print("Model and label save")
