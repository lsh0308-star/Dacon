import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd 
from train import ViTClassifier

class TestDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx, 0]  
        img_path = self.data.iloc[idx, 1] 
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img_id, img

test_csv_file = "./test.csv"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# test loader
test_dataset = TestDataset(csv_file=test_csv_file, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
index_to_label = torch.load("./label_mapping.pth")

# Model load
model = ViTClassifier(num_classes=25)
model.load_state_dict(torch.load("./best_model.pth"))

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

output_data = []

with torch.no_grad():
    for img_ids, images in test_dataloader:
        images = images.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1) 

        for img_id, pred_label_idx in zip(img_ids, predicted):
            label_name = index_to_label[pred_label_idx.item()] 
            output_data.append({'id': img_id, 'label': label_name})

# Result submission
output_df = pd.DataFrame(output_data)
output_df.to_csv("test_predictions.csv", index=False)

print(".csv file save : test_predictions.csv")
