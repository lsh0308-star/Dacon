import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import timm
from timm.models.layers import ClassifierHead

# 1. 데이터 준비
class MVTecDataset(Dataset):
    def __init__(self, dataframe, root_dir, label_mapping=None, transform=None, is_test=False):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.label_mapping = label_mapping  # 문자열 -> 숫자 매핑
        self.transform = transform
        self.is_test = is_test  # 테스트 데이터 여부

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['file_name'])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if not self.is_test:  # 테스트 데이터가 아니라면 레이블도 반환
            label_str = self.dataframe.iloc[idx]['label']
            label = self.label_mapping[label_str]  # 문자열 레이블을 숫자로 변환
            return image, torch.tensor(label)
        else:  # 테스트 데이터는 이미지만 반환
            return image

# 2. 데이터 로드
train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')

# 문자열 레이블 -> 숫자 매핑 생성
label_mapping = {label: idx for idx, label in enumerate(train_df['label'].unique())}
num_classes = len(label_mapping)  # 총 클래스 수
# print(f"Label Mapping: {label_mapping}")

# 데이터 전처리 (Transform 정의)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),  # 데이터 증강: 랜덤 수평 반전
    transforms.RandomRotation(15),  # 데이터 증강: 랜덤 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 색상 변화
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.RandomErasing(p=0.5, scale=(0.05, 0.20), ratio=(1, 1)),  # 데이터 증강: Random Erase
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),  # 이미지를 텐서로 변환
])

train_dataset = MVTecDataset(train_df, root_dir='train', label_mapping=label_mapping, transform=transform)
test_dataset = MVTecDataset(test_df, root_dir='test', transform=transform_test, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

# 3. Swin-Large 모델 로드 (마지막 분류 레이어 수정)
model = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=768)

num_features = model.head.in_features
new_head = nn.Sequential(
    ClassifierHead(in_features=num_features, num_classes=768, pool_type='avg', drop_rate=0.0, input_fmt='NHWC'),  # ClassifierHead를 먼저 추가
    nn.ReLU(),
    nn.Dropout(p=0.1),
    nn.Linear(768, 256),  # 두 번째 Linear 레이어 추가
    nn.ReLU(),
    nn.Dropout(p=0.1),
    nn.Linear(256, num_classes)  # 최종 클래스 수 (num_classes)로 변환
)

model.head = new_head
model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# 4. 손실 함수, 옵티마이저, 스케줄러
criterion = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# 5. 학습 함수 정의
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=20):
    best_loss = float('inf')  # 가장 낮은 loss 초기화
    best_model_weights = None
    
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            labels = labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            optimizer.zero_grad()
            # print(images.shape, labels.shape)

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 학습 후 스케줄러로 LR 업데이트
        scheduler.step()

        # 현재 에포크와 LR 출력
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, LR: {current_lr:.6f}')
        
        # 가장 낮은 loss가 나올 때마다 모델의 가중치 저장
        if running_loss / len(dataloader) < best_loss:
            print('Best model save!')
            best_loss = running_loss / len(dataloader)
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, 'model_focal_1e-4.pth')

# 6. 모델 학습
train_model(model, train_loader, criterion, optimizer, lr_scheduler, num_epochs=20)

# 7. 추론 및 제출 파일 생성
def inference(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    return predictions

# 학습 후 모델 가중치 로드
model.load_state_dict(torch.load('model_focal_1e-4.pth', weights_only=True))

# 추론 실행
predictions = inference(model, test_loader)

# 숫자 -> 문자열 레이블 변환
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
test_df['label'] = [reverse_label_mapping[pred] for pred in predictions]

# 제출 파일 생성
test_df[['index', 'label']].to_csv('submission_focal_1e-4.csv', index=False)
