
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/

# !kaggle datasets download -d davilsena/ckdataset

# !kaggle datasets download -d msambare/fer2013


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)


# Applying transform function
transform=transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])


data_dir= '/media/bennettpc/fe11dcbb-f25e-4ef7-83dc-f1876b222064/SHREYA/face_recognition_using_TL/dataset'

data=ImageFolder(root=data_dir,transform=transform)
train, test = train_test_split(data, test_size=0.3, random_state=42)

train_loader= DataLoader(train, batch_size=32, shuffle=True)
test_loader=DataLoader(test, batch_size=32, shuffle=True)


#DenseNet
model_densenet=models.densenet121(pretrained=True)

for param in model_densenet.parameters():
    param.requires_grad=False
num_classes=31
print(num_classes)

num_features=model_densenet.classifier.in_features

# model_densenet.classifier = nn.Linear(num_features, num_classes)

class custom_layers(nn.Module):
    def __init__(self,num_classes):
        super(custom_layers,self).__init__()
        self.fc1=nn.Linear(1024,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,256)
        self.fc4=nn.Linear(256,128)
        self.fc5=nn.Linear(128,num_classes)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.5)

    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc3(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc4(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc5(x)
        return x
    
model_densenet.classifier = custom_layers(num_classes=num_classes)

modelSavedir=os.path.join(os.getcwd(),'model_densenet')
os.makedirs(modelSavedir, exist_ok=True)

model_densenet.to(device)
criterion=nn.CrossEntropyLoss().to(device)
optimizer=optim.Adam(model_densenet.parameters(),lr=0.001)

num_epochs=25
# from tqdm import tqdm
for epoch in range(num_epochs):
    running_loss=0
    correct=0
    total=0
    print('in epoch')
    for i , data in enumerate(train_loader,0):
        inputs,labels= data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs=model_densenet(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        
        _,predicted=torch.max(outputs.data,1)
        total+= labels.size(0)
        correct+= (predicted==labels).sum().item()
        
        accuracy=100* correct/total
 
        print(f'Epoch [{epoch+1}/{num_epochs}] item[{i}/{len(train_loader)}]  Loss= {loss.item():.4f}   Accuracy: {accuracy}')
        
        # correct=0
        # total=0 
        # val_loss=0.0
        # with torch.no_grad():
        #     for inputs, labels in valid_loader:
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #         outputs=model_densenet(inputs)
        #         loss=criterion(outputs,labels)
        #         val_loss+=loss.item()
        #         _,predicted=torch.max(outputs,1)
        #         total+=labels.size(0)
        #         correct+=(predicted==labels).sum().item()
        #         accuracy = 100*correct/total
        #         print(f'Validation - Epoch [{epoch+1}/{num_epochs}] Loss: {val_loss}  Accuracy: {accuracy}%')
    print(f'Epoch [{epoch+1}/{num_epochs}], iter[{i}/{len(train_loader)}] Loss: {running_loss/len(train_loader):.4f} Accuracy: {accuracy:.4f}%')
     
    if epoch == 0:
        best_acc = accuracy
        
    # Save model checkpoint 
    checkpoint = {'epoch': epoch + 1,'state_dict': model_densenet.state_dict(),'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(modelSavedir , f'checkpoint_densenet.pth'))
    
    if accuracy > best_acc:
        torch.save(checkpoint, os.path.join(modelSavedir , f'checkpoint_densenet_acc{accuracy:.4f}.pth'))

        best_acc=accuracy
print("Training completed!")
    

            
model_densenet.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        
        outputs = model_densenet(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Total Accuracy for DenseNet: {accuracy:.2f}%')
    
# with torch.no_grad():
#     correct=0
#     total=0
#     for images,labels in test_loader:
#         images,labels= images.to(device)
#         outputs=model_densenet(images)
#         _,predicted=torch.max(outputs.data,1)
#         total+=labels.size(0)
#         correct+=(predicted==labels).sum().item()
        
#     accurcay=100* correct/total
#     print(f'Total Accuracy for DenseNet: {accuracy:.2f}%')
    
