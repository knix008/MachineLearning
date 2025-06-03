import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from PIL import Image
import os

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

# DogCat
train_dir = "./data/train/train"
test_dir = "./data/test/test"
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)


class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, dir, mode="train", transform=None):
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        self.transform = transform
        if self.mode == "train":
            if "dog" in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == "train":
            img = img.numpy()
            return img.astype("float32"), self.label
        else:
            img = img.numpy()
            return img.astype("float32"), self.file_list[idx]


data_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor(),
    ]
)

cat_files = [tf for tf in train_files if "cat" in tf]
dog_files = [tf for tf in train_files if "dog" in tf]

cats = CatDogDataset(cat_files, train_dir, transform=data_transform)
dogs = CatDogDataset(dog_files, train_dir, transform=data_transform)

catdogs = torch.utils.data.ConcatDataset([cats, dogs])

# dataset loader
data_loader = torch.utils.data.DataLoader(
    dataset=catdogs, batch_size=batch_size, shuffle=True, drop_last=True
)


# CNN Model (2 conv layers)
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = torch.nn.Linear(3 * 32 * 32, 2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# instantiate CNN model
model = CNN().to(device)

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train my model
total_batch = len(data_loader)
print("Learning started. It takes sometime.")
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost))

print("> Training Finished!")
