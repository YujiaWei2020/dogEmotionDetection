import kagglehub
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

work_path = os.getcwd()
print(work_path)

# Define the path to the dataset folders
happy_folder = "./archive/happy"
sad_folder = "./archive/Sad"
angry_folder = "./archive/Angry"

# Verify that the folders exist
for folder in [happy_folder, sad_folder, angry_folder]:
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Dataset folder not found: {folder}")


# Function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))  # Resize to a fixed size
            images.append(img)
    return images


# Custom Dataset class
class DogEmotionDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images).unsqueeze(1)  # Add channel dimension
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# CNN model
class DogFaceCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DogFaceCNN, self).__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 32 x 24 x 24

            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 64 x 12 x 12

            # Third block
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 128 x 6 x 6
        )

        # Calculate the size of the flattened features
        self.flatten_size = 128 * 6 * 6  # Based on input size of 48x48

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {running_loss / 100:.4f}, Accuracy: {100 * correct / total:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0


def main():
    # Load and prepare data
    print("Loading images...")
    happy_images = load_images_from_folder(happy_folder)
    sad_images = load_images_from_folder(sad_folder)
    angry_images = load_images_from_folder(angry_folder)

    # Create labels
    happy_labels = [0] * len(happy_images)
    sad_labels = [1] * len(sad_images)
    angry_labels = [2] * len(angry_images)

    # Combine data
    X = np.array(happy_images + sad_images + angry_images)
    y = np.array(happy_labels + sad_labels + angry_labels)

    # Normalize pixel values
    X = X.astype('float32') / 255.0

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = DogEmotionDataset(X_train, y_train)
    test_dataset = DogEmotionDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DogFaceCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=100, device=device)

    # Save the model
    torch.save(model.state_dict(), 'dog_emotion_model.pth')
    print("Training completed and model saved!")


if __name__ == "__main__":
    main()