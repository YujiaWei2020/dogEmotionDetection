import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Define the CNN model
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


def predict_emotion(model, image_path):
    try:
        print("1. Starting image processing...")

        # Load image
        print(f"2. Loading image from: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
        print(f"3. Image loaded successfully. Shape: {img.shape}")

        # Convert to grayscale
        print("4. Converting to grayscale...")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("5. Grayscale conversion successful")

        # Resize image
        print("6. Resizing image...")
        img_resized = cv2.resize(img_gray, (48, 48))
        print("7. Resize successful")

        # Normalize
        print("8. Normalizing image...")
        img_normalized = img_resized.astype('float32') / 255.0

        # Convert to tensor
        img_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0)

        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_tensor = img_tensor.to(device)
        model = model.to(device)

        # Set model to evaluation mode
        model.eval()

        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_class].item() * 100

        # Map prediction to emotion
        emotion_map = {0: 'Happy', 1: 'Sad', 2: 'Angry'}
        predicted_emotion = emotion_map[predicted_class]

        # Display results
        plt.figure(figsize=(8, 4))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        # Preprocessed image
        plt.subplot(1, 2, 2)
        plt.imshow(img_resized, cmap='gray')
        plt.title('Preprocessed Image')
        plt.axis('off')

        plt.suptitle(f'Predicted Emotion: {predicted_emotion} ({confidence:.2f}% confident)')
        plt.show()

        return predicted_emotion, confidence

    except Exception as e:
        print(f"Error during image processing: {str(e)}")
        raise


def main():
    print("\nInitializing...")
    # Load the trained model
    device = torch.device('cpu')
    model = DogFaceCNN().to(device)

    try:
        model.load_state_dict(torch.load('dog_emotion_model.pth', map_location=device))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Could not load model: {str(e)}")
        return

    print("\nModel is ready for inference...")

    # Fixed: Don't use input() for hardcoded path
    image_path = "poppy4.JPG"  # Direct path assignment
    print(f"Processing image: {image_path}")

    try:
        # Check if file exists before processing
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file '{image_path}' not found")
            return

        emotion, confidence = predict_emotion(model, image_path)
        print(f"\nPredicted Emotion: {emotion}")
        print(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    main()