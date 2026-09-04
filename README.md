# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement

Images captured from real-world environments may contain unwanted noise that reduces their quality and affects subsequent image-processing tasks. The objective of this experiment is to develop a Convolutional Autoencoder using deep learning to remove noise from images and reconstruct a cleaner version of the original image.

The model is trained by taking a noisy image as input and using the corresponding original image as the target output. The encoder extracts important features from the noisy image, while the decoder reconstructs the denoised image.

Dataset

The MNIST handwritten digit dataset is used for this experiment. It contains grayscale images of handwritten digits from 0 to 9, with each image having a size of 28 × 28 pixels. Artificial Gaussian noise is added to the images to create noisy input images.


## DESIGN STEPS
### STEP 1: 

Load the MNIST dataset and convert the images into tensors using suitable image transformations.

### STEP 2: 

Add artificial Gaussian noise to the original MNIST images to create noisy images that are used as input to the autoencoder.


### STEP 3: 

Design the convolutional autoencoder consisting of an encoder and decoder. The encoder extracts important features and compresses the input image, while the decoder reconstructs the image.

### STEP 4: 

Initialize the model, define Mean Squared Error (MSE) as the loss function, and use the Adam optimizer for updating the model parameters.

### STEP 5: 

Train the autoencoder using noisy images as input and original clean images as target output. Monitor the reconstruction loss for each epoch.


### STEP 6: 

Evaluate the trained model using test images and visualize the Original, Noisy, and Reconstructed images to verify the denoising performance.





## PROGRAM

### Name: SRUTHI A

### Register Number: 212224240162

```python
# DL - Convolutional Autoencoder for Image Denoising using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchsummary import summary

# ============================================================
# 1. Device Configuration
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# 2. Transform
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor()
])


# ============================================================
# 3. Load MNIST Dataset
# ============================================================

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# ============================================================
# 4. Add Noise to Images
# ============================================================

def add_noise(inputs, noise_factor=0.5):

    noisy = inputs + noise_factor * torch.randn_like(inputs)

    noisy = torch.clamp(noisy, 0., 1.)

    return noisy


# ============================================================
# 5. Convolutional Autoencoder Definition
# ============================================================

class DenoisingAutoencoder(nn.Module):

    def __init__(self):

        super(DenoisingAutoencoder, self).__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.encoder = nn.Sequential(

            # 1 x 28 x 28
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            # 32 x 14 x 14
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            # 64 x 7 x 7
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            # 128 x 7 x 7
            nn.ReLU()
        )


        # -------------------------
        # Decoder
        # -------------------------

        self.decoder = nn.Sequential(

            # 128 x 7 x 7
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            # 64 x 7 x 7
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            # 32 x 14 x 14
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            # 1 x 28 x 28
            nn.Sigmoid()
        )


    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x


# ============================================================
# 6. Initialize Model
# ============================================================

model = DenoisingAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 7. Print Model Summary
# ============================================================

print("\nModel Summary:")
summary(model, input_size=(1, 28, 28))


# ============================================================
# 8. Training Function
# ============================================================

def train(model, loader, criterion, optimizer, epochs=5):

    model.train()

    train_losses = []

    for epoch in range(epochs):

        total_loss = 0

        for images, _ in loader:

            images = images.to(device)

            # Create noisy images
            noisy_images = add_noise(images)

            # Forward pass
            outputs = model(noisy_images)

            # Compare reconstructed image with original image
            loss = criterion(outputs, images)

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        train_losses.append(average_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {average_loss:.4f}"
        )

    return train_losses


# ============================================================
# 9. Train the Model
# ============================================================

train_losses = train(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs=5
)


# ============================================================
# 10. Plot Training Loss
# ============================================================

print("\nName: Haashika Sinou")
print("Register Number: 212224110018")

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(train_losses) + 1),
    train_losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss of Convolutional Autoencoder")
plt.grid(True)

plt.show()


# ============================================================
# 11. Visualization Function
# ============================================================

def visualize_denoising(model, loader, num_images=10):

    model.eval()

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            # Add noise
            noisy_images = add_noise(images)

            # Reconstruct / denoise
            outputs = model(noisy_images)

            break


    # Move tensors to CPU
    images = images.cpu()
    noisy_images = noisy_images.cpu()
    outputs = outputs.cpu()


    print("\nName: Haashika Sinou")
    print("Register Number: 212224110018")


    # Create figure
    plt.figure(figsize=(18, 6))


    for i in range(num_images):

        # -------------------------
        # Original Image
        # -------------------------

        ax = plt.subplot(3, num_images, i + 1)

        plt.imshow(
            images[i].squeeze(),
            cmap="gray"
        )

        ax.set_title("Original")

        plt.axis("off")


        # -------------------------
        # Noisy Image
        # -------------------------

        ax = plt.subplot(
            3,
            num_images,
            i + 1 + num_images
        )

        plt.imshow(
            noisy_images[i].squeeze(),
            cmap="gray"
        )

        ax.set_title("Noisy")

        plt.axis("off")


        # -------------------------
        # Denoised Image
        # -------------------------

        ax = plt.subplot(
            3,
            num_images,
            i + 1 + 2 * num_images
        )

        plt.imshow(
            outputs[i].squeeze(),
            cmap="gray"
        )

        ax.set_title("Denoised")

        plt.axis("off")


    plt.tight_layout()

    plt.show()


# ============================================================
# 12. Visualize Results
# ============================================================

visualize_denoising(
    model,
    test_loader,
    num_images=10
)


# ============================================================
# 13. Check Output Shape
# ============================================================

images, _ = next(iter(test_loader))

images = images.to(device)

with torch.no_grad():

    reconstructed = model(images)

print("\nOriginal image shape     :", images.shape)
print("Reconstructed image shape:", reconstructed.shape)
```

### OUTPUT

### Model Summary

<img width="867" height="580" alt="image" src="https://github.com/user-attachments/assets/d682b66c-c0bd-44a3-8b51-3a76bb2a83ec" />

### Training loss

<img width="1011" height="665" alt="image" src="https://github.com/user-attachments/assets/ab2aeb87-2d7e-423d-88be-856441b97521" />


## Original vs Noisy Vs Reconstructed Image


<img width="1042" height="341" alt="image" src="https://github.com/user-attachments/assets/34fd5bd3-407d-4faa-aef2-36250bdc7f9e" />


## RESULT
Thus, a convolutional autoencoder for image denoising was developed and trained successfully.

