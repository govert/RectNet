import torch
import numpy as np

def generate_data(num_samples):
    """Generates images with rectangles and their bounding box coordinates."""
    images = []
    labels = []
    image_size = 64

    for _ in range(num_samples):
        # Create a blank black image
        image = np.zeros((image_size, image_size), dtype=np.float32)
        
        # Generate random rectangle parameters
        w = np.random.randint(5, 20)
        h = np.random.randint(5, 20)
        x_start = np.random.randint(0, image_size - w)
        y_start = np.random.randint(0, image_size - h)
        x_end = x_start + w
        y_end = y_start + h

        # Draw the rectangle (white)
        image[y_start:y_end, x_start:x_end] = 1.0
        
        # Add a channel dimension for PyTorch
        images.append(image[np.newaxis, :]) 
        
        # Normalize coordinates for the label
        labels.append([x_start / image_size, y_start / image_size, x_end / image_size, y_end / image_size])

    return torch.from_numpy(np.array(images)), torch.tensor(labels, dtype=torch.float32)

# 1. Generate the data
print("Generating training and validation data...")
train_images, train_labels = generate_data(2000)
val_images, val_labels = generate_data(400)

# 2. Save the tensors to .pt files
print("Saving data to files...")
torch.save(train_images, 'data/train_images.pt')
torch.save(train_labels, 'data/train_labels.pt')

torch.save(val_images, 'data/val_images.pt')
torch.save(val_labels, 'data/val_labels.pt')

print("Data saved successfully!")

