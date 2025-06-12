import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detector import RectangleDetector

# --- Create the validation DataLoader ---
val_images = torch.load('data/val_images.pt')
val_labels = torch.load('data/val_labels.pt')
val_dataset = TensorDataset(val_images, val_labels)
batch_size = 32
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# -- Load the trained model from the saved state dictionary
model = RectangleDetector()  # Assuming RectangleDetector is defined in detector.py
MODEL_PATH = 'rectnet_model.pth'
model.load_state_dict(torch.load(MODEL_PATH))

# --- Evaluation and Visualization ---
model.eval() # Set the model to evaluation mode
with torch.no_grad():
    # Get a batch of validation data
    images, actual_labels = next(iter(val_loader))
    
    # Make predictions
    predicted_labels = model(images)

    # Plot the first 4 images from the batch
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    image_size = 64

    for i in range(4):
        ax = axes[i]
        # Display the image
        ax.imshow(images[i].squeeze(), cmap='gray')

        # --- Draw Ground Truth Rectangle (in Green) ---
        actual = actual_labels[i].numpy() * image_size
        x1_act, y1_act, x2_act, y2_act = actual
        w_act, h_act = x2_act - x1_act, y2_act - y1_act
        rect_actual = patches.Rectangle((x1_act, y1_act), w_act, h_act, linewidth=2, edgecolor='g', facecolor='none', label='Actual')
        ax.add_patch(rect_actual)
        
        # --- Draw Predicted Rectangle (in Red) ---
        predicted = predicted_labels[i].numpy() * image_size
        x1_pred, y1_pred, x2_pred, y2_pred = predicted
        w_pred, h_pred = x2_pred - x1_pred, y2_pred - y1_pred
        rect_pred = patches.Rectangle((x1_pred, y1_pred), w_pred, h_pred, linewidth=2, edgecolor='r', facecolor='none', label='Predicted')
        ax.add_patch(rect_pred)
        
        ax.set_title(f"Sample {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Model Predictions vs. Ground Truth", fontsize=16)
    plt.show()