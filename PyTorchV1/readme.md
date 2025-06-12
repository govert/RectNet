### RectNet: A Design and Training Rationale

This document outlines the core machine learning decisions involved in creating the `RectNet` model, a convolutional neural network designed for rectangle localization. The goal is not classification (e.g., "is this a cat?") but **regression**—predicting the continuous numerical values of a bounding box `[x_start, y_start, x_end, y_end]`.

### 1. Model Architecture Decisions

The model's architecture is designed to transform spatial information from a 2D image into four numerical coordinates.

#### 1.1. Core Choice: Convolutional Neural Network (CNN)

* **What was chosen:** A CNN as the foundational architecture.
* **Reasoning:** CNNs are the industry standard for tasks involving images because they are specifically designed to recognize spatial hierarchies. They learn to identify simple patterns like edges in early layers, and then combine those patterns into more complex shapes like corners and eventually rectangles in later layers. This is far more effective and parameter-efficient than using a simple fully-connected network, which would ignore the spatial structure of the pixels.
* **Alternatives:** A simple Multi-Layer Perceptron (MLP) could be used, but it would require flattening the image into a 4096-node input vector (64x64) and would lose all spatial information. It would be highly inefficient and likely perform poorly.

#### 1.2. Convolutional Layers (`Conv2d`) & Max Pooling (`MaxPool2d`)

* **What was chosen:** A stack of three `Conv2d` layers, each followed by a `MaxPool2d` layer. The number of channels (filters) increases with each layer (1 -> 16 -> 32 -> 64).
* **Reasoning:**
  * **Hierarchy of Features:** The stack allows the model to build an increasingly abstract understanding of the image. The first layer detects edges, the second uses those edges to find corners, and the third can identify more complete rectangular shapes.
  * **Increasing Channels:** Increasing the number of filters (e.g., from 16 to 32) gives the model more capacity to learn a wider variety of features at each level of abstraction.
  * **Max Pooling:** This downsamples the feature maps. It serves two purposes: it reduces the computational load and, more importantly, it makes the learned features more robust to small shifts in position (a property called translation invariance).
* **Alternatives:**
  * **Deeper or Shallower Network:** We could use more layers (for more complex problems) or fewer (for simpler ones). Three is a reasonable starting point for a 64x64 image.
  * **Different Kernel Sizes:** We used 3x3 kernels, which are standard. Larger 5x5 kernels could capture broader features but at a higher computational cost.
  * **Stride Convolutions:** Instead of `MaxPool2d`, we could use a `Conv2d` layer with a `stride` of 2 to achieve downsampling. This is a more modern approach but slightly more complex.

#### 1.3. The Bridge: Flattening Operation

* **What was chosen:** A "flatten" operation (`x.view(...)`) to convert the final 3D feature map (e.g., 64 channels of 8x8 maps) into a single 1D vector.
* **Reasoning:** This is a **mandatory structural step**. The convolutional layers output 2D feature maps that preserve spatial information. However, the final dense layers (which perform the regression) expect a flat list of numbers as input. Flattening is the bridge between these two parts of the network.

#### 1.4. The Head: Fully-Connected Layers & Output

* **What was chosen:** A sequence of two fully-connected (`Linear`) layers, with the final layer having 4 output neurons and **no activation function**.
* **Reasoning:**
  * **Regression, Not Classification:** The purpose of the head is to map the high-level features learned by the CNN to the four bounding box coordinates.
  * **4 Output Neurons:** Each neuron corresponds directly to one of the required outputs: `x_start`, `y_start`, `x_end`, `y_end`.
  * **No Final Activation Function:** This is the most critical decision for a regression model. Activation functions like `Sigmoid` or `Softmax` are used for classification because they squash outputs into a specific range (like 0 to 1 for probabilities). For regression, we need to predict raw, unbounded numerical values. Applying an activation function would artificially constrain the model's predictions and prevent it from learning correctly.

### 2. Training Process Decisions

The training process is defined by how the model's error is measured and corrected.

#### 2.1. Loss Function: Mean Squared Error (`nn.MSELoss`)

* **What was chosen:** Mean Squared Error (MSE) loss.
* **Reasoning:** MSE is the standard loss function for regression tasks. It calculates the average of the squared differences between the model's predicted coordinates and the actual coordinates. By squaring the error, it heavily penalizes larger mistakes, strongly encouraging the model to make predictions that are very close to the ground truth.
* **Alternatives:**
  * **L1 Loss / Mean Absolute Error (`nn.L1Loss`):** Calculates the average of the absolute differences. This is less sensitive to outliers than MSE but may lead to less precise models.
  * **Smooth L1 Loss:** A hybrid often used in object detection. It acts like MSE for small errors (providing stable gradients near the target) and like L1 for large errors (reducing the penalty from outliers). MSE is a simpler and perfectly valid starting point.

#### 2.2. Optimizer: Adam (`optim.Adam`)

* **What was chosen:** The Adam optimizer.
* **Reasoning:** Adam is a robust, efficient, and widely used "adaptive" optimizer. It maintains a separate learning rate for each model parameter and adapts them as the training progresses. It effectively combines the benefits of other optimizers (like AdaGrad and RMSProp) and generally requires less manual tuning of the learning rate to achieve good performance. It's an excellent default choice for most problems.
* **Alternatives:**
  * **Stochastic Gradient Descent (SGD):** The classic optimizer. It's often very effective but typically requires more careful manual tuning of the learning rate, momentum, and other hyperparameters.
  * **RMSprop:** Another adaptive optimizer that works well. Adam is slightly more complex but is often considered the state-of-the-art go-to for many tasks.