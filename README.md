# Brain-Tumor-Classification


**Introduction**
Brain tumor is considered one of the aggressive diseases, among children and adults. Brain tumors account for 85 to 90 percent of all primary Central Nervous System(CNS) tumors.
Brain Tumors are classified as: Benign Tumor, Malignant Tumor, Pituitary Tumor, etc. Proper treatment, planning, and accurate diagnostics should be implemented to improve the life expectancy of the patients. The best technique to detect brain tumors is Magnetic Resonance Imaging (MRI). These MRI images are examined by radiologists.

Application of classification techniques such as Machine Learning can be used to get higher accuracy in finding the tumor. Deep Learning Algorithms (CNN, ANN) can be helpful for doctors aiding in detection of tumors in early stages.

**Dataset Description**
This dataset is obtained from Kaggle.
The dataset is split to Training and Testing folder set and has 2870, 394 images respectively. 
Folder Structure-

Training-
		-glioma_tumor
		-meningioma_tumor
		-no_tumor
		-pituitary_tumor
Testing--
		-glioma_tumor
		-meningioma_tumor
		-no_tumor
		-pituitary_tumor


**Implementation**
Here I have used Python and it's various libraries( Pandas, Numpy, matplotlib, PyTorch). The CNN Model is similar to the AlexNet with added Convolutional Layer & Fully Connected Layer.
ImageFolder and DataLoader function to read the images from different folders and create batches for faster processing. The images within each folder are of varying sizes(512x512,350x350), CNN requires all images to be of the same size. Hence, we use the Resize function from the transforms module to resize all the images to 512x512.

**Model Summary**
+-------------------------+-------------+------------+-------------+--------+------------+
|          Layer          | Feature Map |    Size    | Kernel Size | Stride | Activation |
+-------+-----------------+-------------+------------+-------------+--------+------------+
| Input |      Image      |      1      |  512x512x3 |      -      |    -   |      -     |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|   1   |   Convolution   |      96     | 126x126x96 |    11x11    |    4   |    ReLU    |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|       |   Max Pooling   |      96     |  62x62x96  |     3x3     |    2   |      -     |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|   2   |   Convolution   |     256     |  58x58x256 |     5x5     |    1   |    ReLU    |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|       |   Max Pooling   |     256     |  28x28x256 |     3x3     |    2   |      -     |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|   3   |   Convolution   |     384     |  26x26x384 |     3x3     |    1   |    ReLU    |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|   4   |   Convolution   |     384     |  24x24x384 |     3x3     |    1   |    ReLU    |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|   5   |   Convolution   |     256     |  22x22x256 |     3x3     |    1   |    ReLU    |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|       |   Max Pooling   |     256     |  10x10x256 |     3x3     |    2   |      -     |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|   6   | Fully Connected |      -      |    25600   |      -      |    -   |    ReLU    |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|   7   | Fully Connected |      -      |    9216    |      -      |    -   |    ReLU    |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|   8   | Fully Connected |      -      |    4096    |      -      |    -   |    ReLU    |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|   9   | Fully Connected |      -      |    4096    |      -      |    -   |    ReLU    |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|   10  | Fully Connected |      -      |    1000    |      -      |    -   |    ReLU    |
+-------+-----------------+-------------+------------+-------------+--------+------------+
|   11  | Fully Connected |      -      |      4     |      -      |    -   |   SoftMax  |
+-------+-----------------+-------------+------------+-------------+--------+------------+

**Optimizers**
torch.optim is a package that we use to implement various optimizations algorithm
Here I have mainly focused on finding the optimal batch size and Optimizer function. The two Optimizers used are Adaptive Moment Estimation(Adam) and Stochastic Gradient Descent(SGD).
	**Adam Optimizer**-
		- This uses adaptive learning rates for each parameter.
		- Computational cost is higher due to additional calculations for adaptive learning rates.
		- Speed of convergence is faster, for complex architectures.
	**SGD Optimizer**-
		- This uses a fixed learning rate.
		- Computational cost is lower since it only used gradients.
		- Speed of convergence is slower, but can generalize better in some cases.

**Model Training Time**
Below are the time required to complete the model training for each combination of Optimizer and Batch Size-
Nvidia A100 GPU
+------------+------------------------------+
|            |           Optimizer          |
|            +-------+----------------------+
|            |  Adam | Stochastic Gradient  |
| Batch Size |       |    Descent ( SGD )   |
+------------+-------+----------------------+
|      1     | 24.99 |         13.19        |
+------------+-------+----------------------+
|     32     |  6.65 |         6.54         |
+------------+-------+----------------------+
|     64     |  6.57 |         6.54         |
+------------+-------+----------------------+
* all numbers are in Minutes

Nvidia L4 GPU-
+------------+------------------------------+
|            |           Optimizer          |
|            +-------+----------------------+
|            |  Adam | Stochastic Gradient  |
| Batch Size |       |    Descent ( SGD )   |
+------------+-------+----------------------+
|      1     | 110.1 |         55.6         |
+------------+-------+----------------------+
|     32     |  6.79 |         6.71         |
+------------+-------+----------------------+
|     64     |  6.61 |         6.64         |
+------------+-------+----------------------+
* all numbers are in Minutes


Below are the training loss and accuracy after each epoch-
## **Batch Size = 1**
![1b](https://github.com/user-attachments/assets/548be579-4734-462c-8d7b-ca41f75605b9)
![1a](https://github.com/user-attachments/assets/287eaf59-8079-4773-86b8-ef5767ea22fc)

## **Batch Size = 32**
![2a](https://github.com/user-attachments/assets/585c10ff-2dbd-43ee-80a3-d78ff41b761b)
![2b](https://github.com/user-attachments/assets/4054fa3b-b782-47ce-8033-529a9aea3ead)

## **Batch Size = 64**
![3a](https://github.com/user-attachments/assets/054e2cd4-fbf0-455d-8dd3-ac17c8cf38e8)
![3b](https://github.com/user-attachments/assets/2617715f-4c70-45b9-ab16-a1f6090965cc)

