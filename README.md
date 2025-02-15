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

## **Model Summary**
![image](https://github.com/user-attachments/assets/b48e32f2-fed1-4284-832c-2e6fc7fd1023)

### **Optimizers**
torch.optim is a package that we use to implement various optimizations algorithm
Here I have mainly focused on finding the optimal batch size and Optimizer function. The two Optimizers used are Adaptive Moment Estimation(Adam) and Stochastic Gradient Descent(SGD).

	Adam Optimizer-
		- This uses adaptive learning rates for each parameter.
		- Computational cost is higher due to additional calculations for adaptive learning rates.
		- Speed of convergence is faster, for complex architectures.
	SGD Optimizer-
		- This uses a fixed learning rate.
		- Computational cost is lower since it only used gradients.
		- Speed of convergence is slower, but can generalize better in some cases.

**Model Training Time** <br />
Below are the time required to complete the model training for each combination of Optimizer and Batch Size- 
<br />
Nvidia A100 GPU-

![image](https://github.com/user-attachments/assets/e6bc84a2-7284-46e7-922a-bbe2aa1922f7)


Nvidia L4 GPU-

![image](https://github.com/user-attachments/assets/0c33001a-3d6f-4229-a244-a936ba60c8f6)


Below are the Accuracy and Loss after each Epoch-
## **Batch Size = 1**
![1a](https://github.com/user-attachments/assets/daa6f9da-78cb-4d88-8792-8b135dd03ac7)
![1b](https://github.com/user-attachments/assets/eaaaddb7-cbef-4290-98fd-dc673f528e47)

## **Batch Size = 32**
![2a](https://github.com/user-attachments/assets/598794eb-3eef-49e6-ae29-764d7f09b5b5)
![2b](https://github.com/user-attachments/assets/80f6df05-3946-4ca4-a5fc-4fa7306af701)

## **Batch Size = 64**
![3a](https://github.com/user-attachments/assets/80069a1c-4314-4886-802e-3f9dad2c35a6)
![3b](https://github.com/user-attachments/assets/ac119df5-2ea6-4646-ab93-c4b9b887408a)

