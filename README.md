# Vehicle Object Detection with PyTorch

**Overview**:  
This project involves training a deep learning model for vehicle classification using PyTorch. The model is trained on a dataset of over 5,600 images, with each image labeled to represent a different type of vehicle. The vehicle categories include `Car`, `Rickshaw`, `Bike`, `Motorcycle`, `Aircraft`, `Boat`, and `Train`. The aim of the model is to accurately classify these vehicle types.

---

## Performance Evaluation of Model Versions

The following table outlines the accuracy of different versions of the model after training on the dataset:

| Model Version | Accuracy (%) |
|---------------|--------------|
| Version 1     | 85%          |

---

## How To Use The Model

If you want to run this model on your local machine, follow the steps below:

1. **Clone the repository**:  
   First, clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/your-repo/vehicle-object-detection.git
2. **Install Dependencies**:
   Install the required libraries and dependencies. Ensure you have Python 3.x and pip installed, then run:
   ```bash
   pip install -r requirements.txt
3. **Download the trained model**:
  Download the pre-trained model from the link provided () or use the model checkpoint saved in the repository. Make sure to place the .pth file in the appropriate directory
4. **Run the model for inference**:
   After setting up the environment and downloading the model, you can run inference on your test images by executing the following:
   ```bash
   python test_model.py --model_path /path/to/your/model.pth --image_path /path/to/test/image.jpg
   ```
   The script will output the predicted class for the input image.
