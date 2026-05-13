# TrOCR Training and Testing with Native PyTorch

This project provides scripts for training and testing the TrOCR model using a standard PyTorch training loop, without relying on libraries like Lightning AI or the Hugging Face Trainer. It offers flexibility in configuring training parameters and experimenting with LoRA, DoRA, and LoRA-FA for both the encoder and decoder.

## Project Structure

```
TrOCR_All_In_One/
│
├── config/
│   └── config.yaml           # Configuration file for model and training parameters
│
├── data/
│   ├── train/                # Training data
│   │   ├── images/           # Folder containing training images
│   │   └── labels.csv        # CSV file with training labels (file_name, labels, category)
│   │
│   └── validation/           # Validation data
│       ├── images/           # Folder containing validation images
│       └── labels.csv        # CSV file with validation labels (file_name, labels, category)
│
├── saved_models/             # Directory where trained models are saved
│
├── dataset.py                # PyTorch Dataset class implementation
├── model.py                  # TrOCR model definition
├── test.py                   # Script for testing model performance
├── train.py                  # Script for training the model
└── utils.py                  # Utility functions used across scripts
```


**Description of Folders and Files:**

* **`config/`**: Contains the configuration file (`config.yaml`) for setting training parameters and model configurations.
* **`data/`**: Holds the training and validation datasets.
    * **`train/`**: Contains the training images in the `images/` subfolder and the corresponding labels in `labels.csv`.
    * **`validation/`**: Contains the validation images in the `images/` subfolder and the corresponding labels in `labels.csv`.
* **`saved_models/`**: This folder will store the trained model checkpoints.
    * **`train/`**: History can be found at `126.32.3.23,share=e/AI_DATA/2025/TrOCR`
* **`dataset.py`**: Defines the PyTorch `Dataset` class for loading and processing the image and label data.
* **`model.py`**: Defines the TrOCR model architecture.
* **`test.py`**: Contains the script for evaluating the performance of a trained model on the validation or test dataset.
* **`train.py`**: Contains the main training loop for the TrOCR model.
* **`utils.py`**: Includes utility functions that might be used across different scripts (e.g., saving/loading models).
* **`README.md`**: This file, providing an overview of the project.

## Data Preparation
1. Place your training images in data/train/images/
2. Place your validation images in data/validation/images/
3. Create labels.csv files in both train and validation folders with the following columns:
    - file_name: Name of the image file
    - labels: The text content in the image
    - category: (Optional) Category of the text
The data should be organized under the `data/` folder as follows:
```
data/
├── train/
│   ├── images/          # Contains training image files (e.g., .jpg, .png)
│   └── labels.csv       # CSV file with image filenames and corresponding labels
└── validation/
├── images/          # Contains validation image files
└── labels.csv       # CSV file with image filenames and corresponding labels
```
Sample data can be found at `126.32.3.23,share=e/AI_DATA/2025/TrOCR`  

## Configuration

The training and model parameters are configured in the `config/config.yaml` file. You can modify the following parameters:

```yaml
experiment_name: "trocr_experiment"
train_data_path: "data/train"
validation_data_path: "data/validation"
batch_size: 32
learning_rate: 0.0001
num_epochs: 10
use_half_precision: false # Set to true to enable mixed-precision training
use_lora_encoder: false
use_dora_encoder: false
use_lorafa_encoder: false
lora_rank_encoder: 8
lora_alpha_encoder: 16
lora_target_modules_encoder: ["query", "key", "value"] # Specify target modules for LoRA/DoRA/LoRA-FA
use_lora_decoder: false
use_dora_decoder: false
use_lorafa_decoder: false
lora_rank_decoder: 8
lora_alpha_decoder: 16
lora_target_modules_decoder: ["q_proj", "k_proj", "v_proj"] # Specify target modules for LoRA/DoRA/LoRA-FA
```

## Installation
1. Clone this repository
2. Install the required dependencies:
```
pip install -r  requirements.txt
```
3. Placed base model in **`saved_models`**

## Usage
1. Prepare your data according to the structure described in the Data Preparation section.
2. Configure the training parameters in the config/config.yaml file according to your needs.
3. Run the training script:
```
python train.py
```
The trained model checkpoints will be saved in the saved_models/ folder under a subdirectory named after the experiment_name specified in the configuration file.
4. Run the testing script:
```
python test.py --model **`model_name`**
```

## Scripts Overview
* **`dataset.py`**: This script defines the **`OCRDataset`** class, which inherits from **`torch.utils.data.Dataset`**. It handles loading images and their corresponding labels, and performs any necessary preprocessing steps.

* **`model.py`**: This script defines the TrOCR model architecture. It loads a pre-trained vision transformer and a text decoder from the Hugging Face Transformers library and potentially integrates the LoRA/DoRA/LoRA-FA layers based on the configuration.

* **`train.py`**: This is the main training script. It loads the configuration, initializes the dataset and dataloaders, sets up the TrOCR model and optimizer, and implements the training loop. It also handles validation and saving of the best model.

* **`test.py`**: This script loads a trained model from the specified path and evaluates its performance on the test dataset. It calculates relevant metrics for OCR, such as character error rate (CER) and word error rate (WER).

* **`utils.py`**: This script contains helper functions for calculating the number of trainable parameters in a model.
