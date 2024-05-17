# The application consists of two main scripts:

## `app.py` 🖼️

### Overview
The `app.py` script is the main application responsible for implementing the image similarity search. It utilizes the Hugging Face model "microsoft/resnet-50" for feature extraction, allowing users to perform similarity searches on a candidate dataset.

### Usage
1. 🛠️ Install the required dependencies: `numpy`, `streamlit`, `faiss`, `torch`, `PIL`, `transformers`, `datasets`.
2. ▶️ Run the script with `streamlit run app.py`.

### Functionality
- 🔄 Loads the model and necessary resources.
- 🎨 Provides a user-friendly Streamlit UI with sidebar inputs for top_k value and custom query image upload.
- 🚀 Executes the image similarity search algorithm and displays results, including the top-k retrieved images and the query image.
- 💾 Saves the query image and top-k retrieved images in a run folder for further analysis.

## `modeling.py` 🤖

### Overview
The `modeling.py` script contains essential functions for model loading, dataset handling, and feature extraction. It relies on the Hugging Face "microsoft/resnet-50" model for image embeddings and the GATE-engine/mini_imagenet dataset for candidate images.

### Usage
1. 📥 Import the functions into your script or Jupyter Notebook.
2. 🚀 Use the functions to load the model, create an embedded dataset, and perform similarity searches.

### Functions
- 🔍 `check_gpu()`: Checks for GPU availability and returns the device.
- 🔄 `load_model(model_ckpt)`: Loads the feature extractor and model from a Hugging Face checkpoint.
- 📥 `load_candidate_dataset()`: Loads a subset of the GATE-engine/mini_imagenet dataset for candidate images.
- 🎨 `extract_embeddings(image, extractor, model)`: Extracts feature embeddings from an image.
- 🌐 `create_embedded_dataset(candidate_dataset, extractor, model)`: Creates an embedded dataset for similarity searches.
- 🎲 `select_random_query_image(candidate_dataset)`: Selects a random image from the candidate dataset as the query image.
- 🎯 `get_neighbors(query_image, embedded_dataset, top_k=5)`: Retrieves the top-k similar images for a given query image.
- 🖼️ `image_grid(imgs, rows, cols)`: Creates a grid of images for visualization.

### Example Usage
A usage example is provided at the end of the script, demonstrating how to use the functions in a standalone fashion.

## How to run the code:
- ▶️ `streamlit run app.py`
- 🚨 Please make sure to install all necessary libraries before running, and it is recommended to run in a Python virtual environment.
