import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
import streamlit as st
from PIL import Image
from IPython.display import display

def check_gpu():
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

def load_model(model_ckpt):
    extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    hidden_dim = model.config.hidden_sizes[3]
    return extractor, model

# Load Cats_vs_Dogs dataset from Hugging Face
def load_candidate_dataset():
    dataset = load_dataset('cats_vs_dogs', split='train[:1000]')  # Adjust the sample size as needed
    return dataset

def extract_embeddings(image):
    image = image.convert('RGB')
    image_pp = extractor(image, return_tensors="pt")
    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy().squeeze().flatten()
    return features

def create_embedded_dataset(dataset, extractor, model):
    dataset_with_embeddings = dataset.map(lambda example: {'embeddings': extract_embeddings(example["image"])})
    dataset_with_embeddings.add_faiss_index(column='embeddings')
    return dataset_with_embeddings

def select_random_query_image(candidate_dataset):
    random_index = np.random.choice(len(candidate_dataset))
    query_image = candidate_dataset[random_index]["image"]
    return query_image

def get_neighbors(query_image, top_k=5):
    query_image = query_image.convert('RGB')
    qi_embedding = model(**extractor(query_image, return_tensors="pt"))
    qi_embedding = qi_embedding.last_hidden_state[:, 0].detach().numpy().squeeze().flatten()
    scores, retrieved_examples = dataset_with_embeddings.get_nearest_examples('embeddings', qi_embedding, k=top_k)
    return retrieved_examples

def image_grid(imgs, rows, cols):
    resized_imgs = [img.resize((224, 224)) for img in imgs]  # Resize each image to a fixed size
    w,h = resized_imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(resized_imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# Usage:
device = check_gpu()
model_ckpt = "microsoft/resnet-50"
extractor, model = load_model(model_ckpt)
dataset = load_candidate_dataset()  # Load the dataset
dataset_with_embeddings = create_embedded_dataset(dataset, extractor, model)
query_image = dataset[0]["image"]  # Change this to select a specific query image
top_k_value = 5  # Specify the top_k value
retrieved_examples = get_neighbors(query_image, top_k=top_k_value)
images = [query_image]
images.extend(retrieved_examples["image"])
image_grid(images, 1, len(images))
