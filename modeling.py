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

def load_candidate_dataset():
    dataset = load_dataset("GATE-engine/mini_imagenet", split="train[:100]+validation[:100]+test[:100]")
    seed = 98
    num_samples = 100
    candidate_dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    return candidate_dataset

def extract_embeddings(image, extractor, model):
    image_pp = extractor(image, return_tensors="pt")
    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy().squeeze().flatten()
    return features

def create_embedded_dataset(candidate_dataset, extractor, model):
    embedded_dataset = candidate_dataset.map(lambda example: {'embeddings': extract_embeddings(example["image"], extractor, model)})
    embedded_dataset.add_faiss_index(column='embeddings')
    return embedded_dataset

def select_random_query_image(candidate_dataset):
    random_index = np.random.choice(len(candidate_dataset))
    query_image = candidate_dataset[random_index]["image"]
    return query_image

def get_neighbors(query_image, embedded_dataset, top_k=5):
    qi_embedding = model(**extractor(query_image, return_tensors="pt"))
    qi_embedding = qi_embedding.last_hidden_state[:, 0].detach().numpy().squeeze().flatten()
    scores, retrieved_examples = embedded_dataset.get_nearest_examples('embeddings', qi_embedding, k=top_k)
    return retrieved_examples

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# Usage:
device = check_gpu()
model_ckpt = "microsoft/resnet-50"
extractor, model = load_model(model_ckpt)
candidate_dataset = load_candidate_dataset()
embedded_dataset = create_embedded_dataset(candidate_dataset, extractor, model)
query_image = select_random_query_image(candidate_dataset)
top_k_value = 5  # Specify the top_k value
retrieved_examples = get_neighbors(query_image, embedded_dataset, top_k=top_k_value)
images = [query_image]
images.extend(retrieved_examples["image"])
image_grid(images, 1, len(images))
