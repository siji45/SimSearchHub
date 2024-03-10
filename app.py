import streamlit as st
import os
from PIL import Image
from modeling import check_gpu, load_model, load_candidate_dataset, create_embedded_dataset, get_neighbors, image_grid, select_random_query_image

# Set the page layout
st.set_page_config(page_title="SimSearchHub", page_icon="🚀", layout="wide")

# Streamlit UI
st.title("🚀 SimSearchHub: Image Similarity Search")

# Load model and other resources
device = check_gpu()
model_ckpt = "microsoft/resnet-50"
extractor, model = load_model(model_ckpt)
candidate_dataset = load_candidate_dataset()
embedded_dataset = create_embedded_dataset(candidate_dataset, extractor, model)

# Sidebar with top_k input
top_k_value = st.sidebar.number_input("Enter top_k value:", min_value=1, value=5, step=1)

# File uploader for custom query image
uploaded_file = st.sidebar.file_uploader("Upload Custom Query Image", type=["jpg", "jpeg", "png"])

# Initialize i
i = 1

# Run algorithm button
if st.sidebar.button("Run Algorithm"):

    if uploaded_file is not None:
        # Use the uploaded custom query image
        query_image = Image.open(uploaded_file).convert("RGB")
    else:
        # Get random query image
        query_image = select_random_query_image(candidate_dataset)

    # Run algorithm
    retrieved_examples = get_neighbors(query_image, embedded_dataset, top_k=top_k_value)

    # Find a unique name for the run folder
    run_folder = f"Database/run_{i}"
    while os.path.exists(run_folder):
        i += 1
        run_folder = f"Database/run_{i}"

    # Create directories
    os.makedirs(run_folder, exist_ok=True)

    # Display current run number
    st.sidebar.info(f"Current Run: {i}")

    # Display results
    st.subheader("Query Image")
    st.image(image_grid([query_image], 1, 1))

    # Display scores
    st.subheader(f"Top {top_k_value} Images Retrieved:")
    st.image(image_grid(retrieved_examples["image"], 1, top_k_value))

    # Save query image to run folder
    query_image.save(os.path.join(run_folder, "query_image.jpg"), format="JPEG", quality=95)

    # Save top-k images to run folder
    topk_directory = os.path.join(run_folder, "topk_image")
    os.makedirs(topk_directory, exist_ok=True)

    for j, img in enumerate(retrieved_examples["image"]):
        img.save(os.path.join(topk_directory, f"retrieved_image_{j + 1}.jpg"), format="JPEG", quality=95)
