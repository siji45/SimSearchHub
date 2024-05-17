import streamlit as st
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
uploaded_file = st.sidebar.file_uploader("Upload a custom query image:", type=["jpg", "jpeg", "png"])

if st.sidebar.button("Run Algorithm"):
    if uploaded_file is not None:
        # Read the uploaded image
        query_image = Image.open(uploaded_file)
        resized_query_image = query_image.resize((224, 224))  # Resize the query image

        # Get top k similar images
        retrieved_examples = get_neighbors(resized_query_image, top_k=top_k_value)

        # Prepare images for display
        images = [query_image]
        images.extend(retrieved_examples["image"])

        # Display the query image
        st.image(query_image, caption="Query Image", use_column_width=True)

        # Display the top k similar images
        st.image(image_grid(images, 1, len(images)), caption="Top {} Similar Images".format(top_k_value), use_column_width=True)
