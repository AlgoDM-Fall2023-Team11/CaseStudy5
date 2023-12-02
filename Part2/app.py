# import streamlit as st
# import pickle
# import matplotlib.pyplot as plt
# from llama_index.schema import ImageDocument
# from PIL import Image
# import torch
# import clip
# import qdrant_client
# from llama_index import (
#     ServiceContext,
#     SimpleDirectoryReader,
# )
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_index import VectorStoreIndex, StorageContext
# from llama_index.llms import OpenAI
# from llama_index.vector_stores import VectorStoreQuery

# # Load the saved image index and related objects
# with open("image_index.pickle", "rb") as pickle_file:
#     image_index = pickle.load(pickle_file)

# with open("text_query_engine.pickle", "rb") as pickle_file:
#     text_query_engine = pickle.load(pickle_file)

# image_vector_store = image_index.storage_context.vector_store

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)


# def retrieve_results_from_image_index(query, similarity_threshold):
#     """ take a text query as input and return the most similar image from the vector store """

#     # first tokenize the text query and convert it to a tensor
#     text = clip.tokenize(query).to(device)

#     # encode the text tensor using the CLIP model to produce a query embedding
#     query_embedding = model.encode_text(text).tolist()[0]

#     # create a VectorStoreQuery
#     image_vector_store_query = VectorStoreQuery(
#         query_embedding=query_embedding,
#         similarity_top_k=1, # returns 1 image
#         mode="default",
#     )

#     # execute the query against the image vector store
#     image_retrieval_results = image_vector_store.query(
#         image_vector_store_query
#     )
#     return image_retrieval_results

# def plot_image_retrieve_results(image_retrieval_results):
#     """ take a list of image retrieval results and create a new figure"""

#     plt.figure(figsize=(16, 5))

#     img_cnt = 0

#     # iterate over the image retrieval results, and for each result, display the corresponding image and its score in a subplot.
#     # The title of the subplot is the score of the image, formatted to four decimal places.

#     for returned_image, score in zip(
#         image_retrieval_results.nodes, image_retrieval_results.similarities
#     ):
#         img_name = returned_image.text
#         img_path = returned_image.metadata["filepath"]
#         image = Image.open(img_path).convert("RGB")

#         plt.subplot(2, 3, img_cnt + 1)
#         plt.title("{:.4f}".format(score))

#         plt.imshow(image)
#         plt.xticks([])
#         plt.yticks([])
#         img_cnt += 1

#     plt.figure(figsize=(16, 5))

#     img_cnt = 0

#     # iterate over the image retrieval results, and for each result, display the corresponding image and its score in a subplot.
#     # The title of the subplot is the score of the image, formatted to four decimal places.

#     for returned_image, score in zip(
#         image_retrieval_results.nodes, image_retrieval_results.similarities
#     ):
#         img_name = returned_image.text
#         img_path = returned_image.metadata["filepath"]
#         image = Image.open(img_path).convert("RGB")

#         plt.subplot(2, 3, img_cnt + 1)
#         plt.title("{:.4f}".format(score))

#         plt.imshow(image)
#         plt.xticks([])
#         plt.yticks([])
#         img_cnt += 1

# def image_query(query, similarity_threshold):
#     image_retrieval_results = retrieve_results_from_image_index(query, similarity_threshold)
#     plot_image_retrieve_results(image_retrieval_results)

# def main():
#     st.title("Multi-Modal Retrieval App")

#     # User input for query
#     user_query = st.text_input("Enter your query:")

#     # User input for similarity threshold (slider)
#     similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.8, step=0.01)

#     if st.button("Search"):
#         # Perform image and text retrieval
#         image_query(user_query, similarity_threshold)
#         text_retrieval_results = text_query_engine.query(user_query)

#         # Display text retrieval results
#         st.write("Text retrieval results:")
#         for result in text_retrieval_results:
#             st.write(result.text)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import torch

# # Load the saved PyTorch model
# text_query_engine = torch.load('text_model.pth')

# # Streamlit app
# st.title("Text Query App")

# # Get user input
# user_query = st.text_input("Enter your query:")

# # Perform inference using the loaded model
# if st.button("Get Results"):
#     # Use the text_query_engine for inference
#     text_retrieval_results = text_query_engine.query(user_query)
    
#     # Display the results in Streamlit
#     st.write("Text retrieval results:", text_retrieval_results)


import streamlit as st
from serpapi import GoogleSearch
import openai

# Set up OpenAI API key
openai.api_key = "12345"  # Replace with your actual OpenAI API key

# Streamlit app
st.title("Query Exploration App")

# User input for query
user_query = st.text_input("Enter your query:")

# Function to generate explanation and answer using GPT-3
def generate_explanation_and_answer(query):
    prompt = f"Explain and answer the following question:\n{query}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150
    )
    explanation_and_answer = response['choices'][0]['text']
    return explanation_and_answer

# Function to search for similar images using SerpApi
def search_similar_images(query):
    serpapi_key = "12345"  # Replace with your actual SerpApi key
    params = {
        "q": query,
        "engine": "google_images",
        "api_key": serpapi_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    images_results = results.get("images_results", [])
    return images_results

# Main logic
if user_query:
    # Use GPT-3 to generate explanation and answer
    explanation_and_answer = generate_explanation_and_answer(user_query)

    st.subheader("Explanation and Answer:")
    st.write(explanation_and_answer)

    # Image search using SerpApi
    similar_images = search_similar_images(user_query)

    if similar_images:
        st.subheader("Similar Images:")
        for i, image_data in enumerate(similar_images[:5]):  # Limit to the first 5 images
            image_url = image_data["original"]
            st.image(image_url, caption=f"Image {i + 1}", use_column_width=True)
    else:
        st.warning("No similar images found.")



import streamlit as st
from serpapi import GoogleSearch
import openai
import requests
import base64

# Set up OpenAI API key
openai.api_key = "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"  # Replace with your actual OpenAI API key

# Function to generate explanation and answer using GPT-3
def generate_explanation_and_answer(query):
    prompt = f"Explain and answer the following question:\n{query}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150
    )
    explanation_and_answer = response['choices'][0]['text']
    return explanation_and_answer

# Function to search for similar images using SerpApi
def search_similar_images(query):
    serpapi_key = "f14f0e548e9000b9ea664ad352f9300fee0c0b44a5d6743ad5d6e86b0971e1b4"  # Replace with your actual SerpApi key
    params = {
        "q": query,
        "engine": "google_images",
        "api_key": serpapi_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    images_results = results.get("images_results", [])
    return images_results

# Function to upload image to ImgBB and get its URL
def upload_image_to_imgbb(image):
    imgbb_key = "d42d89d45fa141ad280928242551c829"  # Replace with your actual ImgBB API key

    # Create the file payload
    files = {"image": image}

    # ImgBB API endpoint
    endpoint = "https://api.imgbb.com/1/upload"

    # ImgBB API key parameter
    params = {"key": imgbb_key}

    # Send POST request to ImgBB
    response = requests.post(endpoint, params=params, files=files)

    if response.status_code == 200:
        data = response.json()
        image_url = data["data"]["url"]
        return image_url
    else:
        st.warning(f"Image upload failed with status code {response.status_code}.")
        st.warning(response.text)

    return None


# Page 1: Text Query Page
def text_query_page():
    st.title("Text Query Page")
    
    # User input for text query
    user_query = st.text_input("Enter your text query:")

    if st.button("Generate Explanation and Answer"):
        # Use GPT-3 to generate explanation and answer
        explanation_and_answer = generate_explanation_and_answer(user_query)

        st.subheader("Explanation and Answer:")
        st.write(explanation_and_answer)

        # Image search using SerpApi
        similar_images = search_similar_images(user_query)

        if similar_images:
            st.subheader("Similar Images:")
            for i, image_data in enumerate(similar_images[:5]):  # Limit to the first 5 images
                image_url = image_data["original"]
                st.image(image_url, caption=f"Image {i + 1}", use_column_width=True)
        else:
            st.warning("No similar images found.")

# Page 2: Image Query Page
def image_query_page():
    st.title("Image Query Page")
    
    # User input for image query
    uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Upload the image to ImgBB and get its URL
        image_url = upload_image_to_imgbb(uploaded_image)

        # Perform reverse image search using SerpApi
        serpapi_key = "f14f0e548e9000b9ea664ad352f9300fee0c0b44a5d6743ad5d6e86b0971e1b4"  # Replace with your actual SerpApi key
        params = {
            "engine": "google_reverse_image",
            "image_url": image_url,
            "api_key": serpapi_key
        }

        # Simulate reverse image search using the ImgBB image URL
        search = GoogleSearch(params)
        results = search.get_dict()
        inline_images = results.get("inline_images", [])

        if inline_images:
            st.subheader("Similar Images:")
            for i, image_data in enumerate(inline_images[:5]):  # Limit to the first 5 images
                image_url = image_data["original"]
                st.image(image_url, caption=f"Similar Image {i + 1}", use_column_width=True)
        else:
            st.warning("No similar images found.")

# Main App
def main():
    st.title("Multi-Page App")

    # Sidebar to select pages
    page = st.sidebar.selectbox("Select Page", ["Text Query", "Image Query"])

    # Display selected page
    if page == "Text Query":
        text_query_page()
    elif page == "Image Query":
        image_query_page()

if __name__ == "__main__":
    main()


