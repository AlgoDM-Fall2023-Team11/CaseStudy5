# Algorithmic Digital Marketing - Image Annotation and Multimodal Retrieval
Codelabs Link: https://codelabs-preview.appspot.com/?file_id=1U1kp4HCopbhkkSJZEohzWwusqqJOJZjN6zjAP7eIWE4#2

## Team Members
- Krishna Barfiwala (NUID: 002771997)
- Nakul Shiledar (NUID: 002738981)
- Usashi Roy (NUID: 002752872)

## Part 1: Image Annotation with OpenAI GPT-4 Vision API

### Overview
The OpenAI GPT-4 Vision API is employed for automating the annotation of fashion product images. The goal is to tag product images with detailed information, enhancing the understanding of clothing items.

### Steps

1. **Connecting to Amazon S3:**
   - Utilize the `boto3` library to connect to the 'adm-fashion-images' Amazon S3 bucket.
   - Retrieve a list of images from the S3 bucket.

2. **Defining Functions:**
   - `form_and_table_understanding(image_path, prompt_text, key):`
     - Interacts with the OpenAI GPT-4 Vision API.
     - Takes a base64-encoded image, a prompt text, and an image key.
     - Sends a request to the OpenAI API and returns the response along with the key.
   - `encode_image(image_path):`
     - Encodes an image in base64 format.
     - Takes an image path, reads the image in binary mode, encodes it, and returns the encoded string.

3. **Main Loop:**
   - Iterates over the list of images obtained from the S3 bucket.

4. **Downloading Images:**
   - For each image, download it from the S3 bucket using the image key.

5. **Base64 Encoding:**
   - Encode each image in base64 format for use in the OpenAI API call.

6. **Prompt Text:**
   - Define a prompt text providing instructions for the GPT-4 Vision API to generate labels for the image.

7. **Calling OpenAI API:**
   - Call the `form_and_table_understanding` function with the base64-encoded image, prompt text, and image key.
   - The function interacts with the OpenAI GPT-4 Vision API, generating annotations for the image.

8. **Saving Responses to JSON:**
   - Convert the array of responses to JSON format.
   - Save the JSON data to a file named 'data.json'.

9. **Snowflake Database:**
   - Establish a connection to Snowflake, a cloud-based data warehousing platform.
   - Create a table named 'annotations' within the specified database and schema.
   - Read data from a JSON file containing image annotations.
   - Iterate through the JSON data, extracting image IDs and corresponding annotations.
   - Insert this information into the Snowflake table.

10. **Security Considerations:**
    - Note that credentials, such as the password, are directly specified in the script. For security reasons, it is advisable to use secure methods for handling credentials in a production environment.

## Part 2: Multimodal Retrieval Model

### Overview
A multimodal retrieval model is built based on the [2] paper, using CLIP and BiLSTM to find similar images and text descriptions.

### Task 1: Given an Image, Find Similar Images and Text Tags

1. **Extract Embeddings using CLIP:**
   - Utilize CLIP, such as ViT-B/32, to extract embeddings for the given image.

2. **Retrieve Similar Images:**
   - Compare image features for similarity.
   - Calculate similarity scores between the given image and others.
   - Sort images based on similarity scores.

3. **Retrieve Text Tags for Similar Images:**
   - If available, retrieve text tags for identified similar images.

4. **Present Results:**
   - Present a list of similar images with their similarity scores.
   - Display associated text tags for these similar images.

### Task 2: Given a Text String, Find Matching Images

1. **Download Texts and Images Raw Files:**
   - Obtain raw text data from Wikipedia articles.
   - Download relevant images associated with the Wikipedia articles.

2. **Text Embeddings with BAAI/bge-base-en-v1.5:**
   - Use the BAAI/bge-base-en-v1.5 model for text embeddings.

3. **Image Embeddings with OpenAI CLIP (ViT-B/32):**
   - Utilize OpenAI CLIP's ViT-B/32 model for image embeddings.

4. **Storage in Qdrant:**
   - Store text and image embeddings as separate collections in Qdrant.

5. **Querying for Text and Images:**
   - Convert the query text into embeddings using BAAI/bge-base-en-v1.5.
   - Simultaneously, convert the query image into embeddings using ViT-B/32.
   - Query text and image embeddings collections in Qdrant to retrieve relevant information.

6. **Multi-Modal Retrieval:**
   - Combine results from text and image queries to provide a multi-modal response.
   - Rank results based on similarity scores.

7. **Text Response Synthesizing with gpt-3.5-turbo:**
   - Optionally, use gpt-3.5-turbo to synthesize a coherent text response based on retrieved information.

8. **Evaluation and Optimization:**
   - Evaluate system performance using metrics such as precision, recall, and F1 score.
   - Fine-tune model parameters, embeddings, and retrieval strategies based on user feedback.

## Part 3: Streamlit App

### Overview
Create a Streamlit app providing a user-friendly interface for the multimodal retrieval model.

### Features
- Upload an image and retrieve similar images from the database.
- Enter a text description and retrieve matching images.

### Purpose
The Streamlit app makes it easy for users to explore the capabilities of the multimodal retrieval model, allowing them to interact with and derive insights from their fashion product images.

## Conclusion
This comprehensive solution integrates image annotation, multimodal retrieval, and a user-friendly interface through a Streamlit app. The combination of OpenAI GPT-4 Vision API, CLIP, and other advanced models enhances the capabilities for image understanding and retrieval in the domain of algorithmic digital marketing.
