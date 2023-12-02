import streamlit as st
import requests

# Nyckel API endpoint
nyckel_url = 'https://www.nyckel.com/v0.9/functions/3904x22lwf9xm6hx/search'

# Nyckel API access token
nyckel_access_token = '0qfu87wfpiz00u2wrqnpweeazfwkp9exfesnnc3h7y2yyh375gban43lb5utabqg'  # Replace with your actual Nyckel access token

# Streamlit app
st.title("Image-to-Image Search App")

# User input for image
uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Perform image-to-image search using Nyckel API
    headers = {'Authorization': 'Bearer ' + nyckel_access_token}

    # Prepare image data
    files = {'data': uploaded_image.getvalue()}

    try:
        # Make the API request
        response = requests.post(nyckel_url, headers=headers, files=files)

        # Print response content and status code for debugging
        print("Response Content:", response.text)
        print("Status Code:", response.status_code)

        # Check the response
        response.raise_for_status()

        if response.status_code == 200:
            result = response.json()
            st.subheader("Similar Images:")
            for i, similar_image_data in enumerate(result.get("similar_samples", [])):
                similar_image_url = similar_image_data.get("image_url")
                st.image(similar_image_url, caption=f"Similar Image {i + 1}", use_column_width=True)
        else:
            st.warning(f"Image search failed with status code {response.status_code}.")
    except requests.exceptions.HTTPError as errh:
        st.warning(f"HTTP Error: {errh}")
        st.warning(response.text)
    except requests.exceptions.RequestException as err:
        st.warning(f"Request Error: {err}")
