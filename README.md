# Convert to ONNX (Docker)

## Overview

This project provides a Streamlit application that converts Hugging Face models to ONNX (Open Neural Network Exchange) format, enabling broader model compatibility and deployment options. The application streamlines the process of downloading, converting, and uploading models to Hugging Face.

## Features

- **One-Click Model Conversion**: Convert Hugging Face models to ONNX format with minimal configuration
- **User-Friendly Interface**: Intuitive Streamlit-based web interface
- **Quantization Support**: Automatic model quantization for reduced size and faster inference

## Prerequisites

- Docker and Docker Compose
- Hugging Face account and API token

## Installation

1. Create a `.streamlit` directory.

2. Inside of the `.streamlit` directory, create a `secrets.toml` file.

3. Inside of the `secrets.toml` file, add a line with your Hugging Face write access API token: `HF_TOKEN="yourToken"`.

4. Run the [docker-compose.yml](https://github.com/doppeltilde/convert_to_onnx/blob/main/docker-compose.yml).

   ```yml
   services:
      convert_to_onnx:
         image: ghcr.io/doppeltilde/convert_to_onnx:latest
         ports:
            - "8501:8501"
         volumes:
            - ./models:/root/.cache/huggingface/hub:rw
         environment:
            - HF_TOKEN
         restart: unless-stopped
   ```

## Usage

1. Access the web interface at `http://localhost:8501`

2. Enter a Hugging Face model ID (e.g., `EleutherAI/pythia-14m`)

3. Click "Proceed" to start the conversion

The converted model will be available in your Hugging Face account as `{username}/{model-name}-ONNX`.
