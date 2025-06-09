# AI_Hackathon-Flickd
# Fashion Product Detection and Similarity Search Pipeline

This repository contains the complete pipeline for detecting fashion products from images, generating natural language prompts, fine-tuning the CLIP model on these crops, and building a FAISS similarity index for efficient product retrieval.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Setup Instructions](#setup-instructions)
- [Step 1: Object Detection and Cropping](#step-1-object-detection-and-cropping)
- [Step 2: Dataset Preparation for CLIP Fine-tuning](#step-2-dataset-preparation-for-clip-fine-tuning)
- [Step 3: Fine-tuning CLIP](#step-3-fine-tuning-clip)
- [Step 4: Embedding Generation and FAISS Indexing](#step-4-embedding-generation-and-faiss-indexing)


---

## Project Overview

The goal of this project is to build a robust fashion product detection and retrieval system. The system detects clothing items from images, crops these items, and generates descriptive text prompts. Using these crops and prompts, the CLIP model is fine-tuned to better understand the fashion domain and associate images with textual descriptions. Finally, image embeddings are indexed using FAISS for efficient similarity search, enabling fast retrieval of similar fashion products.

---

## Features

- **Object Detection:** Uses the YOLOS Fashionpedia pre-trained model to detect fashion items in images.
- **Crop Extraction:** Extracts and saves image crops of detected items with bounding box coordinates.
- **Prompt Generation:** Generates natural language prompts for each detected item using predefined templates.
- **CLIP Fine-tuning:** Fine-tunes OpenAI’s CLIP model on the cropped images and prompts to adapt it for the fashion domain.
- **Embedding & Indexing:** Generates normalized image embeddings and builds a FAISS index for fast similarity search.
- **Fault Tolerant:** Handles missing or corrupt images gracefully during training and inference.

---

## Dataset

- **Source:** Original dataset consists of fashion product images along with metadata including image URLs and unique IDs.
- **Detection Outputs:** Crops saved locally with labels and associated prompts.
- **CLIP Training Data:** CSV file containing image paths, labels, and textual prompts for fine-tuning.
- **Color Dataset:** An additional dataset of 835 colors with RGB values was used to assign color labels to crops via KNN (not included in this repo but can be integrated).

---

## Dependencies

- Python 3.8+
- PyTorch
- Transformers (Huggingface)
- FAISS
- PIL (Pillow)
- Requests
- TQDM
- Pandas


