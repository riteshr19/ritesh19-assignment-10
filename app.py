from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for
import pandas as pd
import numpy as np
import os
from PIL import Image as PILImage, UnidentifiedImageError
import torch
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer

app = Flask(__name__)

# Load the precomputed embeddings
df = pd.read_pickle('image_embeddings.pickle')

# Load the model and tokenizer
model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = get_tokenizer('ViT-B-32')
model.eval()

# Define the image folder path
image_folder = "coco_images_resized"  # Replace with your folder path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image-to-image', methods=['POST'])
def image_to_image():
    file = request.files.get('image')
    if not file:
        return "No file uploaded", 400

    try:
        image = preprocess(PILImage.open(file).convert("RGB")).unsqueeze(0)
    except UnidentifiedImageError:
        return "Invalid image file", 400

    query_embedding = F.normalize(model.encode_image(image), p=2, dim=-1).detach().cpu().numpy()
    df["cosine_similarity"] = df["embedding"].apply(lambda emb: np.dot(query_embedding.squeeze(), emb) / (np.linalg.norm(query_embedding.squeeze()) * np.linalg.norm(emb)))
    top_5 = df.nlargest(5, "cosine_similarity")
    results = top_5[["file_name", "cosine_similarity"]].to_dict(orient="records")
    
    return render_template('results.html', results=results)

@app.route('/text-to-image', methods=['POST'])
def text_to_image():
    text_query = request.form.get('text')
    if not text_query:
        return "No text query provided", 400

    text = tokenizer([text_query])
    query_embedding = F.normalize(model.encode_text(text), p=2, dim=-1).detach().cpu().numpy()
    df["cosine_similarity"] = df["embedding"].apply(lambda emb: np.dot(query_embedding.squeeze(), emb) / (np.linalg.norm(query_embedding.squeeze()) * np.linalg.norm(emb)))
    top_5 = df.nlargest(5, "cosine_similarity")
    results = top_5[["file_name", "cosine_similarity"]].to_dict(orient="records")
    
    return render_template('results.html', results=results)

@app.route('/hybrid-search', methods=['POST'])
def hybrid_search():
    text_query = request.form.get('text')
    file = request.files.get('image')
    lam = float(request.form.get('lambda', 0.5))

    if not file:
        return "No file uploaded", 400
    if not text_query:
        return "No text query provided", 400

    try:
        image = preprocess(PILImage.open(file).convert("RGB")).unsqueeze(0)
    except UnidentifiedImageError:
        return "Invalid image file", 400

    image_query = F.normalize(model.encode_image(image), p=2, dim=-1).detach().cpu()
    text = tokenizer([text_query])
    text_query = F.normalize(model.encode_text(text), p=2, dim=-1).detach().cpu()
    query_embedding = F.normalize(lam * text_query + (1.0 - lam) * image_query, p=2, dim=-1).squeeze().numpy()
    df["cosine_similarity"] = df["embedding"].apply(lambda emb: np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb)))
    top_5 = df.nlargest(5, "cosine_similarity")
    results = top_5[["file_name", "cosine_similarity"]].to_dict(orient="records")
    
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)