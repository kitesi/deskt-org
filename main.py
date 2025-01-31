import argparse
import mimetypes
import os
import re
from collections import defaultdict

import PyPDF2
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# Initialize a text summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_cluster_content(file_texts):
    """Summarize the content of a cluster to generate a descriptive name."""
    combined_text = " ".join(file_texts)
    if len(combined_text) > 1024:  # Summarization limit for some models
        combined_text = combined_text[:1024]
    summary = summarizer(combined_text, max_length=10, min_length=5, do_sample=False)[0]['summary_text']
    return summary

def cluster_files(file_paths, n_clusters):
    features, vectorizer = extract_text_features(file_paths)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(features)

    cluster_files_map = defaultdict(list)
    for file, label in zip(file_paths, labels):
        cluster_files_map[label].append(file)

    cluster_names = {}
    for label, files in cluster_files_map.items():
        file_texts = [extract_text_from_pdf(file) if file.endswith('.pdf') else open(file).read() for file in files]
        cluster_name = summarize_cluster_content(file_texts)
        cluster_names[label] = re.sub(r"[^\w\s-]", "", cluster_name) or f"Cluster_{label}"

    return labels, cluster_names


def extract_text_from_pdf(pdf_path: str):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_text_features(file_paths):
    texts: list[str] = []

    for f in file_paths:
        if f.endswith('.pdf'):
            texts.append(extract_text_from_pdf(f))
        else:
            with open(f, 'r') as file:
                texts.append(file.read())

    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(texts).toarray(), vectorizer

def pre_process_file(files: list[str], directory: str):
    large_files = []
    text_readable_files = []
    image_files = []
    skipped_files = []

    for file in files:
        mtype = mimetypes.guess_type(file)[0]

        if os.path.getsize(file) > 10e6:
            large_files.append(file)
        elif mtype is None:
            skipped_files.append(file)
        elif mtype.startswith('text') or file.endswith('.pdf'):
            text_readable_files.append(file)
        elif mtype.startswith('image'):
            image_files.append(file)
        else:
            skipped_files.append(file)

    large_files_d = os.path.join(directory, 'large_files')

    if not os.path.exists(large_files_d):
        os.makedirs(large_files_d)

    for file in large_files:
        os.rename(file, os.path.join(large_files_d, os.path.basename(file)))

    return text_readable_files, image_files, large_files, skipped_files

def main(): 
    parser = argparse.ArgumentParser(
        prog="organize",
        description="Organize files into clusters",
    )

    parser.add_argument(
        "directory", type=str, default="./", help="Input directory"
    )

    parser.add_argument(
        "-k", "--n_clusters", type=int, default=3, help="Number of clusters"
    )

    args = parser.parse_args()
    n_clusters: int = args.n_clusters
    input_dir: str = args.directory
    
    files: list[str] = []
    
    for f in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, f)):
            files.append(os.path.join(input_dir, f))

    text_readable_files, image_files, large_files, skipped_files = pre_process_file(files, input_dir)

    if len(large_files) > 0:
        print(f'Moved {len(large_files)} large files to large_files directory')

    print(f'Found {len(text_readable_files)} text readable files')
    print(f'Found {len(image_files)} image files')
    print(f'Skipped {len(skipped_files)} files')

    if len(text_readable_files) < n_clusters:
        print("Not enough files to create the specified number of clusters.")
        return

    labels, cluster_names = cluster_files(text_readable_files, n_clusters)

    cluster_moves = {}
    
    for file, label in zip(text_readable_files, labels):
        cluster_label = cluster_names[label]
        if cluster_label not in cluster_moves:
            cluster_moves[cluster_label] = []
        cluster_moves[cluster_label].append(file)

    for cluster_label, files in cluster_moves.items():
        print(f'/{cluster_label}')
        for file in files:
            print(f'   - {file} -> {cluster_label}')

    for label, name in cluster_names.items():
        print(f'Cluster {label} -> {name}')


if __name__ == '__main__':
    main()
