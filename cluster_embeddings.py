#!/usr/bin/env python3

"""Take BERT-like model embeddings and cluster them using k-means and replace the embeddings with the cluster centroids."""

import argparse
import logging

import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="BERT-like model name")
    parser.add_argument("cluster_num", help="Number of clusters to use", type=int)
    parser.add_argument("output", help="Output directory for the new model")
    args = parser.parse_args()

    logging.info("Loading model %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    embeddings = model.get_input_embeddings().weight.detach().numpy()

    logging.info("Clustering embeddings")
    kmeans = KMeans(n_clusters=args.cluster_num, random_state=0, verbose=1, n_init='auto').fit(embeddings)

    logging.info("Replacing embeddings with cluster centroids")
    model.get_input_embeddings().weight.data = torch.tensor(kmeans.cluster_centers_)

    logging.info("Saving model to %s", args.output)
    model.config.vocab_size = args.cluster_num
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    logging.info("Save cluster mapping to %s", args.output + "/cluster_mapping.txt")
    with open(args.output + "/cluster_mapping.txt", "w") as f:
        for cluster in enumerate(kmeans.labels_):
            print(cluster[1], file=f)

    logging.info("Save readable cluster mapping to %s", args.output + "/cluster_mapping_readable.txt")
    clusters = [[] for _ in range(args.cluster_num)]
    for i, cluster in enumerate(kmeans.labels_):
        clusters[cluster].append(tokenizer.convert_ids_to_tokens(i))
    with open(args.output + "/cluster_mapping_readable.txt", "w") as f:
        for cluster in clusters:
            print(", ".join(cluster), file=f)

    logging.info("Done.")

if __name__ == "__main__":
    main()
