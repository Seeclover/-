import os
import torch
import data

if __name__ == "__main__":
    data_path = 'data/penn'
    corpus = data.Corpus(data_path)
    torch.save(corpus, os.path.join(data_path, 'corpus.pth'))
    print("Data preparation complete.")
