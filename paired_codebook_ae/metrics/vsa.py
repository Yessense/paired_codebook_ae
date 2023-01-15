import torch
from tqdm import tqdm

from ..codebook.vsa import unbind, sim


def vsa_decoding_accuracy(placeholders, codebook, latent_vectors, labels,
                          device=torch.device("cuda:0")):
    accuracy = 0
    accuracies = [0] * 5
    codebook = [codebook[i].to(device) for i in range(len(codebook))]

    for latent_vector, labels in tqdm(zip(latent_vectors, labels)):
        for i, placeholder in enumerate(placeholders):
            unbinded_vector = unbind(placeholder, latent_vector).to(device)

            similarities = []
            for j, codebook_feature in enumerate(codebook[i]):
                similarities.append(sim(unbinded_vector, codebook_feature))

            similarities = torch.stack(similarities, dim=0)
            max_pos = torch.argmax(similarities, dim=0)
            sucess_unbind = max_pos == labels[i]
            accuracy += sucess_unbind
            accuracies[i] += sucess_unbind

    accuracy = accuracy.float() / (5 * len(labels))
    accuracies = [i.float() / len(labels) for i in accuracies]

    print(accuracy)
    print(accuracies)

    return {'Codebook/Accuracy': accuracy, 'Codebook/Accuracies': accuracies}
