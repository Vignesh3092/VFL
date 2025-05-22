# Federated Split Learning with KMeans Defense
This project demonstrates a federated learning system using Split Learning. In this setup, each client processes part of an input image and sends intermediate embeddings to a central server, which completes the classification task. While split learning improves communication efficiency and preserves data privacy, it remains vulnerable to model poisoning attacks. To address this, we implement a KMeans-based defense mechanism to detect and mitigate poisoned client contributions.
# Overview
We use the MNIST dataset for handwritten digit recognition. Each input image is 28x28 pixels and grayscale. Instead of giving full images to every client, we vertically split each image into 5 equal parts, and assign each part to a different client. Thus, every client only sees a slice of the image (approximately 5–6 pixels in width).
Each client uses a convolutional neural network (CNN) to compute feature embeddings from their input slice. These embeddings are then sent to the server. The server concatenates embeddings from all clients and uses a fully connected network to classify the digit.
This setup is efficient and privacy-friendly, but if one client is malicious, it can disrupt training by sending corrupted embeddings or gradients.
# Attack
To simulate a poisoning attack, we intentionally corrupt one of the clients. This malicious client adds random noise to its gradients during backpropagation. These noisy gradients are then used for updating the client’s model parameters, which in turn negatively affects the global model’s performance.
Before applying any defense, the system becomes unstable due to this attack, resulting in a severe drop in accuracy.
# Defense: KMeans Clustering
To defend against the poisoned client, we apply KMeans clustering on the feature embeddings received from all clients.

Here’s how it works:

We collect the embeddings from all clients for each batch.

We apply KMeans with n_clusters=2 to the set of embeddings.

We assume the majority cluster represents benign clients.

We discard embeddings from the minority cluster, which are considered anomalous or poisoned.

Only embeddings from the trusted cluster are used for server-side classification and loss computation.

This defense technique is simple, does not require labels or prior knowledge of the attack, and effectively removes suspicious client contributions.
#  Results
We trained the model under three different scenarios:

In the clean setting (no attack), the model achieved an accuracy of 97.84%.

When a poisoned client was introduced, the accuracy dropped sharply to 43.46%.

After applying the KMeans defense, the accuracy recovered significantly to 96.10%.

This demonstrates that the KMeans-based defense is effective in filtering out malicious updates and restoring the model’s performance.
# DataSet
We used the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits from 0 to 9. All images are grayscale and of size 28x28 pixels.
# Model Architecture
In this project, we use a Split Learning architecture where:

Client Model: Each client runs a Convolutional Neural Network (CNN). The CNN processes only a vertical slice of the input image (around 5–6 pixels in width) and extracts feature embeddings from it. The client-side CNN consists of two convolutional layers followed by ReLU activation and max pooling.

Central Server Model: The server acts as the central model and uses a Multilayer Perceptron (MLP). It receives the concatenated embeddings from all client models and passes them through a fully connected network (MLP) to perform the final classification. The MLP contains one hidden layer and an output layer with 10 classes (for the MNIST digits 0–9).
