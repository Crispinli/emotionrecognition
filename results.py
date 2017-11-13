import numpy as np
from collections import Counter
import os

label_index = {
    '0': "angry",
    '1': "anxious",
    '2': "disgust",
    '3': "happy",
    '4': "neutral",
    '5': "sad",
    '6': "surprise",
    '7': "worried"
}

result_of_cnn = np.load("./npy_data/results_of_cnn.npy").tolist()

for key in result_of_cnn:
    results = result_of_cnn[key]
    counter = Counter(results)
    classification = counter.most_common(1)
    result_of_cnn[key] = label_index[str(classification[0][0])]

np.save('./total_results/cnn_results.npy', result_of_cnn)
