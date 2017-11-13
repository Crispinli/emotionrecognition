import numpy as np
import os

rnn = np.load('./total_results/rnn_results.npy').tolist()
cnn = np.load('./total_results/cnn_results.npy').tolist()

total_results = {}

if os.path.exists('./total_results/total_results.npy') == False:
    with open(r'C:\Users\lwp\Desktop\test_data\test.txt', 'r') as file:
        for line in file:
            line = line[0: -1]
            if line in rnn:
                total_results[line] = rnn[line]
            elif line in cnn:
                total_results[line] = cnn[line]
            else:
                total_results[line] = "NULL"
    np.save('./total_results/total_results.npy', total_results)
else:
    total_results = np.load('./total_results/total_results.npy').tolist()
    file1 = open(r'C:\Users\lwp\Desktop\test_data\test.txt', 'r')
    file2 = open(r'C:\Users\lwp\Desktop\test_data\total_results.txt', 'a')
    for line in file1:
        line = line[0: -1]
        file2.write(line + " " + total_results[line] + "\n")

    file2.close()
    file1.close()