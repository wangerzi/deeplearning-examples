from perceptrons import multiple
import random
import math
import pandas as pd


def load_dataset(file, split, normalization=True):
    data = pd.read_csv(file, sep="\s+").values.tolist()
    random.shuffle(data)

    if normalization:
        for i in range(len(data[0]) - 1):
            col = [row[i] for row in data]
            col_min = min(col)
            col_max = max(col)
            for j in range(len(data)):
                data[j][i] = (data[j][i] - col_min) / (col_max - col_min)
    train_num = math.ceil(len(data) * split)
    return data[0:train_num], data[train_num:]


def format_dataset(data):
    for row in data:
        index = row[-1] - 1
        row[-1] = [0, 0, 0]
        row[-1][int(index)] = 1


def main():
    train_data, test_data = load_dataset('seeds_dataset.txt', 1)
    format_dataset(train_data)
    format_dataset(test_data)
    multi = multiple.PerceptronsMultiple(7, 8, 3, multiple.sigmod_activate, multiple.sigmod_activate_inverse)
    multiple.show_time_graph(multi, train_data, train_data, 0.3, 10, 2000)


if __name__ == '__main__':
    main()
