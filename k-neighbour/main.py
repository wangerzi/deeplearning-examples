import pandas as pd
import matplotlib.pyplot as plt
import random
import math


def show_image(file):
    data = pd.read_csv(file)

    x = data.iloc[0:150, [0, 2]].values
    plt.scatter(x[0:50, 0], x[0:50, 1], color="blue", marker="x", label="setosa")
    plt.scatter(x[50:100, 0], x[50:100, 1], color="red", marker="o", label="versicolor")
    plt.scatter(x[100:150, 0], x[100:150, 1], color="green", marker="*", label="virginica")
    plt.xlabel('sepal_length')
    plt.ylabel('petal_width')
    plt.show()


# xalis is k, y is rate
def show_result_image(result):
    list_k = [row[0] for row in result]
    max_k = max(list_k) + 1

    plt.axis([0, max_k, 0, 1.1])
    plt.plot(
        list_k,
        [row[1] for row in result],
        'r-',
        label="vote by num"
    )
    plt.plot(
        list_k,
        [row[2] for row in result],
        'r--',
        label="vote by weight"
    )
    plt.plot(
        list_k,
        [row[3] for row in result],
        'b-',
        label="vote by num(nor)"
    )
    plt.plot(
        list_k,
        [row[4] for row in result],
        'b--',
        label="vote by weight(nor)"
    )
    plt.legend()
    plt.show()


def load_dataset(file, split, normalization=True):
    data = pd.read_csv(file).values
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


def get_distance(instance1, instance2, num):
    return sum([(instance1[i] - instance2[i]) ** 2 for i in range(num)]) ** 0.5


def get_neighbors(tran_data, instance, k):
    # calculate all the distance
    distance = []
    for train_instance in tran_data:
        # the last data is label
        distance.append([train_instance, get_distance(train_instance, instance, len(instance) - 1)])

    # sort by distance
    distance.sort(key=lambda v: v[1])

    # get k neighbors, 0 is instance, 1 is distance
    return distance[0:k]


def vote_label_by_num(neighbors):
    vote_map = {}
    for row in neighbors:
        label = row[0][-1]
        if label in vote_map:
            vote_map[label] += 1
        else:
            vote_map[label] = 1
    sorted_map = sorted(vote_map.items(), key=lambda v: v[1], reverse=True)
    return sorted_map[0][0]


def vote_label_by_weight(neighbors):
    vote_map = {}
    list_distance = [row[1] for row in neighbors]
    total_distance = sum(list_distance)
    rate = 1
    for i in range(len(neighbors)):
        neighbors[i].append(-neighbors[i][1] / (total_distance + 0.000001) * rate)
    for row in neighbors:
        label = row[0][-1]
        format_distance = row[-1]
        if label in vote_map:
            vote_map[label] += format_distance
        else:
            vote_map[label] = format_distance
    sorted_map = sorted(vote_map.items(), key=lambda v: v[1], reverse=True)
    return sorted_map[0][0]


def predict_data(tran, test, vote, k=3):
    correct = 0
    for instance in test:
        neighbors = get_neighbors(tran, instance, k)
        cate = vote(neighbors)
        if cate == instance[-1]:
            correct += 1

    return round(correct / float(len(test)), 4)


def main():
    tran_data, test_data = load_dataset('iris.data', 0.7, False)
    nor_tran_data, nor_test_data = load_dataset('iris.data', 0.7)

    result_data = []
    for k in range(1, 20):
        rate_num = predict_data(tran_data, test_data, vote_label_by_num, k)
        rate_weight = predict_data(tran_data, test_data, vote_label_by_weight, k)
        nor_rate_num = predict_data(nor_tran_data, nor_test_data, vote_label_by_num, k)
        nor_rate_weight = predict_data(nor_tran_data, nor_test_data, vote_label_by_weight, k)

        print("tran data: %d, test data: %d, k: %d, vote by num: %.3f:%.3f, vote by weight: %.3f:%.3f" % (
            len(tran_data), len(test_data), k, rate_num, nor_rate_num, rate_weight, nor_rate_weight
        ))
        result_data.append([k, rate_num, rate_weight, nor_rate_num, nor_rate_weight])

    show_result_image(result_data)


if __name__ == '__main__':
    main()
