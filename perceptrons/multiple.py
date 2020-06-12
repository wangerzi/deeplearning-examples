import random
import math
from perceptrons.single import show_line_graph


def sigmod_activate(val):
    try:
        return 1.0 / (1.0 + math.exp(-val))
    except OverflowError:
        return float(1000)


def sigmod_activate_inverse(output):
    return output * (1 - output)


def simple_activate(val):
    return 1.0 if val > 0 else 0.0


class PerceptronsMultiple(object):
    def __init__(self, input_num, hidden_num, out_num, active, active_inverse):
        # we have input_num * hidden_num weights for hidden layer
        hidden_layer = [
            {"weights": [random.random() for _ in range(input_num + 1)]} for _ in range(hidden_num)
        ]
        # we have hidden_num * out_num weights for output layer
        out_layer = [
            {"weights": [random.random() for _ in range(hidden_num + 1)]} for _ in range(out_num)
        ]
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.out_num = out_num
        self.active = active
        self.active_inverse = active_inverse
        self.network = [hidden_layer, out_layer]

    def train(self, train_data, times=10, rate=0.1):
        for _ in range(times):
            for row in train_data:
                self.forward(row)
                self.backward(row[-1])
                self.__update_weights(rate)

    def forward(self, row):
        input_data = row
        input_num = self.input_num

        for layer in self.network:
            output_data = []
            for node in layer:
                node['sum'] = sum([input_data[i] * node['weights'][i] for i in range(input_num)])
                node['output'] = self.active(node['sum'] + node['weights'][-1])
                node['input_data'] = input_data
                output_data.append(node['output'])
            input_data = output_data
            input_num = len(input_data)
        return input_data

    # must called forward first
    def backward(self, label):
        # calc the responsibility
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            if i == len(self.network) - 1:
                for node_index in range(len(layer)):
                    node = layer[node_index]
                    node['responsibility'] = self.active_inverse(node['output']) * (label[node_index] - node['output'])
            else:
                next_layer = self.network[i + 1]
                next_layer_len = len(next_layer)
                for node_index in range(len(layer)):
                    node = layer[node_index]
                    node['responsibility'] = self.active_inverse(node['output']) * sum(
                        [node['weights'][i] * next_layer[i]['responsibility'] for i in range(next_layer_len)])
        return self.network

    def __update_weights(self, rate):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            for node in layer:
                for index in range(len(node['weights']) - 1):
                    node['weights'][index] += rate * node['responsibility'] * node['input_data'][index]
                node['weights'][-1] += rate * node['responsibility']


def validate(multi, test_data, debug=False):
    correct = 0
    error = 0
    for row in test_data:
        output = multi.forward(row)
        predict = output.index(max(output))
        label = row[-1].index(max(row[-1]))
        if debug:
            print(row, output, label, predict)

        if predict == label:
            correct += 1
        else:
            error += (label - predict) ** 2

    return round(correct / float(len(test_data)), 4), error


def show_time_graph(multi, train_data, test_data, study_rate=0.5, time_chunk=10, time_total=20000):
    time_current = 0

    rate_result = []
    error_result = []
    while time_current < time_total:
        multi.train(train_data, time_chunk, study_rate)
        rate, error = validate(multi, test_data)

        rate_result.append([time_current, rate])
        error_result.append([time_current, error])
        time_current += time_chunk

    show_line_graph([
        [rate_result, 'r-', "correct rate for study rate %.3f" % study_rate],
        [error_result, 'b--', 'loss'],
    ], 'epoch', 'data')

    rate, error = validate(multi, test_data)
    print('final rate %.4f, error %.4f' % (rate, error))


def main():
    data = [
        [1, 1, [0, 1]],  # 1
        [0, 0, [0, 1]],  # 1
        [1, 0, [1, 0]],  # 0
        [0, 1, [1, 0]],  # 0
    ]
    multi = PerceptronsMultiple(2, 3, 2, sigmod_activate, sigmod_activate_inverse)
    show_time_graph(multi, data, data)


if __name__ == '__main__':
    main()
