import matplotlib.pyplot as plt


def activator(val):
    return 1.0 if val > 0 else 0.0


class PerceptronSingle(object):
    def __init__(self, num, active):
        self.weight = [0.0 for _ in range(num)]
        self.active = active
        self.num = num

    def predict(self, row):
        if len(row) < self.num:
            raise Exception('Predict must have ' + str(self.num) + ' items')
        return self.active(sum([self.weight[i] * row[i] for i in range(self.num)]))

    def train(self, tran_data, times=10, rate=0.1):
        if len(tran_data) < 1 or len(tran_data[0]) < self.num + 1:
            raise Exception('Tran data can\'t empty and must have ' + str(self.num + 1) + ' item each row!')
        for t in range(times):
            for row in tran_data:
                predict = self.predict(row)
                self.__update_weights(row, row[-1], predict, rate)

    def validate(self, test_data):
        correct = 0
        for row in test_data:
            if self.predict(row) == row[-1]:
                correct += 1
        return round(correct / float(len(test_data)), 4)

    def __update_weights(self, row, label, predict, rate):
        delta = label - predict
        for i in range(self.num):
            self.weight[i] += rate * delta * row[i]


# data: [ [[x1, x2], [y1, y1]], 'marker(like r-)', 'line label' ]
def show_line_graph(data, x_label="x", y_label="rate"):
    for row in data:
        x_row = [v[0] for v in row[0]]
        y_row = [v[1] for v in row[0]]
        plt.plot(x_row, y_row, row[1], label=row[2])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def show_times_graph(and_tran_data, or_tran_data, not_tran_data):
    single = PerceptronSingle(3, activator)
    and_times_rate = []
    or_times_rate = []
    not_times_rate = []
    for times in range(1, 12):
        single.train(and_tran_data, times, 0.1)
        rate = single.validate(and_tran_data)
        and_times_rate.append([times, rate])

        single.train(or_tran_data, times, 0.1)
        rate = single.validate(or_tran_data)
        or_times_rate.append([times, rate])

        single.train(not_tran_data, times, 0.1)
        rate = single.validate(not_tran_data)
        not_times_rate.append([times, rate])
    show_line_graph([
        [and_times_rate, 'r-', 'and'],
        [or_times_rate, 'b-', 'or'],
        [not_times_rate, 'g-', 'not'],
    ], 'times', 'correct rate')


def show_rate_graph(and_tran_data, or_tran_data, not_tran_data):
    single = PerceptronSingle(3, activator)
    and_times_rate = []
    or_times_rate = []
    not_times_rate = []
    times = 4
    study_rate = 0
    while study_rate < 3:
        single.train(and_tran_data, times, study_rate)
        rate = single.validate(and_tran_data)
        and_times_rate.append([study_rate, rate])

        single.train(or_tran_data, times, rate)
        rate = single.validate(or_tran_data)
        or_times_rate.append([study_rate, rate])

        single.train(not_tran_data, times, rate)
        rate = single.validate(not_tran_data)
        not_times_rate.append([study_rate, rate])

        study_rate += 0.01
    show_line_graph([
        [and_times_rate, 'r-', 'and'],
        [or_times_rate, 'b-', 'or'],
        [not_times_rate, 'g-', 'not'],
    ], 'rate', 'correct rate')


def main():
    and_tran_data = [
        [-1, 1, 1, 1],
        [-1, 1, 0, 0],
        [-1, 0, 1, 0],
        [-1, 0, 0, 0],
    ]
    or_tran_data = [
        [-1, 1, 1, 1],
        [-1, 1, 0, 1],
        [-1, 0, 1, 1],
        [-1, 0, 0, 0],
    ]
    not_tran_data = [
        [-1, -1, 1, 0],
        [-1, -1, 0, 1],
    ]

    # show_times_graph(and_tran_data, or_tran_data, not_tran_data)
    show_rate_graph(and_tran_data, or_tran_data, not_tran_data)
    

if __name__ == '__main__':
    main()
