import matplotlib.pyplot as plt
import random


def get_dataset(num, start, end, w1=2, w0=3, strength=2.):
    data = []
    for _ in range(0, num):
        x = round(start + random.random() * (end - start), 2)
        y = round(w1 * x + w0, 2) + random.random() * strength
        data.append([x, y])
    return data


def show_dataset(data, x_axis, y_axis, w1, w0):
    x = [row[0] for row in data]
    y = [row[1] for row in data]

    plt.axis(x_axis + y_axis)
    plt.plot(x, y, 'r^')

    line_x = [x_axis[0], x_axis[-1]]
    line_y = [w1 * line_x[0] + w0, w1 * line_x[1] + w0]
    plt.plot(line_x, line_y, 'b-')
    plt.grid()
    plt.show()


def mean(row):
    return sum(row) / float(len(row))


def variance(row):
    m = mean(row)
    return sum([(i - m) ** 2 for i in row])


def covariance(x_row, y_row):
    m_x = mean(x_row)
    m_y = mean(y_row)

    return sum([(x_row[i] - m_x) * (y_row[i] - m_y) for i in range(len(x_row))])


def line_coefficients(data):
    x_row = [row[0] for row in data]
    y_row = [row[1] for row in data]

    w1 = covariance(x_row, y_row) / variance(x_row)
    w0 = mean(y_row) - w1 * mean(x_row)

    return w0, w1


def simple_line_regression(train, test):
    w0, w1 = line_coefficients(train)

    print('regression w1 is %.3f, w0 is %.3f' % (w1, w0))

    show_dataset(train, [0, 7], [0, 20], w1, w0)

    return [(w1 * row[0] + w0) for row in test]


def rmse(actual, predict):
    num = len(actual)
    return (sum((predict[i] - actual[i]) ** 2 for i in range(num)) / num) ** 0.5


def evaluate_algorithm(train_data, test_data, algorithm):
    y_row = [row[1] for row in test_data]

    predict = algorithm(train_data, test_data)
    res = rmse(y_row, predict)

    print("RSME: %.3f" % res)


def main():
    test_w1 = 2
    test_w0 = 3
    strength = 1.5

    train_data = get_dataset(100, 0, 6, test_w1, test_w0, strength)
    test_data = get_dataset(20, 6, 12, test_w1, test_w0, strength)

    evaluate_algorithm(train_data, test_data, simple_line_regression)


if __name__ == '__main__':
    main()
