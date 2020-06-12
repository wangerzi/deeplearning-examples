import tensorflow as tf


def main():
    a = tf.constant(4, name="input_a")
    b = tf.constant(2, name="input_b")
    c = tf.multiply(a, b, name="op-multi")
    d = tf.add(a, b)
    e = tf.add(c, d)

    with tf.Session() as sess:
        print(sess.run([d, e]))
        writer = tf.summary.FileWriter('./graph-data/1')
        writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()
