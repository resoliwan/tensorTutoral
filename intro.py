import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0, dtype=tf.float32)

total = a + b
print('a', a)
print('b', b)
print('total', total)

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()

print(sess.run((total, a, b)))
print(sess.run({'total': total, 'ab': (a, b)}))

vec = tf.random_uniform(shape=(3,))
print('vec', vec)
out1 = vec + 1
out2 = vec + 2

print(sess.run(vec))
print(sess.run((out1, out2)))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x - y
print(sess.run(z, feed_dict={x: 3, y: 1}))
print(sess.run(z, feed_dict={x: 1, y: 1}))
print(sess.run(z, feed_dict={x: [1, 2], y: [1, 3]}))

print(type([0, 1]))
print(type([0, 1,]))

my_data = [
    [0, 1,], 
    [2, 3,],
    [2, 1],
]

slices = tf.data.Dataset.from_tensor_slices(my_data)
print('slices', slices)
