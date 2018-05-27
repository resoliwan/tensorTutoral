import tensorflow as tf

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)
print(dataset1.output_shapes)

dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random_uniform([4]),
     tf.random_uniform([4, 100], dtype=tf.int32, maxval=100)))
print(dataset2.output_types)
print(dataset2.output_shapes)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)
print(dataset3.output_shapes)

data = tf.random_uniform([100], dtype=tf.int32, maxval=100)
sess = tf.Session()
print(sess.run(data))

dataset = tf.data.Dataset.from_tensor_slices(
    {'a': tf.random_uniform([4]),
     'b': tf.random_uniform([4, 100], dtype=tf.int32, maxval=100)})
print(dataset.output_types)
print(dataset.output_shapes)

dataset = tf.data.Dataset.range(10)
iterator = dataset.make_one_shot_iterator()
sess = tf.Session()
for i in range(10):
  value = sess.run(iterator.get_next())
  print(i, value)
  assert i == value

dataset = tf.data.Dataset.range(100)

print(sess.run(tf.fill([0], 0)))
print(sess.run(tf.fill([1], 1)))
print(sess.run(tf.fill([2], 2)))
print(sess.run(tf.fill([3], 3)))
print(sess.run(tf.fill([4], 4)))

dataset = tf.data.Dataset.range(10)
dataset = dataset.repeat(2)
dataset = dataset.batch(10)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess.run(iterator.initializer)
for i in range(2):
  sess.run(next_element)

dataset = tf.data.Dataset.range(10)
dataset = dataset.repeat(10)
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(10)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
for i in range(10):
  sess.run(next_element)

