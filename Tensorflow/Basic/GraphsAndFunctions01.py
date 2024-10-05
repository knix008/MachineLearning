import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import timeit
from datetime import datetime

print("Tensorflow version : ", tf.__version__)

from tensorflow.python.client import device_lib
if tf.test.is_gpu_available(): 
  # Check GPU support 
  if tf.test.is_built_with_cuda(): 
      os.system("nvidia-smi")
  print(device_lib.list_local_devices())

def a_regular_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

a_function_that_uses_a_graph = tf.function(a_regular_function)

# Make some tensors.
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

orig_value = a_regular_function(x1, y1, b1).numpy()
# Call a `tf.function` like a Python function.
tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()
assert(orig_value == tf_function_value)

def inner_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# Using the `tf.function` decorator makes `outer_function` into a
# `PolymorphicFunction`.
@tf.function
def outer_function(x):
  y = tf.constant([[2.0], [3.0]])
  b = tf.constant(4.0)

  return inner_function(x, y, b)

# Note that the callable will create a graph that
# includes `inner_function` as well as `outer_function`.
outer_function(tf.constant([[1.0, 2.0]])).numpy()

def simple_relu(x):
  if tf.greater(x, 0):
    return x
  else:
    return 0

# Using `tf.function` makes `tf_simple_relu` a `PolymorphicFunction` that wraps
# `simple_relu`.
tf_simple_relu = tf.function(simple_relu)

print("First branch, with graph:", tf_simple_relu(tf.constant(1)).numpy())
print("Second branch, with graph:", tf_simple_relu(tf.constant(-1)).numpy())

# This is the graph-generating output of AutoGraph.
print(tf.autograph.to_code(simple_relu))

# This is the graph itself.
print(tf_simple_relu.get_concrete_function(tf.constant(1)).graph.as_graph_def())

@tf.function
def my_relu(x):
  return tf.maximum(0., x)

# `my_relu` creates new graphs as it observes different input types.
print(my_relu(tf.constant(5.5)))
print(my_relu([1, -1]))
print(my_relu(tf.constant([3., -3.])))

# These two calls do *not* create new graphs.
print(my_relu(tf.constant(-2.5))) # Input type matches `tf.constant(5.5)`.
print(my_relu(tf.constant([-1., 1.]))) # Input type matches `tf.constant([3., -3.])`.

# There are three `ConcreteFunction`s (one for each graph) in `my_relu`.
# The `ConcreteFunction` also knows the return type and shape!
print(my_relu.pretty_printed_concrete_signatures())

@tf.function
def get_MSE(y_true, y_pred):
  sq_diff = tf.pow(y_true - y_pred, 2)
  return tf.reduce_mean(sq_diff)

y_true = tf.random.uniform([5], maxval=10, dtype=tf.int32)
y_pred = tf.random.uniform([5], maxval=10, dtype=tf.int32)
print(y_true)
print(y_pred)

get_MSE(y_true, y_pred)

tf.config.run_functions_eagerly(True)

get_MSE(y_true, y_pred)

# Don't forget to set it back when you are done.
tf.config.run_functions_eagerly(False)

@tf.function
def get_MSE(y_true, y_pred):
  print("Calculating MSE!")
  sq_diff = tf.pow(y_true - y_pred, 2)
  return tf.reduce_mean(sq_diff)

error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)

# Now, globally set everything to run eagerly to force eager execution.
tf.config.run_functions_eagerly(True)

# Observe what is printed below.
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)

tf.config.run_functions_eagerly(False)

def unused_return_eager(x):
  # Get index 1 will fail when `len(x) == 1`
  tf.gather(x, [1]) # unused 
  return x

try:
  print(unused_return_eager(tf.constant([0.0])))
except tf.errors.InvalidArgumentError as e:
  # All operations are run during eager execution so an error is raised.
  print(f'{type(e).__name__}: {e}')

@tf.function
def unused_return_graph(x):
  tf.gather(x, [1]) # unused
  return x

# Only needed operations are run during graph execution. The error is not raised.
print(unused_return_graph(tf.constant([0.0])))

x = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)

def power(x, y):
  result = tf.eye(10, dtype=tf.dtypes.int32)
  for _ in range(y):
    result = tf.matmul(x, result)
  return result

print("Eager execution:", timeit.timeit(lambda: power(x, 100), number=1000), "seconds")

power_as_graph = tf.function(power)
print("Graph execution:", timeit.timeit(lambda: power_as_graph(x, 100), number=1000), "seconds")

@tf.function
def a_function_with_python_side_effect(x):
  print("Tracing!") # An eager-only side effect.
  return x * x + tf.constant(2)

# This is traced the first time.
print(a_function_with_python_side_effect(tf.constant(2)))
# The second time through, you won't see the side effect.
print(a_function_with_python_side_effect(tf.constant(3)))

# This retraces each time the Python argument changes,
# as a Python argument could be an epoch count or other
# hyperparameter.
print(a_function_with_python_side_effect(2))
print(a_function_with_python_side_effect(3))