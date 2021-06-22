from re import X
from numpy.core.fromnumeric import shape
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
from PIL import Image

# This part of the code saves the model to be used afterwards
model = onnx.load('/home/iris/catkin_ws/src/nn_tester/NNetworks/RollerBal/RollerBall/RollerBall-4858.onnx') # Load the ONNX file
tf_model = prepare(model) # Import the ONNX model to Tensorflow
tf_model.export_graph('/home/iris/catkin_ws/src/nn_tester/NNetworks/RollerBal/RollerBall_saved')
# print(shape(tf_model.inputs)) # Input nodes to the model
print(tf_model.outputs) # Output nodes from the model
tf_model = tf.saved_model.load('/home/iris/catkin_ws/src/nn_tester/NNetworks/RollerBal/RollerBall_saved')

X = np.array([[1,0.5,1,0,0.5,0,0,0]]).astype(np.float32)
# seq_lens = np.repeat(X.shape[0],X.shape[1]).astype(np.int32)
# init_h = 0.1 * np.ones((1, X.shape[1], 2)).astype(np.float32)
# y = tf_model.run({"obs_0": X })
y = tf_model(obs_0=X)

print(y[2])
print(y[2][0][1].numpy())



# imported = tf.keras.models.load_model('/home/iris/Desktop/Training_ball/results/rb_06/b_test')
# imported.s
# input = [0,0,0,0,0,0,0,0]
# print(imported.predict(test_images).shape)

# f = imported.signatures["serving_default"]
# print(f(obs_0=tf.cast(X, tf.float32),obs_0=tf.cast(, tf.float32)))
# input = tf.constant([0])
# input = ['0,0,0,0,0,0,0,0']
# # img = np.asarray(input, dtype=np.float32)
# print(type(input))
# print(shape(input))
# Y = tf_model.run(input)[0][0] #Inference