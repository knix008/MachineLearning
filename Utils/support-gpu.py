import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# tensorflow test메소드를 통한 체크
import tensorflow as tf
tf.test.is_built_with_cuda()   


# device_lib를 통한 체크
# 실행결과에 cpu, gpu 둘 다 나와야 정상 
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())