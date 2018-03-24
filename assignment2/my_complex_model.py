# define model
def complex_model(X,y,is_training):
    
    # 7x7 Convolutional Layer with 32 filters and stride of 1
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='SAME') + bconv1
    
    # ReLU Activation Layer
    a2 = tf.nn.relu(a1)
    
    # Spatial Batch Normalization Layer 
    mean2, var2 = tf.nn.moments(a2, [0,1,2])
    gamma2 = tf.get_variable("gamma2", shape=[32])
    beta2 = tf.get_variable("beta2", shape=[32])
    a3 = tf.nn.batch_normalization(a2, mean2, var2, beta2, gamma2, 1e-4)
    
    # 2x2 Max Pooling layer with a stride of 2
    a4 = tf.nn.max_pool(a3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
    
    # Affine layer with 1024 output units
    W4 = tf.get_variable("W4", shape=[14*14*32, 1024])
    b4 = tf.get_variable("b4", shape=[1024])
    a4_flat = tf.reshape(a4,[-1,14*14*32])
    a5 = tf.matmul(a4_flat,W4) + b4
    
    # ReLU Activation Layer
    a6 = tf.nn.relu(a5)
    
    # Affine layer from 1024 input units to 10 outputs
    W6 = tf.get_variable("W6", shape=[1024, 10])
    b6 = tf.get_variable("b6", shape=[10])
    a7 = tf.matmul(a6,W6) + b6
    
    y_out = a7
    pass
