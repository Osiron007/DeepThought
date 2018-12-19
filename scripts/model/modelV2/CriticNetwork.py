from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.layers.merge import concatenate, add
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import normal, RandomNormal
import keras.backend as K
import tensorflow as tf


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim):
        print("Now we build the model CriticV2")
        S = Input(shape=[state_size], name='state_input', dtype='float32')
        A = Input(shape=[action_dim], name='action_input', dtype='float32')
        w1 = Dense(1024, activation='relu', kernel_initializer=RandomNormal(stddev=1e-4), name='StateLayer')(S)
        a1 = Dense(512, activation='linear', kernel_initializer=RandomNormal(stddev=1e-4), name='ActionLayer')(A)
        h1 = Dense(512, activation='linear', kernel_initializer=RandomNormal(stddev=1e-4), name='HiddenLayer1')(w1)
        h2 = add([h1, a1], name='HiddenLayer2')
        h3 = Dense(512, activation='relu', kernel_initializer=RandomNormal(stddev=1e-4), name='HiddenLayer3')(h2)
        V = Dense(action_dim, activation='linear', kernel_initializer=RandomNormal(stddev=1e-4), name='OutputLayer')(h3)
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S
