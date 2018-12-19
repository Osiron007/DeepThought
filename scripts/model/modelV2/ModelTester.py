from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import tensorflow as tf

from keras.utils import plot_model

sess = tf.Session()
from keras import backend as K

K.set_session(sess)

state_dim = 53

action_dim = 2

BATCH_SIZE = 32

TAU = 0.01

LRA = 0.001

actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)

critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)

print(actor.model.summary())
print(critic.model.summary())

#print(actor.target_model.summary())

#plot_model(actor.model, show_shapes=True, to_file='ActorModel_Exp8-5.png')
#plot_model(critic.model, show_shapes=True, to_file='CriticModel_Exp8-5.png')

