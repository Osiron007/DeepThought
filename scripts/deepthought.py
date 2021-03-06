#Basic
import numpy as np
import json
from time import *
import os

#ROS for Python
import rospy
#ROS Services
from rl_srvs.srv import Predict, PredictResponse
#ROS Msgs
from rl_msgs.msg import Transition

#Tensorflow
import tensorflow as tf

#Keras
from keras import backend as K

K.set_learning_phase(1) #set learning phase for Dropout usage!!!

#Parameter
from parameter_handler import parameter_handler as PH

#ReplayBuffer SQLite
#from ReplayBuffer_SQLite import ReplayBufferSQLite as ReplayBuffer
from scripts.replay_buffer.ReplayBuffer_SQLiteV1 import ReplayBufferSQLite as ReplayBuffer

from model.modelV2.ActorNetwork import ActorNetwork
from model.modelV2.CriticNetwork import CriticNetwork

MODEL_PATH="model/modelV2"

CONFIG_FILE = "configuration1.cfg"


def ros_add_to_replaybuffer(Transition_from_ros):
    print("adding Data to ReplayBuffer")

    ##########################################
    #          parameter_handler             #
    ##########################################
    global params

    ############################################
    #              ReplayBuffer                #
    ############################################
    global Buffer_new_data_List
    global adding_data


    #convert data to np.arrays
    ros_s_t = np.array(np.zeros_like(params.STATE_SIZE),dtype='float32')
    ros_s_t1 = np.array(np.zeros_like(params.STATE_SIZE), dtype='float32')
    ros_a_t = np.array(np.zeros_like(params.ACTION_SIZE), dtype='float32')
    ros_reward = 0.0
    ros_done = False

    ros_s_t = np.asarray(Transition_from_ros.s_t,dtype='float32')
    ros_a_t = np.asarray(Transition_from_ros.a_t, dtype='float32')
    ros_s_t1 = np.asarray(Transition_from_ros.s_t1, dtype='float32')
    ros_reward = float(Transition_from_ros.reward)
    ros_done = bool(Transition_from_ros.done)

    data_OK = True

    if not ros_s_t.shape[0] == params.STATE_SIZE:
        print("Error: shape of s_t wrong!")
        print("Shape is " + str(ros_s_t.shape[0]) + " instead of " + str(params.STATE_SIZE))
        data_OK = False

    if not ros_s_t1.shape[0] == params.STATE_SIZE:
        print("Error: shape of s_t1 wrong!")
        print("Shape is " + str(ros_s_t1.shape[0]) + " instead of " + str(params.STATE_SIZE))
        data_OK = False

    if not ros_a_t.shape[0] == params.ACTION_SIZE:
        print("Error: shape of a_t wrong!")
        print("Shape is " + str(ros_a_t.shape[0]) + " instead of " + str(params.ACTION_SIZE))
        data_OK = False

    #print("Empfangen: " + str(Transition_from_ros.s_t))
    #print("Adde: " + str(ros_s_t))

    #print("Reward empf: " + str(ros_reward))
    #print("Reward: " + str(ros_reward))

    #def add(self, state, action, reward, new_state, done):

    if data_OK and not adding_data:
        element = (ros_s_t, ros_a_t, ros_reward, ros_s_t1, ros_done)
        Buffer_new_data_List.append(element)
        print("Number of elements " + str(len(Buffer_new_data_List)))
    else:
        print("FAILED to add Transition to ReplayBuffer")

    print("Transitions in Buffer: " + str(Buffer.count()))


def log_data_to_file(filepath, target_q_value, loss):
    print("logging target_q_value and loss to file")
    f = open(str(filepath) + 'q_list.log', 'a+')
    msg = '' + str(target_q_value) + "," + str(ctime(time()))
    f.write(str(msg) + "\n")
    f.close()

    f = open(str(filepath) + 'loss.log', 'a+')
    f.write(str(loss) + "\n")
    f.close()

if __name__ == "__main__":

    ##########################################
    #          parameter_handler             #
    ##########################################
    global params
    params = PH.param_handler()
    params.read_parameter(CONFIG_FILE)
    params.print_parameter()


    #ROS init and definitions

    ##########################################
    #                ROS-Node                #
    ##########################################
    rospy.init_node('multi_learning', anonymous=True)
    rate = rospy.Rate(20)  # 20Hz

    sub_transitions = rospy.Subscriber("multi_learning/transitions", Transition, ros_add_to_replaybuffer)

    ############################################
    #              ReplayBuffer                #
    ############################################
    global Buffer
    Buffer = ReplayBuffer(params.BUFFER_PATH, params.STATE_SIZE, params.ACTION_SIZE)
    global Buffer_new_data_List
    Buffer_new_data_List = []
    global adding_data
    adding_data = False

    ############################################
    #          Tensorflow and Keras            #
    ############################################

    #aviod TF from allocation all GPU mem
    #https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #sess = tf.Session()
    K.set_session(sess)

    ############################################
    #                Networks                  #
    ############################################

    global actor

    actor = ActorNetwork(sess, params.STATE_SIZE, params.ACTION_SIZE, params.BATCH_SIZE,
                         params.TAU, params.LRA)
    critic = CriticNetwork(sess, params.STATE_SIZE, params.ACTION_SIZE,
                           params.BATCH_SIZE, params.TAU, params.LRC)

    #make TF graph usable for callback functions
    global graph
    graph = tf.get_default_graph()

    # Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights(str(params.backup_path) + "actormodel.h5")
        critic.model.load_weights(str(params.backup_path) + "criticmodel.h5")
        actor.target_model.load_weights(str(params.backup_path) + "target_actormodel.h5")
        critic.target_model.load_weights(str(params.backup_path) + "target_criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")
        #TODO store initial weights at backup location


    print("ReplayBuffer size on startup: " + str(Buffer.count()))

    iteration = params.last_iteration_cnt

    backup_id = params.next_backup_id

    print("Starting with iteration " + str(iteration) + "  " + str(ctime(time())))

    while not rospy.is_shutdown() and iteration < params.MAX_ITERATIONS:

        if Buffer.count() > 1000 and not params.do_not_learn:

            loss = 0

            batch_list = Buffer.getBatchList(params.BATCH_SIZE,5)


            for batch in batch_list:

                print("Iteration " + str(iteration))
                #print("Batch_List")
                # Do the batch update
                #batch = Buffer.getBatch(BATCH_SIZE)

                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[1] for e in batch])

                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

                #print(target_q_values)

                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + params.GAMMA * target_q_values[k]

                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                #print(a_for_grad)
                grads = critic.gradients(states, a_for_grad)
                #print(grads)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

                # log data and save model every backup_interval iteration
                if np.mod(iteration, params.backup_interval) == 0:

                    print("Adding Data to ReplayBuffer")
                    adding_data = True
                    #add new data to ReplayBuffer
                    for transition in Buffer_new_data_List:
                        print("Adding")
                        Buffer.add(transition[0],transition[1],transition[2],transition[3],transition[4])

                    #delete list
                    Buffer_new_data_List = []

                    #save statistics to db
                    Buffer.save_db()

                    adding_data = False

                    print("Iteration: " + str(iteration))

                    #calc mean target q value
                    log_q_value = 0
                    q_cnt = 0
                    for value in target_q_values:
                        log_q_value = log_q_value + value
                        q_cnt = q_cnt + 1

                    log_q_value = log_q_value/q_cnt

                    log_data_to_file(params.log_path,log_q_value,loss)

                    ##f = open(str(params.log_path) + 'q_list.log', 'a+')
                    #msg = '' + str(log_q_value) + "," + str(ctime(time()))
                    #f.write(str(msg) + "\n")
                    #f.close()

                    #print("Now we save loss")
                    #f = open(str(params.log_path) + 'loss.log', 'a+')
                    #f.write(str(loss) + "\n")
                    #f.close()

                    # every backup_interval_evaluation iteration create a new file for backup
                    if np.mod(iteration, params.backup_interval_evaluation) == 0:
                        print("Now we save model for evaluation")

                        #create folder for evaluation backup
                        os.mkdir(str(params.backup_path) + str(backup_id))

                        backup_path_eval = str(params.backup_path) + str(backup_id)

                        actor.model.save_weights(str(backup_path_eval) + "/actormodel.h5", overwrite=True)
                        with open(str(backup_path_eval) + "/actormodel.json", "w") as outfile:
                            json.dump(actor.model.to_json(), outfile)

                        actor.target_model.save_weights(str(backup_path_eval) + "/target_actormodel.h5", overwrite=True)
                        with open(str(backup_path_eval) + "/target_actormodel.json", "w") as outfile:
                            json.dump(actor.target_model.to_json(), outfile)

                        critic.model.save_weights(str(backup_path_eval) + "/criticmodel.h5", overwrite=True)
                        with open(str(backup_path_eval) + "/criticmodel.json", "w") as outfile:
                            json.dump(critic.model.to_json(), outfile)

                        critic.target_model.save_weights(str(backup_path_eval) + "/target_criticmodel.h5", overwrite=True)
                        with open(str(backup_path_eval) + "/target_criticmodel.json", "w") as outfile:
                            json.dump(critic.target_model.to_json(), outfile)

                        backup_id = backup_id + 1

                    else:
                        print("Now we save model tmp")

                        actor.model.save_weights(str(params.backup_path) + "actormodel.h5", overwrite=True)
                        with open(str(params.backup_path) + "actormodel.json", "w") as outfile:
                            json.dump(actor.model.to_json(), outfile)

                        actor.target_model.save_weights(str(params.backup_path) + "target_actormodel.h5", overwrite=True)
                        with open(str(params.backup_path) + "target_actormodel.json", "w") as outfile:
                            json.dump(actor.target_model.to_json(), outfile)

                        critic.model.save_weights(str(params.backup_path) + "criticmodel.h5", overwrite=True)
                        with open(str(params.backup_path) + "criticmodel.json", "w") as outfile:
                            json.dump(critic.model.to_json(), outfile)

                        critic.target_model.save_weights(str(params.backup_path) + "target_criticmodel.h5", overwrite=True)
                        with open(str(params.backup_path) + "target_criticmodel.json", "w") as outfile:
                            json.dump(critic.target_model.to_json(), outfile)

                iteration = iteration + 1

        else:
            if len(Buffer_new_data_List) > 1000:
                print("Adding Data to ReplayBuffer")
                adding_data = True
        #         # add new data to ReplayBuffer
                for transition in Buffer_new_data_List:
                    print("Adding")
                    Buffer.add(transition[0], transition[1], transition[2], transition[3], transition[4])
        #
        #         # delete list
                Buffer_new_data_List = []
        #
        #         # save statistics to db
                Buffer.save_db()
        #
                adding_data = False
        #
        #         print("Iteration: " + str(iteration))

            #s1 = time()
            #diff = s1 - s
            #print("Cycle Time: " + str(diff * 1000) + " [ms]")
        #print("sleeping")
        rate.sleep()

        #print("resumed")

    Buffer.close_db()
    print("async_multi_ddpg finished after " + str(iteration) + " iterations")



