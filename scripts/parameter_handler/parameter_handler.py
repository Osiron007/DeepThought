#! /usr/bin/env python

########################################################################################################################
# This file contains functions for goal point/trajectory generation
#   public functions:
#   __init__
#   read_parameter
#   print_parameter
#
#
########################################################################################################################

import ConfigParser

class param_handler(object):

    def __init__(self):

        self.init = False

        self.config = ConfigParser.ConfigParser()

        # hyperparameter
        self.BUFFER_PATH = "NOT PATH DEFINED"
        self.BATCH_SIZE = 128
        self.MAX_ITERATIONS = 100000000
        self.STATE_SIZE = 53
        self.ACTION_SIZE = 2
        self.SPLIT_STATE_SPACE = True
        self.STATE_SIZE_ENV = 48
        self.STATE_SIZE_KIN = 5

        # learning
        self.GAMMA = 0.99
        self.TAU = 0.001
        self.LRA = 0.0001
        self.LRC = 0.001

        # backup
        self.model_version = 99
        self.last_iteration_cnt = 1
        self.backup_path = "NOT PATH DEFINED"
        self.backup_interval = 500
        self.backup_interval_evaluation = 50000

        # logging
        self.log_path = "NOT PATH DEFINED"

        # other
        self.do_not_learn = True


    def read_parameter(self, filename):

        # print("#############Read configuration###################")
        print("OPENING CONFIGURATION AT: " + str(filename))
        self.config.readfp(open(filename))

        # hyperparameter
        self.BUFFER_PATH = self.config.get("hyperparameter", "buffer_path")
        self.BATCH_SIZE = self.config.getint("hyperparameter", "batchsize")
        self.MAX_ITERATIONS = self.config.getint("hyperparameter", "maxiterations")
        self.STATE_SIZE = self.config.getint("hyperparameter", "state_dim")
        self.ACTION_SIZE = self.config.getint("hyperparameter", "action_dim")
        self.SPLIT_STATE_SPACE = self.config.getboolean("hyperparameter", "split_state_space")

        # learning
        self.GAMMA = self.config.getfloat("learning", "gamma")
        self.TAU = self.config.getfloat("learning", "tau")
        self.LRA = self.config.getfloat("learning", "lra")
        self.LRC = self.config.getfloat("learning", "lrc")

        # backup
        self.model_version = self.config.getint("backup", "model_version")
        self.last_iteration_cnt = self.config.getint("backup", "last_iteration_cnt")
        self.backup_path = self.config.get("backup", "backup_path")
        self.backup_interval = self.config.getint("backup", "backup_interval")
        self.backup_interval_evaluation = self.config.getint("backup", "backup_interval_evaluation")

        # logging
        self.log_path = self.config.get("logging", "log_path")

        # other
        self.do_not_learn = self.config.getboolean("other", "do_not_learn")

        self.init = True

        return True

    def print_parameter(self):
        print("###############Hyperparameter#####################")
        print("Path to ReplayBuffer: " + str(self.BUFFER_PATH))
        print("Batch size: " + str(self.BATCH_SIZE))
        print("Max Iterations: " + str(self.MAX_ITERATIONS))
        print("State dimension: " + str(self.STATE_SIZE))
        print("Action dimension: " + str(self.ACTION_SIZE))
        print("Split State Space: " + str(self.SPLIT_STATE_SPACE))

        # learning
        print("#################Learning########################")
        print("GAMMA: " + str(self.GAMMA))
        print("TAU: " + str(self.TAU))
        print("LRA: " + str(self.LRA))
        print("LRC: " + str(self.LRC))

        # backup
        print("#################Backup########################")
        print("model_version: " + str(self.model_version))
        print("last_iteration_cnt: " + str(self.last_iteration_cnt))
        print("backup_path: " + str(self.backup_path))
        print("backup_interval: " + str(self.backup_interval))
        print("backup_interval_evaluation: " + str(self.backup_interval_evaluation))

        # logging
        print("#################Logging########################")
        print("log_path: " + str(self.log_path))

        # other
        print("#################Other########################")
        print("do_not_learn: " + str(self.do_not_learn))

        print("\n")


