
import numpy as np
import sqlite3
import random
from StringIO import StringIO

#THIS SCRIPT HANDLES A ReplayBuffer DB for StateSpaceV3

class ReplayBufferSQLite(object):

    def __init__(self, filepath_to_db, STATE_SIZE, ACTION_SIZE):

        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE

        #indicates state space version that can be handled
        self.version = 1

        # SQLite
        #db_name = "ReplayBuffer_Exp8-1.db"
        #path_to_db = "/home/nikas/Studium/MasterThesis/ReplayBuffer/" + db_name

        # open connection to db
        self.connection = sqlite3.connect(filepath_to_db)

        # create a cursor for db
        self.cursor = self.connection.cursor()

        # ID for next transition
        self.id = self.size()

        print("ReplayBuffer: Current ID on StartUp: " + str(self.id))

        #set cache size
        self.cursor.execute("PRAGMA cache_size = 500000")

        # set journal mode
        self.cursor.execute("PRAGMA journal_mode = MEMORY")

        # set temp store
        self.cursor.execute("PRAGMA temp_store = MEMORY")

        # set synchronous
        self.cursor.execute("PRAGMA synchronous = OFF")


    def getBatch(self, batch_size):

        element_list = []
        # Randomly sample batch_size examples
        BufferLength = self.id - 1
        list_ids = random.sample(xrange(BufferLength), batch_size)

        sql_command_get_batch = 'SELECT * FROM ReplayBuffer WHERE '

        for id in list_ids:
            sql_command_get_batch = sql_command_get_batch + 'id = ' + str(id) + ' OR '

        sql_command_get_batch = sql_command_get_batch + 'id = ' + str(id) + ';'

        self.cursor.execute(sql_command_get_batch)
        results = self.cursor.fetchall()

        for result in results:
            #s_t = np.zeros(self.state_size)
            #s_t1 = np.zeros(self.state_size)
            #a_t = np.zeros(self.action_size)
            #r_t = 0
            #drone = 0

            s_t = result[1:6]
            a_t = result[6:8]
            r_t = result[8]
            s_t1 = result[9:14]
            done = result[14]

            element = (s_t, a_t, r_t, s_t1, done)

            element_list.append(element)

        #update statistics
        #self.update_statistics(list_ids)

        return element_list

    def getBatchList(self, batch_size, list_size):

        Batch_List = []
        Batch = []
        # Randomly sample batch_size examples
        BufferLength = self.id - 1
        id_cnt = batch_size * list_size
        list_ids = random.sample(xrange(BufferLength), id_cnt)

        #print(id_cnt)

        sql_command_get_batch = 'SELECT * FROM ReplayBuffer WHERE '

        id = 0

        for id in list_ids:
            sql_command_get_batch = sql_command_get_batch + 'id = ' + str(id) + ' OR '

        sql_command_get_batch = sql_command_get_batch + 'id = ' + str(id) + ';'

        #print(sql_command_get_batch)

        self.cursor.execute(sql_command_get_batch)
        results = self.cursor.fetchall()

        element_cnt = 1

        for result in results:

            if element_cnt == batch_size:
                #print("Elements: " + str(element_cnt))
                #print("Batch Full")
                element_cnt = 1
                Batch_List.append(Batch)
                Batch = []
            else:
                s_t = result[1:6]
                a_t = result[6:8]
                r_t = result[8]
                s_t1 = result[9:14]
                done = result[14]

                element = (s_t, a_t, r_t, s_t1, done)

                Batch.append(element)

            element_cnt = element_cnt + 1

        # update statistics
        #self.update_statistics(list_ids)

        return Batch_List

    def update_statistics(self, list_ids):

        # Add one to all IDs statistics cnt
        # UPDATE Products SET Price = Price + 50 WHERE ProductID = 1
        # UPDATE {Table} SET {Column} = {Column} + {Value} WHERE {Condition}


        sql_command_update_statistics = 'UPDATE Statistics SET Count = Count + 1 WHERE '

        id_cnt = 0
        id_max_cnt = len(list_ids)-1

        #print(id_cnt)
        #print(id_max_cnt)

        for id in list_ids:
            if id_cnt < id_max_cnt:
                sql_command_update_statistics = sql_command_update_statistics + 'id = ' + str(id) + ' OR '
            else:
                sql_command_update_statistics = sql_command_update_statistics + 'id = ' + str(id) + ';'
            id_cnt = id_cnt + 1

        #print(sql_command_update_statistics)

        self.cursor.execute(sql_command_update_statistics)
        #self.save_db()
        # results = self.cursor.fetchall()

    def reset_statistics(self, list_ids):

        # Add one to all IDs statistics cnt
        # UPDATE Products SET Price = Price + 50 WHERE ProductID = 1
        # UPDATE {Table} SET {Column} = {Column} + {Value} WHERE {Condition}


        sql_command_update_statistics = 'UPDATE Statistics SET Count = 0 WHERE '

        id_cnt = 0
        id_max_cnt = len(list_ids)-1

        #print(id_cnt)
        #print(id_max_cnt)

        for id in list_ids:
            if id_cnt < id_max_cnt:
                sql_command_update_statistics = sql_command_update_statistics + 'id = ' + str(id) + ' OR '
            else:
                sql_command_update_statistics = sql_command_update_statistics + 'id = ' + str(id) + ';'
            id_cnt = id_cnt + 1

        print(sql_command_update_statistics)

        self.cursor.execute(sql_command_update_statistics)
        self.save_db()
        # results = self.cursor.fetchall()

    def getElement(self, id):

        sql_command_get_batch = 'SELECT * FROM ReplayBuffer WHERE id = ' + str(id)

        self.cursor.execute(sql_command_get_batch)
        result = self.cursor.fetchall()

        #print(len(result[0]))

        s_t = np.zeros(self.state_size)
        s_t1 = np.zeros(self.state_size)
        a_t = np.zeros(self.action_size)
        r_t = 0
        drone = 0

        s_t = result[0][1:6]
        a_t = result[0][6:8]
        r_t = result[0][8]
        s_t1 = result[0][9:14]
        done = result[0][14]

        element = (s_t, a_t, r_t, s_t1, done)

        return element


    def size(self):

        sql_command_get_size = 'SELECT Count(*) FROM ReplayBuffer'

        # execute sql command
        self.cursor.execute(sql_command_get_size)

        result = self.cursor.fetchall()

        #print(result[0][0])

        size = int(result[0][0])

        #print(size)

        print("ReplayBuffer: Size: " + str(size) + " max ID: " + str(size-1) + " next ID: " + str(size))

        return size

    def count(self):
        return self.id

    def add(self, state, action, reward, new_state, done):
        states_tmp = np.zeros(self.state_size)
        next_states_tmp = np.zeros(self.state_size)
        actions_tmp = np.zeros(self.action_size)
        reward_tmp = reward
        done_tmp = done

        states_tmp[:] = state[:]
        next_states_tmp[:] = new_state[:]
        actions_tmp[:] = action[:]

        #######################################
        #          CREATE SQL STRING          #
        #######################################

        # create sql_command string for Transition insertion
        sql_command_add_transition = 'INSERT INTO ReplayBuffer (id, '

        ###########################
        #       ADD FIELDS        #
        ###########################

        # s_t fields
        for field_nr in range(0, 5, 1):
            field_name = 's_t_' + str(field_nr) + ', '
            sql_command_add_transition = sql_command_add_transition + field_name

        # a_t fields
        sql_command_add_transition = sql_command_add_transition + 'a_t_0, a_t_1, '

        # r_t field
        sql_command_add_transition = sql_command_add_transition + 'r_t, '

        # s_t1 fields
        for field_nr in range(0, 5, 1):
            field_name = 's_t1_' + str(field_nr) + ', '
            sql_command_add_transition = sql_command_add_transition + field_name

        # done field
        sql_command_add_transition = sql_command_add_transition + 'done'
        sql_command_add_transition = sql_command_add_transition + ') VALUES ('

        ###########################
        #       ADD DATA          #
        ###########################

        # ID field
        sql_command_add_transition = sql_command_add_transition + str(self.id) + ', '

        # s_t fields
        for field_nr in range(0, 5, 1):
            field_value = str(states_tmp[field_nr]) + ', '
            sql_command_add_transition = sql_command_add_transition + field_value

        # a_t fields
        sql_command_add_transition = sql_command_add_transition + str(actions_tmp[0]) + ', ' + str(actions_tmp[1]) + ', '

        # r_t field
        sql_command_add_transition = sql_command_add_transition + str(reward_tmp) + ', '

        # s_t1 fields
        for field_nr in range(0, 5, 1):
            field_value = str(next_states_tmp[field_nr]) + ', '
            sql_command_add_transition = sql_command_add_transition + field_value

        # done field
        sql_command_add_transition = sql_command_add_transition + str(int(done_tmp))
        sql_command_add_transition = sql_command_add_transition + ');'

        ###########################
        #    execute command      #
        ###########################

        print(sql_command_add_transition)

        # execute sql command
        self.cursor.execute(sql_command_add_transition)

        ###########################
        #   add id to statistics  #
        ###########################

        # create sql_command string for adding new id to statistics table
        #sql_command_create_statistics_id = 'INSERT INTO Statistics (id, Count) VALUES ('

        #ID field
        #sql_command_create_statistics_id = sql_command_create_statistics_id + str(self.id) + ', '

        #Count field
        #sql_command_create_statistics_id = sql_command_create_statistics_id + str(0) + ');'

        ###########################
        #    execute command      #
        ###########################

        #print(sql_command_create_statistics_id)
        # execute sql command
        #self.cursor.execute(sql_command_create_statistics_id)

        # ONLY COMMIT EVERY 500 Iteration with self.save_db for performance boost
        # never forget this, if you want the changes to be saved:
        # self.connection.commit()

        self.id = self.id + 1

    def execute_sql(self, sql_string):
        try:
            # execute sql command
            self.cursor.execute(sql_string)
        except:
            print("REPLAYBUFFER: Error in SQL Statement")

    def save_db(self):
        # never forget this, if you want the changes to be saved:
        self.connection.commit()

    def close_db(self):
        print("REPLAYBUFFER: Close DB")
        # close connection
        self.connection.close()
