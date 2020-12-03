import pandas as pd
import numpy as np
from das.performance_evaluation import is_larger_better
import logging
import das
import time
import math

logger = logging.getLogger(das.logger_name)

class QLearningAgent:

    def __init__(self,
                 states,
                 actions,
                 state_action_value,
                 learning_rate=0.3,
                 epsilon=0.88,
                 gamma=0.99,
                 datapreprocessing_state=None,
                 featureselect_state=None,
                 final_state=None,
                 is_larger_better=True,
                 method="random",
                 init_val=0.5,
                 time_budget=3600,
                 random_state=None):

        self.g_time_start = time.time()
        self.time_budget = time_budget

        # hard code: we could get a nice curve when decay_parameter = 7
        self.decay_parameter = 7

        logger.debug("init val: {}".format(init_val))

        self.actions = actions
        self.learning_rate = learning_rate
        self.epsilon = 1
        self.final_epsilon = epsilon
        self.gamma = gamma
        self.datapreprocess_state = datapreprocessing_state
        self.featureselect_state = featureselect_state
        self.final_state = final_state
        self.is_larger_better = is_larger_better
        self.state_action_value = state_action_value
        self.random_state = random_state

        self.init_q_table(method=method, init_val=init_val)
        logger.debug("self.state_action_value: {}".format(self.state_action_value))
        self.q_learn_table = pd.DataFrame(data=self.state_action_value,
                                          index=states,
                                          columns=actions)


    def _sigmoid_func(self, time_left_ratio):
        res = 1 / (1 + math.exp(self.decay_parameter * (time_left_ratio - 0.5)))
        return res

    def _epsilon_judge(self):
        logger.debug("epsilon: {}".format(self.epsilon))

        # update epsilon
        time_left_ratio = (time.time() - self.g_time_start) / self.time_budget

        logger.debug("time: {}, self.time_budget:{}, self.g_time_start:{}".format(time.time(), self.time_budget, self.g_time_start))

        if self.epsilon > self.final_epsilon:
            self.epsilon = max(self._sigmoid_func(time_left_ratio), self.final_epsilon)


        if self.epsilon < np.random.uniform():
            return True
        else:
            return False

    def get_greedy_action(self, state):
        action = None

        self._check_state_exist(state)
        state_action = self.q_learn_table.loc[state, :]
        # Attention !!!, the raw method below, numpy 1.15.x not working here, will cause bug !
        # state_action = state_action.reindex(np.random.permutation(state_action.index))

        state_action = state_action.sample(frac=1, random_state=self.random_state)

        if self.is_larger_better:
            action = state_action.idxmax()
        else:

            action = state_action.idxmax()

        return action

    def get_action_from_final_state(self):
        pass

    def get_action_from_datapreprocess_state(self):
        pass

    def get_action_from_featureselect_state(self):
        pass

    @DeprecationWarning
    def choose_action(self, state):
        action = None

        self._check_state_exist(state)


        for i in range(100):
            if self._epsilon_judge():
                state_action = self.q_learn_table.loc[state, :]
                # state_action = state_action.reindex(np.random.permutation(state_action.index))
                action = state_action.idxmax()

            else:
                action = np.random.choice(self.q_learn_table.loc[state, self.q_learn_table.loc[state] >= 0].index)

            if self.q_learn_table.loc[state, action] > 0.1:
                break

        #print(action)
        return action

    def learn(self, state, action, reward, state_):

        logger.debug("state: {}, action: {}, state_: {}".format(state, action, state_))

        self._check_state_exist(state)
        q_predict = self.q_learn_table.loc[state, action]
        # if state_ != "terminal":
        #     q_target = reward + self.gamma * self.q_learn_table.loc[state_, :].max()
        # else:
        #     q_target = reward

        if state_ not in self.final_state:
            q_target = reward + self.gamma * self.q_learn_table.loc[state_, :].max()
        else:
            q_target = reward

        self.q_learn_table.loc[state, action] += self.learning_rate * (q_target - q_predict)

    """
    def learn_for_pipeline(self, states, reward, worst_score, rule, crashed_state=None):
        logger.debug("states {}".format(states))
        logger.debug("crashed state {}".format(crashed_state))
        reward = self.get_reward(reward, rule=rule, worst_score=worst_score)
        logger.debug("reward: {}".format(reward))

        if crashed_state is None:
            for s in range(len(states), -1, -1):
                if s == len(states):
                    self.learn(states[s - 1], "terminate", reward, "terminate")
                elif s == 0:
                    self.learn("start", states[s], reward, states[s])
                else:
                    self.learn(states[s - 1], states[s], reward, states[s])

        else:
            for s in range(len(states)-1, -1, -1):
                if s == len(states):
                    self.learn(states[s - 1], "terminate", reward, "terminate")

                elif s == 0:
                    self.learn("start", states[s], reward, states[s])
                else:
                    self.learn(states[s - 1], states[s], reward, states[s])

                # if states[s] == crashed_state:
                #     break
    """


    # """
    def learn_for_pipeline(self, states, reward, worst_score, rule, crashed_state=None):

        logger.debug("states {}".format(states))
        if crashed_state == 'start':
            logger.debug("CRASHED state is 'start'!!! ")
            return
        logger.info("crashed state {}".format(crashed_state))
        reward = self.get_reward(reward, rule=rule, worst_score=worst_score)
        logger.info("reward: {}".format(reward))
        if crashed_state is None:
            for s in range(len(states)+1):
                if s == 0:
                    self.learn("start", states[s], reward, states[s])
                elif s == len(states):
                    self.learn(states[s-1], "terminate", reward, "terminate")
                else:
                    self.learn(states[s-1], states[s], reward, states[s])
        else:
            for s in range(len(states)+1):
                # if crashed, "start" --> pipeline[0] must be punished.
                if s == 0:
                    self.learn("start", states[s], reward, states[s])
                elif s == len(states):
                    self.learn(states[s-1], "terminate", reward, "terminate")
                    break
                else:
                    self.learn(states[s-1], states[s], reward, states[s])
                if states[s] == crashed_state:
                    break
    # """


    def _check_state_exist(self, state):
        #print(state)
        if state not in self.q_learn_table.index:
            raise Exception("no such state")


    def get_q_learn_table(self):
        return self.q_learn_table

    def get_reward(self, reward, rule, worst_score):

        res = 0
        if is_larger_better(rule=rule):
            if reward <= worst_score:
                res = worst_score * 0.9
            else:
                res = reward
        else:
            if worst_score - reward < 0:
                res = worst_score * 0.9
            else:
                res = worst_score - reward + worst_score

        return res * (1 - self.gamma)

    def init_q_table(self, method="random", init_val=0.5):


        for i in range(self.state_action_value.shape[0]):
            for j in range(self.state_action_value.shape[1]):
                if self.state_action_value[i, j] < 0:
                    continue
                else:
                    if method == "uniform":
                        self.state_action_value[i, j] = init_val
                    elif method == "random":
                        self.state_action_value[i, j] = np.random.normal(init_val, 0.2*init_val)





if __name__ == "__main__":

    state = [
        "start",

        "StandardScalar",

        "PCA",

        "LogisticRegression",

        "terminal"
    ]

    action = [
        "StandardScalar",

        "PCA",

        "LogisticRegression",

        "terminal"
    ]

    sa = [
        [0, 0, 0, -1],
        [-1, 0, 0, -1],
        [-1, -1, 0, -1],
        [-1, -1, -1, 0],
        [0, 0, 0, 0]
    ]
    import sklearn.datasets

    import sklearn.model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X, y = sklearn.datasets.load_digits(return_X_y=True)

    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(X, y, random_state=1)
    print(X.shape)
    print(y.shape)

    qtable = QLearningAgent(state, action, sa)

    for episode in range(1000):
        total = []
        #print(qtable.get_q_learn_table)

        a = "start"
        b = "start"
        while a != "terminal":
            b = a
            a = qtable.choose_action(a)
            if a == "StandardScalar":
                total.append((a, StandardScaler()))
                qtable.learn(b, a, 0, a)
            elif a == "PCA":
                total.append((a, PCA()))
                qtable.learn(b, a, 0, a)
            elif a == "LogisticRegression":
                total.append((a, LogisticRegression()))
                qtable.learn(b, a, 0, a)

        print(total)

        pl = Pipeline(total)
        pl.fit(train_x, train_y)
        print("score: ", pl.score(test_x, test_y))
        qtable.learn(b, a, pl.score(test_x, test_y), a)



