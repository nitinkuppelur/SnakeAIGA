from random import randint
import numpy as np

class DNA:
    def __init__(self, num_states = 3, num_action=3, action_val = [0,0]):
        self.num_states = num_states
        self.num_action = num_action
        self.action_val = action_val
        self.create_mask()
        self.Q_table_init()

    def create_mask(self):
        self.masks = []
        self.masks.append(1)
        for i in range(1, self.num_states):
            self.masks.append(self.masks[i-1]*2)

    def Q_table_init(self):
        self.Q_table = {}
        for i in range(0, 2**self.num_states):
            state = []
            for mask in reversed(self.masks):
                state.append(bool(i&mask))
            state = np.array(state)
            self.Q_table.update({state.tobytes():self.get_random_action()})

    def get_random_action(self):
        actions = []
        for i in range(0,self.num_action):
            actions.append(randint(self.action_val[0],self.action_val[1]))
        return actions

    def reward_action(self, state, action, reward):
        self.Q_table[state][action] += reward
        if self.Q_table[state][action] > 1000:
            self.Q_table[state][action] = 1000
            
    def penalise_action(self, state, action, penalty):
        self.Q_table[state][action] -= penalty
        if self.Q_table[state][action] < -1000:
            self.Q_table[state][action] = -1000

    def cross_dna(self, dna):
        new_dna = {}
        for state in self.Q_table:
            choice = randint(0,1)
            if choice == 0:
                new_dna.update({state:self.Q_table[state]})
            if choice == 1:
                new_dna.update({state:dna.Q_table[state]})

if __name__ == "__main__":
    dna = DNA(3,3,[-5,5])
    dna2 = DNA(3,3,[-5,5])
    dna.cross_dna(dna2)
