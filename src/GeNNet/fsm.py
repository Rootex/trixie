# coding=utf-8


class Machine:
    def __init__(self, states, transition, init):
        self.states = states
        self.transition = transition
        self.current_state = init

    def advance(self, trigger):
        pass

if __name__ == "__main__":
    states = []
    transitions = [{}]
    machine = Machine(states=states, transition=transitions, init="init")
    print(machine.current_state)