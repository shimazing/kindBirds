#!/usr/bin/env python3
import socket
import json
from PIL import Image

class Environment(object):
    def __init__(self):
        self.svrsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.create()
        #self.connect_client()

    def create(self):
        self.svrsock.bind(("127.0.0.1", 9090))
        self.svrsock.listen(1)
        print("#### {:^20} ####".format("MAKE SERVER"))

    def connect_client(self):
        self.conn, self.addr = self.svrsock.accept()
        print("#### {:^20} ####".format("CONNECTED"))

    def get_state_reward(self):
        data = self.conn.recv(1024).decode('utf-8')
        print(data)
        data_dict = json.loads(data)
        gamestate = eval(data_dict["gamestate"])
        reward = eval(data_dict["reward"]) / 10000.
        terminal = (gamestate != 0)
        if not terminal:
            state = np.asarray(Image.open("../screenshot.png"))
            state = state / 255.
            n_birds = eval(data_dict["birds"])
            birdtype = eval(data_dict["birdtype"]) # from 0 to 2
        else:
            state = n_birds = birdtype = None
        print("Get state and reward")
        return terminal, reward, state, birdtype, n_birds

    '''
    def get_state_reward(self):
        data = self.conn.recv(1024)
        data_dict = json.loads(data)
        gamestate = data_dict[u"gamestate"][0]
        reward = data_dict[u"reward"][0]
        terminal = gamestate != 0
        if not terminal:
            state = np.array(data_dict[u"screenshow"])
            state = state.astype(int)
            state_R = (state >> 16) & 0xFF
            state_G = (state >> 8) & 0xFF
            state_B = (state) & 0xFF
            state = np.stack([state_R, state_G, state_B], axis=2)
            state = state.reshape(105, 60) / 255.
        print("Get state and reward")
        print(reward)
        print(gamestate)
    '''

    def act(self, theta, v):
        action = {'theta': str(theta), 'v': str(v)}
        action = json.dumps(action)
        self.conn.send(action + '\n')
        print("Send Action", action)

    '''
    def act_random(self):
        angle = np.arange(0.025, 0.5, 0.025) * math.pi
        tap = np.arange(0, 4000, 200)
        action_str = json.dumps({"angle": str(angle[10]), "taptime": str(tap[10])})
        self.conn.send(action_str)

        print("Send Action", action_str)
    '''

def main():
    env = Environment()
    while True:
        env.get_state_reward()
        env.act_random()
    #client = Client()
    #client.create()
    #client.send_data(data)

if __name__ == '__main__':
    main()
