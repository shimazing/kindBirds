#!/usr/bin/env python3
import socket
import json
import struct
from PIL import Image
import numpy as np
import math

class Environment(object):
    def __init__(self):
        self.svrsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.create()
        self.connect_client()

    def create(self):
        self.svrsock.bind(("127.0.0.1", 9090))
        self.svrsock.listen(1)
        print("#### MAKE SERVER ####")

    def connect_client(self):
        self.conn, self.addr = self.svrsock.accept()
        print("####   CONNECTED   ####")
        '''
        print(self.conn, self.addr)
        data = json.loads(self.conn.recv(1024))

        print(data.keys()[0].encode("utf-8"))
        print(data.values())
        self.conn.send(data)
        self.conn.close()
        return data
        '''
    def get_state_reward(self):
        data = self.conn.recv(1024)
        print(data)
        print("adf")
        data_dict = json.loads(data)
        gamestate = eval(data_dict[u"gamestate"])
        reward = eval(data_dict[u"reward"])
        terminal = gamestate != 0
        if not terminal:
            state = np.asarray(Image.open("screenshot.png"))
            state = state / 255.
        print("Get state and reward")
        print(reward)
        print(gamestate)



        # data.keys()[0]
        # return state, reward

    def act(self, theta, v):
        action = {'theta': theta, 'v': v}
        action = json.dumps(action)
        self.conn.sendall(action)

    def act_random(self):
        angle = np.arange(0.025, 0.5, 0.025) * math.pi
        tap = np.arange(0, 4000, 200)
        action_str = json.dumps({"angle": str(angle[10]), "taptime": str(tap[10])})
        self.conn.send(action_str+"\n")
        print("Send Action", action_str)


'''
class Client(object):
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def create(self):
        self.sock.gethostname()
        self.sock.connect(("127.0.0.1", 8080))

    def send_data(self, data):
        self.sock.send(data)
        self.sock.close()
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
