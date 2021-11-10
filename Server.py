import socket
import struct
import pandas as pd
import numpy as np


def get_img(client):
    with open('tst.jpg', 'wb') as img:
        while True:
            data = client.recv(1024)
            if not data:
                break
            img.write(data)
    return img

df = pd.read_csv("plants_info.csv")
df.set_index("name", inplace=True)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 8080))
s.listen(3)


while True:
    client_socket, address = s.accept()
    print("connected")
    data = client_socket.recv(3) # maybe needs to be 2
    data_type = data.decode("utf-8")
    data_type = data_type[2:]
    print("Here", type(data_type), data, data_type)

    # buf = ''
    # while len(buf) < 4:
    #     buf += client_socket.recv(4 - len(buf))
    # size = struct.unpack('!i', buf)
    print("got data_type: ", data_type)
    name = ""
    if data_type == 'f':
        img = get_img(client_socket)
        # give img to model and get back name
        name = "plant"
    elif data_type == 's':
        data = client_socket.recv(1024)
        name = data.decode()
    plant_info_to_send = ""
    print("got name:", name, type(name))
    if name != "":
        print("=======================")#df.loc[name])
        plant_info = df.loc[name]
        print("Here 2", plant_info[0], type(plant_info))
        plant_info_to_send = plant_info.to_string(header=False, index=False)
        plant_info_to_send = plant_info_to_send.replace(" ", "")
        print("Here 3", plant_info_to_send)
    client_socket.send(bytes(plant_info_to_send, "utf-8"))
    client_socket.close()