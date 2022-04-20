import flask
import werkzeug
from flask import Flask, request
from flask_restful import Api, Resource
import pandas as pd
import numpy as np
import tensorflow as tf
import os

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
df = pd.read_csv("plants_info.csv")
df.set_index("type", inplace=True)
print("Loading model")
# loaded_model = tf.saved_model.load("model.pb")  # with save
# loaded_model = tf.keras.Sequential([hub.KerasLayer("model.pb",trainable=False),
#                             tf.keras.layers.Dense(512, activation='relu'),
#                             tf.keras.layers.Dense(5, activation='softmax')])
# loaded_model.build([None, 224, 224, 3])




loaded_model = tf.keras.models.load_model(
    "mobileNet_model"+ os.sep +"inaturalist1640537367")#"model.pb")
labels = np.load("labels.npy")
print(labels)

# print(labels)
# print("model loaded")
# print(type(loaded_model))

app = Flask(__name__)
api = Api(app)
pd.options.display.max_colwidth = 1000

def get_data_from_name(name):
    if name not in df.index:
        return dict()
    plant_info = df.loc[name]
    # print("fip",plant_info,"fin",sep='\n')
    plant_info_to_send = plant_info.to_string(header=False, index=False)
    # plant_info_to_send = plant_info_to_send.replace(" ", "")
    plant_info_to_send = plant_info_to_send.split("\n")
    # print(plant_info_to_send)
    # print(plant_info_to_send[3])
    return {"type": name,
            "desc": plant_info_to_send[0], #.replace(" ",""),
            "water": plant_info_to_send[1], #.replace(" ",""),
            "sun": plant_info_to_send[2], #.replace(" ",""),
            "soil": plant_info_to_send[3],
            "alarm": plant_info_to_send[4].replace(" ","")}

@app.route('/')
def index():
   print('Request for index page received')
   return {"bdika":"works"}


@app.route('/search/<string:searchKey>', methods=['GET', 'POST'])
def handle_search_request(searchKey):
    if request.method == 'GET':
        return get_data_from_name(searchKey)
    else:
        return "Problem on server side"



def prep_image(img):
    img_height = 224
    img_width = 224
    pic = tf.keras.utils.image_dataset_from_directory(img,
                                                      # labels=None,
                                                      batch_size=1,
                                                      image_size=
                                                      (img_height, img_width)
                                                      )

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    pic = pic.map(
        lambda x, y: (normalization_layer(x), y))

    for test_batch, test_labels_batch in pic:
        print(test_batch.shape)
        print(test_labels_batch.shape)
        break

    predicted_test_labeld = loaded_model.predict(test_batch)
    # print(predicted_test_labeld)
    ind = np.argmax(predicted_test_labeld[0])
    return labels[ind]

# def get_label(img):
#     predicted_odds = loaded_model.predict(numpy.array([img]))[0]
#     print(predicted_odds, labels, sep='\n')
#     # predicted_odds = np.abs(predicted_odds)
#     ind = predicted_odds.argmax(axis=0)
#     return labels[ind]

@app.route('/recognize/', methods=['GET', 'POST'])
def handle_recognize_request():
    if request.method == 'POST':
        imagefile = flask.request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        print("\nReceived image File name : " + imagefile.filename)
        imagefile.save("to_predict" + os.sep + "place_holder"+ os.sep
                       + filename) # was
        # just "
                       # "filename
        # img = imageio.imread(filename)
        # img = process_image(img)
        # imagefile.save(imageio.imwrite("img.jpg",img))
        img_lbl = prep_image("to_predict")
        # predicted_label = get_label(img_lbl)
        print(img_lbl)

        return get_data_from_name(img_lbl)
    else:
        return 'problem happened on server side'

# api.add_resource(Search, "/search/<string:searchKey>")


if __name__ == "__main__":
    app.run()#debug=False, host='0.0.0.0') # for connecting from WAN (dont
    # forget to open ports on router!!!
