from flask import  Flask, request
from flask_restful import Api, Resource
import pandas as pd
import numpy as np

df = pd.read_csv("plants_info.csv")
df.set_index("name", inplace=True)


app = Flask(__name__)
api = Api(app)

class Search(Resource):

    def get(self, searchKey):
        if searchKey not in df.index:
            print("in")
            return dict()
        plant_info = df.loc[searchKey]
        # print(plant_info)
        # print("Here 2", plant_info[0], type(plant_info))
        plant_info_to_send = plant_info.to_string(header=False, index=False)
        plant_info_to_send = plant_info_to_send.replace(" ", "")
        plant_info_to_send = plant_info_to_send.split("\n")
        print("Here 3\n", plant_info_to_send)
        return {"times":plant_info_to_send[0], "in":plant_info_to_send[1],
                "sun":plant_info_to_send[2]}


api.add_resource(Search, "/search/<string:searchKey>")

if __name__ == "__main__":
    app.run(debug=True)