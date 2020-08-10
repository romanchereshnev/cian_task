from flask import Flask, request, jsonify
from marshmallow import Schema, fields, post_load
import json
import string
from fasttext_classifier import FastTextClassifier
from numpy import dot
from numpy.linalg import norm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

ft = FastTextClassifier(epoch=10, dim=100)
ft.load("ft_model.bin")

with open("model.pkl", "rb") as f:
    clf: RandomForestClassifier = pickle.load(f)

class Pairs(Schema):
    offer_id_x = fields.String()
    description_x = fields.String()
    geo_x = fields.String()
    building_x = fields.String()
    roomscount_x = fields.Float(allow_nan=True)
    floornumber_x = fields.Float(allow_nan=True)
    category_x = fields.String()
    totalarea_x = fields.Float()
    userid_x = fields.String()
    publisheduserid_x = fields.String()
    flattype_x = fields.String()
    bargainterms_x = fields.String()

    offer_id_y = fields.String()
    description_y = fields.String()
    geo_y = fields.String()
    building_y = fields.String()
    roomscount_y = fields.Float(allow_nan=True)
    floornumber_y = fields.Float(allow_nan=True)
    category_y = fields.String()
    totalarea_y = fields.Float()
    userid_y = fields.String()
    publisheduserid_y = fields.String()
    flattype_y = fields.String()
    bargainterms_y = fields.String()


    geo = ['userInput']
    building = ['floorsCount', 'totalArea']
    bargainterms = ['price']

    @post_load
    def create_data(self, data, many, **kwargs):
        flat = ['description', 'roomscount', 'floornumber', 'category', 'totalarea', 'flattype', 'userid',
                'publisheduserid']

        x = [data[f + "_x"] for f in flat]
        y = [data[f + "_y"] for f in flat]

        x[0] = " ".join(x[0].lower().translate(str.maketrans('', '', string.punctuation)).split())
        y[0] = " ".join(y[0].lower().translate(str.maketrans('', '', string.punctuation)).split())

        def get_geo_data(data, index):
            try:
                geo = json.loads(
                    data["geo" + index].replace("'", '"').replace('\n', "").replace("True", "\"True\"").replace("False",
                                                                                                          "\"False\""))
            except Exception as e:
                user_input = ""
            else:
                user_input = geo['userInput'].lower().translate(str.maketrans('', '', string.punctuation))
            return user_input

        user_input_x = get_geo_data(data, "_x")
        user_input_y = get_geo_data(data, "_y")

        x.append(user_input_x)
        y.append(user_input_y)

        def get_building_data(data, index):
            try:
                building = json.loads(data["building" + index].replace("'", '"').replace('\n', ""))
            except Exception as e:
                floors_count = 0
                total_area = 0
            else:
                floors_count = building['floorsCount'] if 'floorsCount' in building else 0
                total_area = building['totalArea'] if 'totalArea' in building else 0
            return floors_count, total_area

        floors_count_x, total_area_x = get_building_data(data, index="_x")
        floors_count_y, total_area_y = get_building_data(data, index="_y")

        x.append(floors_count_x)
        x.append(total_area_x)

        y.append(floors_count_y)
        y.append(total_area_y)

        def get_price(data, index):
            try:
                bargainterms = json.loads(data["bargainterms"+index].replace("'", '"').replace('\n', ""))
            except Exception as e:
                price = 0
            else:
                price = bargainterms['price']
            return price

        price_x = get_price(data, "_x")
        price_y = get_price(data, "_y")

        x.append(price_x)
        y.append(price_y)

        return [x, y]


def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))


def create_data(X_pairs, ft):
    X = []

    for ix, x in enumerate(X_pairs):
        dis1 = x[0][0]
        dis2 = x[1][0]
        name_cos = cos_sim(ft.transform(dis1), ft.transform(dis2))
        addr_cos = cos_sim(ft.transform(x[0][8]), ft.transform(x[1][8]))
        same_roomscount = int(x[0][1] == x[1][1])
        same_floornumber = int(x[0][2] == x[1][2])
        same_category = int(x[0][3] == x[1][3])
        same_totalarea = int(x[0][4] == x[1][4] )
        same_flattype = int(x[0][5] == x[1][5] )
        same_userid = int(x[0][6] == x[1][6] )
        same_publisheduseri = int(x[0][7] == x[1][7])
        diff_floorsCount = np.abs(x[0][9] - x[1][9])
        diff_totalArea = np.abs(x[0][10] - x[1][10])
        diff_price = np.abs(x[0][11] - x[1][11])

        X.append([name_cos, addr_cos, same_roomscount, same_floornumber, same_category, same_totalarea, same_flattype,
    same_userid, same_publisheduseri, diff_floorsCount, diff_totalArea, diff_price])
    return X


@app.route('/predict', methods=['POST'])
def index():
    json_data = request.json
    schema = Pairs(many=True)
    pairs = schema.load(json_data)

    x = create_data(pairs, ft)
    proba = clf.predict_proba(x)
    return jsonify(proba[:, 1].tolist()), 200


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
