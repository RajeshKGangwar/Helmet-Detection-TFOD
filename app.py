from flask import Flask, request, jsonify, Response
import os
from Preprocessing import EncodeDecode
from Prediction import ImagePredict


app = Flask(__name__)


class App:
    def __init__(self):
        self.filename = "InputImage.jpg"
        modelpath = "research/ssd_mobilenet_v1_coco_2017_11_17"
        self.modelobj = ImagePredict(self.filename,modelpath)



@app.route("/predict", methods=["POST"])
def predict():
    try:

        image = request.json["image"]
        EncodeDecode.Decodetobase64(image,mainapp.filename)
        result = mainapp.modelobj.FinalPrediction()

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)



mainapp = App()
if __name__ == '__main__':


    app.run(debug=True)

