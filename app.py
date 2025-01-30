from flask import Flask, request
from flask_restful import Api, Resource
from flask_cors import CORS
from handler import handle_verification
import json

app = Flask(__name__)
cors = CORS(app,  resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)


class PricePre(Resource):
    def post(self):
        price = handle_verification(json.loads(request.data), [])
        return price


api.add_resource(PricePre, '/cross_verification')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
