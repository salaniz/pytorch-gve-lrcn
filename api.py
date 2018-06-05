from flask import Flask
from flask_restful import Resource, Api
from flask import url_for
import os
from data_api import DataApi
from model_api import ExplanationModel, CounterFactualExplanationModel


app = Flask(__name__, static_folder="data")
api = Api(app)


data_api = DataApi()
explanation_model = ExplanationModel()
cf_explanation_model = CounterFactualExplanationModel()

class AvailableClassesResource(Resource):
    def get(self):
        return data_api.get_classes()

class CounterFactualResource(Resource):
    def get(self, class_true, class_false):
        true_image = data_api.sample_class(class_true)
        false_image = data_api.sample_class(class_false)
        self.fill_image(true_image, counter_factual=False)
        self.fill_image(false_image, counter_factual=True)

        return {
            "class_true": class_true,
            "class_false": class_false,
            "images": [
                true_image, false_image
            ]
        }
    
    def fill_image(self, image, counter_factual=False):
        image["cf_explanation"] = cf_explanation_model.generate_counterfactual_explanation(image)
        image["explanation"] = explanation_model.generate_explanation(image)

        path = os.path.join(*image["path"].split("/")[2:])
        del image["path"]
        image["url"] = url_for('static', filename=path)
                
    

api.add_resource(AvailableClassesResource, '/classes')
api.add_resource(CounterFactualResource, '/counter_factual/<string:class_true>/<string:class_false>')

if __name__ == '__main__':    
    app.run(debug=False)
