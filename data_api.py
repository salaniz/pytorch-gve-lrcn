import sys, os
import codecs
import json
import pickle as pkl
from collections import defaultdict
import random

class DataApi:
    # TODO figure out how to do this for test images
    def __init__(self, location="./data/cub/"):        
        self.description_test = self.get_descriptions(os.path.join(location, "descriptions_bird.train.fg.json"))
        self.data = []
        self.classes = defaultdict(list)
        self.images = {}

        annotations = {}

        for annotation in self.description_test["annotations"]:
            annotations[annotation["image_id"]] = annotation["caption"]

        for index, image in enumerate(self.description_test["images"]):
            class_label = image["id"].split(".")[1].split("/")[0]
            
            self.classes[class_label].append(index)
        
            image_index = "_".join(image["file_name"].split("/")[1].split("_")[:-1])

            self.data.append({
                "class_label": class_label,
                "id": image["id"],
                "path": os.path.join(location, "images",  image["file_name"]),
                "caption": annotations[image["id"]]
            })

            assert os.path.exists(self.data[-1]["path"])
        
        

    def get_descriptions(self, path):
        with codecs.open(path, "r", "utf-8") as reader:
            return json.load(reader)

    def get_classes(self):
        return [k for k in self.classes.keys()]

    def sample_class(self, klass):
        idx = random.choice(self.classes[klass])
        return self.data[idx]