

class ExplanationModel:

    def generate_explanation(self, image):
        return image["caption"]
    
class CounterFactualExplanationModel:

    def generate_counterfactual_explanation(self, image):
        return image["caption"]