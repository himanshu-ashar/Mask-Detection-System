import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np



class MaskDetectionModel(object):

    MASK_LIST = ["Mask","No Mask"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict_mask(self, img):
        img = img/255.0
        #img = np.expand_dims(img, axis=0)
        self.preds = self.loaded_model.predict(img)
        score = self.preds[0]
        if score>0.5:
            return("No Mask: %.3f percent"%(100*score))
        else:
            return("Mask: %.3f percent"%(100*(1-score)))
        # return MaskDetectionModel.MASK_LIST[np.argmax(self.preds)]
