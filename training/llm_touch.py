# modules/nlp.py
import os
import json
from typing import Dict, Optional
# predict_ivory_classifier.py

import pandas as pd
import joblib



""" -- ------------ This is the testing part of the ML classifier ------------ """

# Load the pipeline
pipeline = joblib.load("elephant_ivory_model.joblib")

# Suppose we have a ex article
new_title = "Recently, Dandong Customs seized a piece of elephant ivory jewellery in the postal channel"
new_summary = "The image of the package was abnormal. After unpacking, it was found that a piece of jewelry was wrapped in layers of aluminum foil and sealed in a snack bag.¬†According to the on-site Raman spectrometer, the jewelry is made of ivory."


wrong_title = "US war plans leak shows Five Eyes allies must ‘look out for ourselves’, says Mark Carney"
wrong_example = "Signal blunder likely to put strain on Five Eyes as it weighs how Trump administration handles classified information"

new_text = new_title + " " + new_summary
wrong_text = wrong_title + " " + wrong_example

# Predict
prediction = pipeline.predict([new_text]) 
prob = pipeline.predict_proba([new_text]) 

print("Predicted right label:", prediction[0])  # e.g., 1 means relevant
print("Probability of right relevance:", prob[0][1])

prediction = pipeline.predict([wrong_text])  # returns array of [1] or [0]
prob = pipeline.predict_proba([wrong_text])  # returns probability scores 

print("Predicted wrong label:", prediction[0])  # e.g., 1 means relevant
print("Probability of wrong relevance:", prob[0][1])

