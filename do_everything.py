import pandas as pd
import numpy as np

import actual_modelling
import prep
import misc
import calculus

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,2], "cat1":['a','a','a','a','b','b','b','b'], "cat2":['c','c','d','d','c','d','e','d'], "y":[0,0,0,1,0,0,0,1]})

model = prep.prep_model(df, "y", ["cat1","cat2"],["cont1","cont2"], defaultValue=0)

#Classifier-ify model
model["BASE_VALUE"] = calculus.Logit_delink(model["BASE_VALUE"])
model["mechanism"] = "addl"

print(model)

model = actual_modelling.train_model(df, "y", 500, 0.1, model, link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, pen=0)

print(model)

print(misc.predict(df,model, "Logit"))