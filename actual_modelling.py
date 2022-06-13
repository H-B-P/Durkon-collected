import pandas as pd
import numpy as np
import math
import copy
import time

import util
import misc
import calculus
import pena
import rele

def train_model(inputDf, target, nrounds, lr, startingModel, weights=None, ignoreFeats = [], pen=0, specificPenas={}, lossgrad=calculus.Poisson_grad, link=calculus.Unity_link, linkgrad=calculus.Unity_link_grad):
 
 model = copy.deepcopy(startingModel)
 
 if weights==None:
  weights = np.ones(len(inputDf))
 w = np.array(np.transpose(np.matrix(weights)))
 
 cord = rele.produce_cont_relevances_dict(inputDf,model) #d(feat)/d(pt)
 card = rele.produce_cat_relevances_dict(inputDf,model) #d(feat)/d(pt)
 tord = rele.produce_total_relevances_dict(cord, card)
 
 cowrd = rele.produce_wReleDict(cord, w) #d(feat)/d(pt), adjusted for weighting
 cawrd = rele.produce_wReleDict(card, w) #d(feat)/d(pt), adjusted for weighting
 towrd = rele.produce_total_relevances_dict(cowrd, cawrd)
 
 #Interactions . . .
 
 ird = rele.produce_interxn_relevances_dict(inputDf, model) #d(feat)/d(pt)
 tird = rele.produce_total_irelevances_dict(ird)
 
 wird = rele.produce_wReleDict(ird, w) #d(feat)/d(pt), adjusted for weighting
 twird = rele.produce_total_irelevances_dict(wird)
 
 for i in range(nrounds):
  
  print("epoch: "+str(i+1)+"/"+str(nrounds))
  misc.explain(model)
  
  print("initial pred and effect-gathering")
  
  contEffects = misc.get_effects_of_cont_cols_from_relevance_dict(cord,model)
  catEffects = misc.get_effects_of_cat_cols_from_relevance_dict(card,model)
  interxEffects = misc.get_effects_of_interxns_from_relevance_dict(ird,model)
  
  if model['mechanism']=="addl":
   comb = misc.comb_from_effects_addl(model["BASE_VALUE"], len(inputDf), contEffects, catEffects, interxEffects)
  else:
   comb = misc.comb_from_effects_mult(model["BASE_VALUE"], len(inputDf), contEffects, catEffects, interxEffects)
   
  linkgradient = linkgrad(comb) #d(pred)/d(comb)
  
  pred = comb.apply(link)
  lossgradient = lossgrad(pred, np.array(inputDf[target])) #d(Loss)/d(pred)
  
  if "conts" in model:
   print("adjust conts")
   
   for col in [c for c in model['conts'] if c not in ignoreFeats]:
    
    if model['mechanism']=="addl":
     finalGradients = np.matmul(np.array(lossgradient*linkgradient),cowrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
    else:
     effectOfCol = contEffects[col]
     ceoc = comb/effectOfCol #d(comb)/d(feat)
     finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),cowrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
    
    for k in range(len(model['conts'][col])):
     totRele = towrd["conts"][col][k]
     if totRele>0:
      model["conts"][col][k][1] -= finalGradients[k]*lr/totRele
  
  if "cats" in model:
   print("adjust cats")
   
   for col in [c for c in model['cats'] if c not in ignoreFeats]:
    
    if model['mechanism']=="addl":
     finalGradients = np.matmul(np.array(lossgradient*linkgradient),cawrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
    else:
     effectOfCol = catEffects[col]
     ceoc = comb/effectOfCol #d(comb)/d(feat)
     finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),cawrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
    
    skeys = misc.get_sorted_keys(model['cats'][col])
    
    #all the uniques . . .
    for k in range(len(skeys)):
     totRele = towrd["cats"][col][k]
     if totRele>0:
      model["cats"][col]["uniques"][skeys[k]] -= finalGradients[k]*lr/totRele
     
    # . . . and "OTHER"
    totRele = towrd["cats"][col][-1]
    if totRele>0:
     model["cats"][col]["OTHER"] -= finalGradients[-1]*lr/totRele
    
  if "catcats" in model:
   print('adjust catcats')
   
   for cols in [c for c in model['catcats'] if c not in ignoreFeats]:
    
    if model['mechanism']=="addl":
     finalGradients = np.matmul(np.array(lossgradient*linkgradient),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
    else:
     effectOfCols = interxEffects[cols]
     ceoc = comb/effectOfCols #d(comb)/d(feat)
     finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
    
    skeys1 = misc.get_sorted_keys(model['catcats'][cols])
    skeys2 = misc.get_sorted_keys(model['catcats'][cols]["OTHER"])
    
    for i in range(len(skeys1)):
     for j in range(len(skeys2)):
      totRele = twird[cols][i*(len(skeys2)+1)+j]
      if totRele>0:
       model["catcats"][cols]["uniques"][skeys1[i]]['uniques'][skeys2[i]] -= finalGradients[i*(len(skeys2)+1)+j]*lr/totRele
     totRele = twird[cols][i*(len(skeys2)+1)+len(skeys2)]
     if totRele>0:
      model['catcats'][cols]["uniques"][skeys1[i]]['OTHER'] -= finalGradients[i*(len(skeys2)+1)+len(skeys2)]*lr/totRele
   
    for j in range(len(skeys2)):
     totRele = twird[cols][len(skeys1)*(len(skeys2)+1)+j]
     if totRele>0:
      model["catcats"][cols]['OTHER']['uniques'][skeys2[i]] -= finalGradients[len(skeys1)*(len(skeys2)+1)+j]*lr/totRele
     
    totRele = twird[cols][-1]
    if totRele>0:
     model['catcats'][cols]['OTHER']['OTHER'] -= finalGradients[-1]*lr/totRele
  
  if "catconts" in model:
   print('adjust catconts')
  
   for cols in [c for c in model['catconts'] if c not in ignoreFeats]:
    
    if model['mechanism']=="addl":
     finalGradients = np.matmul(np.array(lossgradient*linkgradient),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
    else:
     effectOfCols = interxEffects[cols]
     ceoc = comb/effectOfCols #d(comb)/d(feat)
     finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
    
    skeys = misc.get_sorted_keys(model['catconts'][cols])
    
    for i in range(len(skeys)):
     for j in range(len(model['catconts'][cols]["OTHER"])):
      totRele = twird[cols][i*(len(model['catconts'][cols]["OTHER"])+1)+j]
      if totRele>0:
       model['catconts'][cols]['uniques'][skeys[i]][j][1] -= finalGradients[i*(len(model['catconts'][cols]["OTHER"])+1)+j]*lr/totRele
    
    for j in range(len(model['catconts'][cols]["OTHER"])):
     totRele = twird[cols][len(skeys)*(len(model['catconts'][cols]["OTHER"])+1)+j]
     if totRele>0:
      model['catconts'][cols]['OTHER'][j][1] -= finalGradients[len(skeys)*(len(model['catconts'][cols]["OTHER"])+1)+j]*lr/totRele
  
  if "contconts" in model:
   print('adjust contconts')
   
   for cols in [c for c in model['contconts'] if c not in ignoreFeats]:
    
    if model['mechanism']=="addl":
     finalGradients = np.matmul(np.array(lossgradient*linkgradient),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
    else:
     effectOfCols = interxEffects[cols]
     ceoc = comb/effectOfCols #d(comb)/d(feat)
     finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
    
    for i in range(len(model['contconts'][cols])):
     for j in range(len(model['contconts'][cols][0][1])):
      totRele = twird[cols][i*(len(model['contconts'][cols][0][1])+1)+j]
      if totRele>0:
       model['contconts'][cols][i][1][j][1] -= finalGradients[i*(len(model['contconts'][cols][0][1])+1)+j]*lr/totRele
  
  #Penalize!
  print("penalties")
  if model['mechanism']=="addl":
   model = pena.penalize_model(model, pen, 0, specificPenas)
  else:
   model = pena.penalize_model(model, pen, 1, specificPenas)
 
 return model




if False: #__name__ == '__main__':
 df =  pd.DataFrame({"cat1":['a','a','a','a','b'],'cat2':['d','c','d','c','c'],"y":[1,1,1,1,2]})
 model = {"BASE_VALUE":1.2, 'conts':{}, 'cats':{}}
 err = misc.predict(df, model) - df["y"]
 print(err)
 print(audition_this_cat({"OTHER":1,"uniques":{"a":1, "b":1}}, df['cat1'], err))
 print(audition_this_cat({"OTHER":1,"uniques":{"c":1, "d":1}}, df['cat2'], err))
 
 df =  pd.DataFrame({"cont1":[1,2,3,4,5],'cont2':[1,2,3,4,3],'cont3':[1,2,3,2,1],"y":[1,2,3,4,5]})
 model = {"BASE_VALUE":3, 'conts':{}, 'cats':{}}
 err = misc.predict(df, model) - df["y"]
 print(err)
 print(audition_this_cont([[1,1],[5,1]], df['cont1'], err))
 print(audition_this_cont([[1,1],[4,1]], df['cont2'], err))
 print(audition_this_cont([[1,1],[3,1]], df['cont3'], err))
 
 df =  pd.DataFrame({"cat1":['a','a','a','a','b','b','b','b'], 'cat2':['c','c','d','d','c','c','d','d'], 'cat3':['e','f','e','f','e','f','e','f'], "y":[1,1,1,1,1,1,2,2]})
 model = {"BASE_VALUE":1.25, 'conts':{}, 'cats':{}}
 err = misc.predict(df, model) - df["y"]
 print(err)
 print(audition_this_catcat({"OTHER":{"OTHER":1,"uniques":{"c":1, "d":1}},"uniques":{"a":{"OTHER":1,"uniques":{"c":1, "d":1}}, "b":{"OTHER":1,"uniques":{"c":1, "d":1}}}}, df['cat1'], df['cat2'], err))
 print(audition_this_catcat({"OTHER":{"OTHER":1,"uniques":{"e":1, "f":1}},"uniques":{"c":{"OTHER":1,"uniques":{"e":1, "f":1}}, "d":{"OTHER":1,"uniques":{"e":1, "f":1}}}}, df['cat2'], df['cat3'], err))
 print(audition_this_catcat({"OTHER":{"OTHER":1,"uniques":{"e":1, "f":1}},"uniques":{"a":{"OTHER":1,"uniques":{"e":1, "f":1}}, "b":{"OTHER":1,"uniques":{"e":1, "f":1}}}}, df['cat1'], df['cat3'], err))
 
 df =  pd.DataFrame({"cat1":['a','a','a','a','b','b','b','b'], 'cat2':['c','c','d','d','c','c','d','d'], 'cont1':[1,2,1,2,1,2,1,2], 'cont2':[1,1000,1,1000,1,1000,1,1000], "y":[1,1,1,1,1,2,1,2]})
 model = {"BASE_VALUE":1.25, 'conts':{}, 'cats':{}}
 err = misc.predict(df, model) - df["y"]
 print(err)
 print(audition_this_catcont({"OTHER":[[1,1],[2,1]],"uniques":{"a":[[1,1],[2,1]], "b":[[1,1],[2,1]]}}, df['cat1'], df['cont1'], err))
 print(audition_this_catcont({"OTHER":[[1,1],[2,1]],"uniques":{"c":[[1,1],[2,1]], "d":[[1,1],[2,1]]}}, df['cat2'], df['cont1'], err))
 
 print(audition_this_catcont({"OTHER":[[1,1],[2,1]],"uniques":{"a":[[1,1],[2,1]], "b":[[1,1],[2,1]]}}, df['cat1'], df['cont2'], err))
 
 
 
 df =  pd.DataFrame({"cont1":[1,1,1,1,2,2,2,2], 'cont2':[1,1,2,2,1,1,2,2], 'cont3':[1,2,1,2,1,2,1,2], "y":[1,1,1,1,1,1,2,2]})
 model = {"BASE_VALUE":1.25, 'conts':{}, 'cats':{}}
 err = misc.predict(df, model) - df["y"]
 print(err)
 print(audition_this_contcont([[1,[[1,1],[2,1]]],[2,[[1,1],[2,1]]]], df['cont1'], df['cont2'], err))
 print(audition_this_contcont([[1,[[1,1],[2,1]]],[2,[[1,1],[2,1]]]], df['cont2'], df['cont3'], err))
 print(audition_this_contcont([[1,[[1,1],[2,1]]],[2,[[1,1],[2,1]]]], df['cont1'], df['cont3'], err))















if __name__ == '__main__':
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q'],'cat2':['c','d','q','c','d','q','c','d','q','q'],"y":[1,2,3,4,5,6,7,8,9,10]})
 model = {"BASE_VALUE":1.0,"conts":{}, "cats":{"cat1":{"uniques":{"a":1,"b":1,},"OTHER":1}, "cat2":{"uniques":{"c":1,"d":1},"OTHER":1}}, 'catcats':{'cat1 X cat2':{'uniques':{'a':{'uniques':{'c':1,'d':1},'OTHER':1},'b':{'uniques':{'c':1,'d':1},'OTHER':1}},'OTHER':{'uniques':{'c':1,'d':1},'OTHER':1}}}, 'catconts':{}, 'contconts':{}}
 print(rele.produce_catcat_relevances(df['cat1'], df['cat2'], model["catcats"]["cat1 X cat2"]))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q'],'cat2':['c','d','q','c','d','q','c','d','q','q'],"y":[1,2,3,4,5,6,7,8,9,10]})
 model = {"BASE_VALUE":1.0,"conts":{}, "cats":{"cat1":{"uniques":{"a":1,"b":1,},"OTHER":1}, "cat2":{"uniques":{"c":1,"d":1},"OTHER":1}}, 'catcats':{'cat1 X cat2':{'uniques':{'a':{'uniques':{'c':1,'d':1},'OTHER':1},'b':{'uniques':{'c':1,'d':2},'OTHER':1}},'OTHER':{'uniques':{'c':1,'d':1},'OTHER':1}}}, 'catconts':{}, 'contconts':{}}
 reles = rele.produce_catcat_relevances(df['cat1'], df['cat2'], model["catcats"]["cat1 X cat2"])
 print(misc.get_effect_of_this_catcat_from_relevances(reles, model, "cat1 X cat2"))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q'],'cat2':['c','d','q','c','d','q','c','d','q','q'],"y":[1,1,1,1,2,1,1,1,1,1]})
 model = {"BASE_VALUE":1.0,"conts":{}, "cats":{"cat1":{"uniques":{"a":1,"b":1,},"OTHER":1}, "cat2":{"uniques":{"c":1,"d":1},"OTHER":1}}, 'catcats':{'cat1 X cat2':{'uniques':{'a':{'uniques':{'c':1,'d':1},'OTHER':1},'b':{'uniques':{'c':1,'d':1},'OTHER':1}},'OTHER':{'uniques':{'c':1,'d':1},'OTHER':1}}}, 'catconts':{}, 'contconts':{}}
 
 newModel = train_model(df, "y",50, 0.4, model, ignoreFeats = ['cat1','cat2'])
 print(newModel)
 
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q','q'],'cont1':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 model = {"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]]}, "cats":{"cat1":{"uniques":{"a":1,"b":1},"OTHER":1}}, 'catcats':{}, 'catconts':{"cat1 X cont1":{"uniques":{"a":[[1,1],[2,1],[3,1]],"b":[[1,1],[2,1],[3,1]]},"OTHER":[[1,1],[2,1],[3,1]]}}, 'contconts':{}}
 print(rele.produce_catcont_relevances(df['cat1'], df['cont1'], model["catconts"]["cat1 X cont1"]))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q','q'],'cont1':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 model = {"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]]}, "cats":{"cat1":{"uniques":{"a":1,"b":1},"OTHER":1}}, 'catcats':{}, 'catconts':{"cat1 X cont1":{"uniques":{"a":[[1,1],[2,1],[3,1]],"b":[[1,1],[2,4],[3,1]]},"OTHER":[[1,1],[2,1],[3,1]]}}, 'contconts':{}}
 reles = rele.produce_catcont_relevances(df['cat1'], df['cont1'], model["catconts"]["cat1 X cont1"])
 print(misc.get_effect_of_this_catcont_from_relevances(reles, model, "cat1 X cont1"))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q','q'],'cont1':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,1,1,1,1,2,1,1,1,1,1]})
 model = {"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]]}, "cats":{"cat1":{"uniques":{"a":1,"b":1},"OTHER":1}}, 'catcats':{}, 'catconts':{"cat1 X cont1":{"uniques":{"a":[[1,1],[2,1],[3,1]],"b":[[1,1],[2,1],[3,1]]},"OTHER":[[1,1],[2,1],[3,1]]}}, 'contconts':{}}
 newModel = train_model(df, "y",50, 0.4, model, ignoreFeats = ['cat1','cont1'])
 print(newModel)
 
 df = pd.DataFrame({"cont1":[1,1,1,2,2,2,3,3,3,1.5,np.nan],'cont2':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 model = {"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]],'cont2':[[1,1],[2,1],[3,1]]}, "cats":{}, 'catcats':{}, 'catconts':{}, 'contconts':{'cont1 X cont2': [[1,[[1,1],[2,1],[3,1]]],[2,[[1,1],[2,1],[3,1]]],[3,[[1,1],[2,1],[3,1]]]]} }
 print(rele.produce_contcont_relevances(df['cont1'], df['cont2'], model["contconts"]["cont1 X cont2"]))
 
 df = pd.DataFrame({"cont1":[1,1,1,2,2,2,3,3,3,1.5,np.nan],'cont2':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 models = [{"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]],'cont2':[[1,1],[2,1],[3,1]]}, "cats":{}, 'catcats':{}, 'catconts':{}, 'contconts':{'cont1 X cont2': [[1,[[1,1],[2,1],[3,1]]],[2,[[1,1],[2,1],[3,1]]],[3,[[1,1],[2,1],[3,5]]]]} }]
 reles = rele.produce_contcont_relevances(df['cont1'], df['cont2'], model["contconts"]["cont1 X cont2"])
 print(misc.get_effect_of_this_contcont_from_relevances(reles, models[0], "cont1 X cont2"))
 
 df = pd.DataFrame({"cont1":[1,1,1,2,2,2,3,3,3,1.5,np.nan],'cont2':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,1,1, 1,1,1, 1,1,2, 1,1]})
 model = {"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]],'cont2':[[1,1],[2,1],[3,1]]}, "cats":{}, 'catcats':{}, 'catconts':{}, 'contconts':{'cont1 X cont2': [[1,[[1,1],[2,1],[3,1]]],[2,[[1,1],[2,1],[3,1]]],[3,[[1,1],[2,1],[3,1]]]]} }
 newModel = train_model(df, "y",50, 0.4, model, ignoreFeats = ['cont1','cont2'])
 print(newModel)
 
if False:
 df = pd.DataFrame({"x":[1,2,3,4,5,6,7,8,9],"y":[1,2,3,4,5,6,7,8,9]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[1,5],[5,5],[9,5]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [2], models,weights=[1,2,3,4,5,6,7,8,9])
 for newModel in newModels:
  misc.explain(newModel)


if False:
 df = pd.DataFrame({"x":[1,2,3,4,5,6,7,8,9],"y":[2,2,2,2,2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[2,1],[5,1],[8,1]]}, "cats":[]}]
 print(produce_cont_relevances(df, models[0], "x"))
 
 df = pd.DataFrame({"x":["cat","dog","cat","mouse","rat"],"y":[2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"x":{"uniques":{"cat":1,"mouse":1,"dog":1},"OTHER":1}}}]
 print(produce_cat_relevances(df, models[0], "x"))
 
 df = pd.DataFrame({"x":[1,2,3,4,5,6,7,8,9],"y":[2,2,2,2,2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[2,1],[5,2],[8,1]]}, "cats":[]}]
 reles = produce_cont_relevances(df, models[0], "x")
 print(misc.get_effect_of_this_cont_col_from_relevances(reles, models[0], "x"))
 
 df = pd.DataFrame({"x":["cat","dog","cat","mouse","rat"],"y":[2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"x":{"uniques":{"cat":2,"mouse":1,"dog":3},"OTHER":1.5}}}]
 reles = produce_cat_relevances(df, models[0], "x")
 print(misc.get_effect_of_this_cat_col_from_relevances(reles, models[0], "x"))
 
 df = pd.DataFrame({"x":[1,2,3],"y":[2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[0,1],[4,1]]}, "cats":[]}]
 newModels = train_model(df, "y",50, [2.0], models)
 for newModel in newModels:
  misc.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2,3],"y":[2,3,4]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[0,1],[4,1]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [2.0], models)
 for newModel in newModels:
  misc.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2],"y":[2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[1,1],[2,1]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [2.0], models)
 for newModel in newModels:
  misc.explain(newModel)
 
 df = pd.DataFrame({"x":[1, 995,996,997,998,999,1000],"y":[2,2,2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[1,1],[1000,1]]}, "cats":[]}]
 newModels = train_model(df, "y",50, [1.5], models)
 for newModel in newModels:
  misc.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2,3,4],"y":[1+1,2+1.2,3+1.4,4+1.6]})
 models = [{"BASE_VALUE":2,"conts":{"x":[[1,1],[4,1]]}, "cats":[]},{"BASE_VALUE":1,"conts":{"x":[[1,1],[4,1]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [1.5, 1.5], models)
 for newModel in newModels:
  misc.explain(newModel)

 df = pd.DataFrame({"x":["cat","dog","cat","mouse","rat"],"y":[2.0,3.0,2.0,1.0,1.5]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"x":{"uniques":{"cat":1,"mouse":1,"dog":1},"OTHER":1}}}]
 newModels = train_model(df, "y",100, [0.5], models)
 for newModel in newModels:
  misc.explain(newModel)
 
