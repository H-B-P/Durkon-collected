import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly

def get_cont_pdp_prevalences(df, col, intervals=10, weightCol=None):
 cdf= df.copy()
 
 if type(intervals)==type([1,2,3]):
  intervs=intervals
 else:
  gap=(max(df[col])-min(df[col]))/float(intervals)
  print(min(df[col]), max(df[col]), gap)
  intervs=list(np.arange(min(df[col]), max(df[col])+gap, gap))
 
 if weightCol==None:
  cdf["weight"]=1
 else:
  cdf["weight"]=cdf[weightCol]
 
 prevs=[]
 
 for i in range(len(intervs)-1):
  loInt = intervs[i]
  hiInt = intervs[i+1]
  if i==(len(intervs)-2):
   prevs.append(sum(cdf[(cdf[col]<=hiInt) & (cdf[col]>=loInt)]["weight"]))
  else:
   prevs.append(sum(cdf[(cdf[col]<hiInt) & (cdf[col]>=loInt)]["weight"]))
 
 return intervs, prevs

def get_cat_pdp_prevalences(df, col, threshold=0.05, weightCol=None):
 cdf= df.copy()
 
 if weightCol==None:
  cdf["weight"]=1
 else:
  cdf["weight"]=cdf[weightCol]
 
 uniques = pd.unique(cdf[col])
 opDict = {}
 totalWeight = float(sum(cdf["weight"]))
 
 for unique in uniques:
  specWeight = float(sum(cdf[cdf[col]==unique]["weight"]))
  if (specWeight/totalWeight)>=threshold:
   opDict[unique] = specWeight
 
 opDict["OTHER"] = sum(cdf[~cdf[col].isin(opDict)]["weight"])
 
 return opDict



def draw_cont_pdp(pts, targetSpan=0, name="graph", model=0, boringValue=1, leeway=0.05, ytitle="Relativity"):
 X = [pt[0] for pt in pts]
 Y = [pt[1] for pt in pts]
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, boringValue-targetSpan), max(max(Y)+leeway, boringValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=go.Scatter(x=X, y=Y), layout=layout)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP Graph for "+name, xaxis_title=name, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP Graph for "+name+", model "+model, xaxis_title=name, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image("graphs/"+name+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'.html')
 else:
  fig.write_image("graphs/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'__'+model+'.html')
 




def draw_cat_pdp(dyct, targetSpan=0, name="graph", model=0, boringValue=1, leeway=0.05, ytitle="Relativity"):
 
 X=[]
 Y=[]
 for thing in dyct["uniques"]:
  X.append(thing)
  Y.append(dyct["uniques"][thing])
 #if (dyct["OTHER"]!=boringValue):
 X.append("OTHER")
 Y.append(dyct["OTHER"])
 
 print(X,Y)
 
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, 0), max(max(Y)+leeway, boringValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=go.Bar(x=X, y=Y), layout=layout)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP Graph for "+name, xaxis_title=name, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP Graph for "+name+", model "+model, xaxis_title=name, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image("graphs/"+name+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'.html')
 else:
  fig.write_image("graphs/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'__'+model+'.html')

def draw_catcat_pdp(catcat, targetSpan=0, name="graph", model=0, boringValue=1, leeway=0.05, ytitle="Relativity", cat1="", cat2="", shapes=['/', '\\', 'x', '-', '|', '+', '.'], colors = ['red','orange','yellow','green','blue','indigo','violet']):
 
 bars=[]
 
 styleno=0
 
 for thing1 in catcat["uniques"]:
  X2=[]
  Y=[]
  for thing2 in catcat["uniques"][thing1]['uniques']:
   X2.append(thing2)
   Y.append(catcat["uniques"][thing1]['uniques'][thing2])
  X2.append("OTHER")
  Y.append(catcat["uniques"][thing1]["OTHER"])
  bars.append(go.Bar(name=thing1, x=X2, y=Y, marker_color=colors[styleno], marker_pattern={'shape':shapes[styleno]}))
  styleno+=1
 X2=[]
 Y=[]
 for thing2 in catcat["OTHER"]["uniques"]:
  X2.append(thing2)
  Y.append(catcat["OTHER"]["uniques"][thing2])
 X2.append("OTHER")
 Y.append(catcat["OTHER"]["OTHER"])
 bars.append(go.Bar(name="OTHER", x=X2, y=Y, marker_color='gray', marker_pattern={'shape':''}))
 
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, 0), max(max(Y)+leeway, boringValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=bars, layout=layout)
 
 fig.update_layout(legend_title_text = cat1)
 fig.update_xaxes(title_text = cat2)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP Graph for "+name, xaxis_title=name, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP Graph for "+name+", model "+model, xaxis_title=name, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image("graphs/"+name+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'.html')
 else:
  fig.write_image("graphs/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'__'+model+'.html')
 

def draw_catcont_pdp(catcont, targetSpan=0, name="graph", model=0, boringValue=1, leeway=0.05, ytitle="Relativity", cat="", cont="", colors = ['red','orange','yellow','green','blue','indigo','violet']):
 
 lines=[]
 
 styleno=0
 
 Y = []
 for thing in catcont['uniques']:
  Y=Y+[pt[1] for pt in catcont['uniques'][thing]]
 Y=Y+[pt[1] for pt in catcont['OTHER']]
 
 for thing in catcont['uniques']:
  lines.append(go.Scatter(name=thing, x=[pt[0] for pt in catcont['uniques'][thing]], y=[pt[1] for pt in catcont['uniques'][thing]], marker_symbol=styleno+1, marker_color=colors[styleno], marker_size=10))
  styleno+=1
 lines.append(go.Scatter(name="OTHER", x=[pt[0] for pt in catcont['OTHER']], y=[pt[1] for pt in catcont['OTHER']], marker_symbol=0, marker_color="gray", marker_size=10))
 
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, 0), max(max(Y)+leeway, boringValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=lines, layout=layout)
 
 fig.update_layout(legend_title_text = cat)
 fig.update_xaxes(title_text = cont)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP Graph for "+name, xaxis_title=name, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP Graph for "+name+", model "+model, xaxis_title=name, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image("graphs/"+name+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'.html')
 else:
  fig.write_image("graphs/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'__'+model+'.html')
 

 
def draw_contcont_pdp(contcont, targetSpan=0, name="graph", model=0, boringValue=1, leeway=0.05, ytitle="Relativity", cont1="", cont2="", lws=[5,7,9,11,13,15,17], colors = ['#00F','#44F','#77F','#99F','#AAF','#CCF']):
 lines = []
 
 styleno=0
 
 Y=[]
 for pt in contcont:
  Y=Y+[ptpt[1] for ptpt in pt[1]]
 
 for pt in contcont:
  lines.append(go.Scatter(name=pt[0], x=[ptpt[0] for ptpt in pt[1]], y=[ptpt[1] for ptpt in pt[1]], line_width = 1+lws[styleno]/4, marker_size = lws[styleno], marker_color=colors[styleno]))#'blue'))
  styleno+=1
 
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, 0), max(max(Y)+leeway, boringValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=lines, layout=layout)
 
 fig.update_layout(legend_title_text = cont1)
 fig.update_xaxes(title_text = cont2)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP Graph for "+name, xaxis_title=name, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP Graph for "+name+", model "+model, xaxis_title=name, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image("graphs/"+name+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'.html')
 else:
  fig.write_image("graphs/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'__'+model+'.html')
 
def draw_contcont_pdp_3D(contcont, targetSpan=0, name="graph", model=0, boringValue=1, leeway=0.05, ytitle="Relativity", cont1="", cont2=""):
 X1=[]
 X2=[]
 Y=[]
 
 for pt in contcont:
  X1=X1+[pt[0] for ptpt in pt[1]]
  X2=X2+[ptpt[0] for ptpt in pt[1]]
  Y=Y+[ptpt[1] for ptpt in pt[1]]
 
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, 0), max(max(Y)+leeway, boringValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=[go.Mesh3d(x=X1, y=X2, z=Y, opacity=0.5)])
 
 fig.update_layout(scene = dict(
                    xaxis_title=cont1,
                    yaxis_title=cont2,
                    zaxis_title=ytitle))
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP Graph for "+name)
  else:
   fig.update_layout(title="PDP Graph for "+name+", model "+model)
 
 if (model==0):
  plotly.offline.plot(fig, filename='graphs/'+name+'.html')
 else:
  plotly.offline.plot(fig, filename='graphs/'+name+'__'+model+'.html')
 

if __name__=="__main__":
 #exampleCatCat = {'uniques':{'a':{'uniques':{'c':1.1,'d':1.2},'OTHER':1.3},'b':{'uniques':{'c':1.4,'d':1.5},'OTHER':1.6}},'OTHER':{'uniques':{'c':1.7,'d':1.8},'OTHER':1.9}}
 #draw_catcat_pdp(exampleCatCat)
 
 #exampleCatCont = {"uniques":{"a":[[1,1.1],[2,1.2],[3,1.3]],"b":[[1,1.4],[2,1.5],[3,1.6]]},"OTHER":[[1,1.7],[2,1.8],[3,1.9]]}
 #draw_catcont_pdp(exampleCatCont)
 
 exampleContCont = [[1,[[1,1.1],[2,1.2],[3,1.3]]],[2,[[1,1.4],[2,1.5],[3,1.6]]],[3,[[1,1.7],[2,1.8],[3,1.9]]]]
 draw_contcont_pdp(exampleContCont)
 #draw_contcont_pdp_3D(exampleContCont)
 
 #exampleCont = [[1,1.4],[2,1.6],[3,0.4]]
 #draw_cont_pdp(exampleCont)
 
 #exampleCat = {"uniques":{"wstfgl":1.05, "florpalorp":0.92, "turlingdrome":0.99}, "OTHER":1.04}
 #draw_cat_pdp(exampleCat)
