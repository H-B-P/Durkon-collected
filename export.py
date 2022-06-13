import misc

exampleModel = {"BASE_VALUE":1700,"conts":{"cont1":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "cont2":[[37,1.2],[98, 0.9]]}, "cats":{"cat1":{"uniques":{"wstfgl":1.05, "florpalorp":0.92}, "OTHER":1.04}}}

def find_max_len(model, multip=1):
 ml=0
 for col in model['conts']:
  if (len(model['conts'][col])*multip)>ml:
   ml = (len(model['conts'][col])*multip)
 for col in model['cats']:
  if (len(model['cats'][col]['uniques'])+1)>ml:
   ml = len(model['cats'][col]['uniques'])+1
 return ml

def get_cont_inputs(model, col, detail=1):
 op=[]
 for i in range(len(model['conts'][col])-1):
  for j in range(detail):
   op.append(float(model['conts'][col][i][0]*(detail-j) + model['conts'][col][i+1][0]*j)/detail)
 op.append(model['conts'][col][-1][0])
 return op

def model_to_lines(model, detail=1, filename="op.csv"):
 lines = ['']*(find_max_len(model, detail)+4)
 
 #Add base value
 
 for l in range(len(lines)):
  if l==1:
   lines[l] = lines[l]+",BASE_VALUE"
  elif l==2:
   lines[l] = lines[l]+","+str(model['BASE_VALUE'])
  else:
   lines[l] = lines[l]+','
 
 #Add conts
 
 for col in model['conts']:
  lines[0] = lines[0]+',,,'
  lines[1] = lines[1]+',,'+col+','
  contPuts=get_cont_inputs(model, col, detail)
  for i in range(len(contPuts)):
    lines[i+2] = lines[i+2] + ',,' + str(contPuts[i]) + ',' + str(misc.get_effect_of_this_cont_col_on_single_input(contPuts[i], model, col))
  for l in range(len(contPuts)+2, len(lines)):
   lines[l] = lines[l]+',,,'
 
 #Add cats
 
 for col in model['cats']:
  lines[0] = lines[0]+',,,'
  lines[1] = lines[1]+',,'+col+','
  keys = misc.get_sorted_keys(model, col)
  for i in range(len(keys)):
   lines[i+2] = lines[i+2]+',,'+str(keys[i])+','+str(model['cats'][col]['uniques'][keys[i]])
  lines[len(keys)+2] = lines[len(keys)+2] +',,OTHER,'+str(model['cats'][col]['OTHER'])
  for l in range(len(keys)+3, len(lines)):
   lines[l] = lines[l]+',,,'
 
 #Write to file
 
 f = open(filename, "w")
 for line in lines:
  print(line)
  f.write(line+'\n')
 f.close()


if __name__ == '__main__':
 exampleModel = {"BASE_VALUE":1700,"conts":{"cont1":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "cont2":[[37,1.2],[98, 0.9]], "cont3":[[12,1.2],[13, 0.9],[14, 1.1]]}, "cats":{"cat1":{"uniques":{"wstfgl":1.05, "florpalorp":0.92}, "OTHER":1.01},"cat2":{"uniques":{"ruska":1.1, "roma":0.93, 'rita':0.3}, "OTHER":1.9}}}
 model_to_lines(exampleModel, 3, 'example.csv')
