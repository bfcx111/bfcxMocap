import os
import json

def GetImgFile(path, imagefilename='images', sortmode='cam'):
    camlists = sorted(os.listdir(path+'/{}'.format(imagefilename)))
    filepath_L=[]
    for camid in camlists:
        file_lists = sorted(os.listdir(path+'/{}/{}'.format(imagefilename,camid)))
        for filename in file_lists:
            filepath_L.append({
                'path' : path+'/{}/{}/{}'.format(imagefilename,camid,filename),
                'fileid' : int(filename.split('.')[0]),
                'camid' : int(camid) 
                })
    # breakpoint()
    if sortmode == 'cam':
        filepath_L = sorted(filepath_L, key = lambda i: (i['camid'], i['fileid']))
    else:
        filepath_L = sorted(filepath_L, key = lambda i: (i['fileid'], i['camid']))
    results=[]
    for itr in filepath_L:
        results.append(itr['path'])
    return results

def read_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data