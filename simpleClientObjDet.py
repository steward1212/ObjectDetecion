import requests
import json
import cv2
import sys
import numpy as np
import datetime
import base64
import matplotlib.pyplot as plt
import visualize


CLASS_NAMES = [
    'BG',
    'chair',
    'couch',
    'bed',
    'dining table',
    'tv',
    'laptop',
    'keyboard',
    'book',
    'door',
    'window']

URL = "http://localhost:9001/"
HEADERS = {'Content-type': 'application/json'}

frame = cv2.imread(sys.argv[1])
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
inputShape = frame.shape
testDict = {"idx": 0, "shape": frame.shape, "frame": base64.b64encode(frame).decode("ASCII")}
jsonData = json.dumps(testDict)
print(f"before send: {datetime.datetime.now()}")
response = requests.post(URL, headers=HEADERS, data=jsonData)
print(f"after send: {datetime.datetime.now()}")
result = json.loads(response.text)

objDetResult = result["ObjDet"]
num = objDetResult["num"]
rois = np.frombuffer(base64.decodebytes(objDetResult["rois"].encode("ASCII")), dtype=np.int32).reshape((num, 4))
class_id = np.frombuffer(base64.decodebytes(objDetResult["class_ids"].encode("ASCII")), dtype=np.int32).reshape((num,))
scores = np.frombuffer(base64.decodebytes(objDetResult["scores"].encode("ASCII")), dtype=np.float32).reshape((num,))
masks = np.frombuffer(base64.decodebytes(objDetResult["masks"].encode("ASCII")), dtype=np.uint8).reshape((inputShape[0], inputShape[1], num))
models = objDetResult["models"]
print(f"after load: {datetime.datetime.now()}")

print(f"num of class: {num}")
print(f"rois: {rois}")
print(f"class_id: {class_id}")
print(f"scores: {scores}")
print(f"model: {models}")

# visualize.display_instances(frame, rois, masks, class_id, CLASS_NAMES, scores)
# plt.show()
