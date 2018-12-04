import os
import model as modellib
import torch
import numpy as np
import coco
import sys

# import for HTTP server
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial
from io import BytesIO
import json
import base64


class ObjectDetectionHTTPRequestHandler(BaseHTTPRequestHandler):

    def __init__(self, model, *args, **kwargs):
        self._model = model
        super().__init__(*args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()
        response = BytesIO()
        recvData = json.loads(body.decode("ASCII"))

        # convert the input to frame
        inputFrame = np.frombuffer(base64.decodebytes(recvData["frame"].encode("ASCII")), dtype=np.uint8)
        inputFrame = inputFrame.reshape(recvData["shape"])

        # inference
        # Run detection
        results = model.detect([inputFrame])[0]

        # Build object detection result dict
        resultDict = {}
        resultDict["ObjDet"] = {
            "num": results["class_ids"].shape[0],
            "rois": base64.b64encode(results["rois"]).decode("ASCII"),
            "class_ids": base64.b64encode(results["class_ids"]).decode("ASCII"),
            "scores": base64.b64encode(np.ascontiguousarray(results["scores"], dtype=np.float32)).decode("ASCII"),
            "masks": base64.b64encode(results["masks"]).decode("ASCII")
        }

        # some ext information such as frame index will be store in "extData"
        # we usuaully do not process extData here, but send back to caller directly
        if "extData" in recvData:
            resultDict["extData"] = recvData["extData"]

        resultData = json.dumps(resultDict)
        response.write(resultData.encode("ASCII"))
        self.wfile.write(response.getvalue())


if __name__ == "__main__":
    PORT = 9000

    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(
        ROOT_DIR, sys.argv[1])

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

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        # GPU_COUNT = 0 for CPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()
    # Create model object.
    model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()

    # Load weights trained on MS-COCO
    model.load_state_dict(torch.load(COCO_MODEL_PATH))

    server = HTTPServer(('', PORT), partial(ObjectDetectionHTTPRequestHandler, model))
    server.serve_forever()
