import os
import model as modellib
import torch
import numpy as np
import coco
import sys
import cv2
import torch.nn as nn

# import for HTTP server
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial
from io import BytesIO
import json
import base64


def get3DMatching(r, image, focusRange):
    class_names = ['BG', 'chair', 'couch', 'bed', 'dining table', 'tv', 'laptop', 'keyboard', 'book', 'door', 'window']
    cos_simi = nn.CosineSimilarity(0)

    results = []

    N = r['rois'].shape[0]
    for xi in range(N):
        y1, x1, y2, x2 = r['rois'][xi]

        if focusRange is not None:
            centerX = (x1 + x2) / 2
            if centerX > focusRange[1] or centerX < focusRange[0]:
                continue

        image_bd = image[y1:y2, x1:x2]
        image_bd_mask = r['masks'][:, :, xi][y1:y2, x1:x2]
        C = np.where(image_bd_mask == 1)
        tmp = image_bd[C[0], C[1], :]
        average_color = np.uint8(np.mean(tmp, axis=0))

        max_rate = 300 / max(image_bd.shape[0], image_bd.shape[1])
        # import pdb; pdb.set_trace()
        re_image = cv2.resize(image_bd,
                              (int(image_bd.shape[1] * max_rate),
                               int(image_bd.shape[0] * max_rate)),
                              interpolation=cv2.INTER_CUBIC)

        back = cv2.imread('background.png')
        back[10:re_image.shape[0] + 10, 10:re_image.shape[1] + 10] = re_image
        image_new = back

        # Run detection
        my_results, my_roi, index = model.detect([image_new])
        a1 = my_roi[index[0]].view(-1).data

        filename = []
        similarity = []

        label = class_names[r['class_ids'][xi]]
        if label == 'dining table':
            file_names = next(os.walk('table_npy'))[2]

            for i, (file_name) in enumerate(file_names):

                if(file_name[:8] != 'table001'and file_name[:8] != 'table003'):

                    # r1 = torch.from_numpy(np.load(os.path.join('table_npy', file_name))).cuda()
                    r1 = torch.from_numpy(
                        np.load(
                            os.path.join(
                                'table_npy',
                                file_name))).cuda()

                    # cos_simi(a1.unsqueeze(0),r1.unsqueeze(0))
                    similarity_2 = cos_simi(a1, r1)

                    filename.append(file_name)
                    similarity.append(similarity_2)

            results.append((xi, filename[similarity.index(max(similarity))], average_color.tolist()))

        elif label == 'chair':

            file_names = next(os.walk('chair_npy'))[2]

            for i, (file_name) in enumerate(file_names):

                r1 = torch.from_numpy(
                    np.load(
                        os.path.join(
                            'chair_npy',
                            file_name))).cuda()

                similarity_2 = cos_simi(a1, r1)

                filename.append(file_name)
                similarity.append(similarity_2)

            results.append((xi, filename[similarity.index(max(similarity))], average_color.tolist()))

        elif label == 'couch':

            file_names = next(os.walk('couch_npy'))[2]

            for i, (file_name) in enumerate(file_names):

                r1 = torch.from_numpy(np.load(os.path.join('couch_npy', file_name))).cuda()

                similarity_2 = cos_simi(a1, r1)

                filename.append(file_name)
                similarity.append(similarity_2)

            results.append((xi, filename[similarity.index(max(similarity))], average_color.tolist()))

        elif label == 'tv':

            file_names = next(os.walk('tv_npy'))[2]

            for i, (file_name) in enumerate(file_names):

                r1 = torch.from_numpy(np.load(os.path.join('tv_npy', file_name))).cuda()

                similarity_2 = cos_simi(a1, r1)

                filename.append(file_name)
                similarity.append(similarity_2)

            results.append((xi, filename[similarity.index(max(similarity))], average_color.tolist()))

    return results


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
        inputFrame = np.frombuffer(
            base64.decodebytes(
                recvData["frame"].encode("ASCII")),
            dtype=np.uint8)
        inputFrame = inputFrame.reshape(recvData["shape"])

        # focus range - the range we will use to determine what objects be
        # places into Unity.
        if "focusRange" in recvData:
            focusRange = recvData["focusRange"]
        else:
            focusRange = None

        # inference
        # Run detection
        results, _, _ = model.detect([inputFrame])
        results = results[0]
        modelMatchingResults = get3DMatching(results, inputFrame, focusRange)

        # Build object detection result dict
        resultDict = {}
        resultDict["ObjDet"] = {
            "num": results["class_ids"].shape[0],
            "rois": base64.b64encode(results["rois"]).decode("ASCII"),
            "class_ids": base64.b64encode(results["class_ids"]).decode("ASCII"),
            "scores": base64.b64encode(np.ascontiguousarray(results["scores"], dtype=np.float32)).decode("ASCII"),
            "masks": base64.b64encode(results["masks"]).decode("ASCII"),
            "models": modelMatchingResults}

        # some ext information such as frame index will be store in "extData"
        # we usuaully do not process extData here, but send back to caller
        # directly
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

    server = HTTPServer(
        ('', PORT), partial(
            ObjectDetectionHTTPRequestHandler, model))
    server.serve_forever()
