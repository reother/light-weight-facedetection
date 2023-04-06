import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from time import monotonic
import cv2
# Load the serialized TensorRT engine from file

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    description: compute the IoU of two bounding boxes
    param:
        box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
        box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
        x1y1x2y2: select the coordinate format
    return:
        iou: computed iou
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

with open('RFB2TRT.engine', 'rb') as f:
    engine_data = f.read()
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)
input_shape = (1,3,240,320)
input_size = np.product(input_shape) * np.dtype(np.float32).itemsize
output_shape = (1,4420,6)
output_size = np.product(output_shape) * np.dtype(np.float32).itemsize


# Allocate numpy arrays for input and output data
input_data = np.zeros(input_shape, dtype=np.float32)
output_data = np.zeros(output_shape, dtype=np.float32)

image_mean = np.array([127, 127, 127])
image_std = 128.0


# Create a context for inference
context = engine.create_execution_context()

webcam = cv2.VideoCapture(-1)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    time0 = monotonic()
    status, frame = webcam.read()
    time1 = monotonic()
    # print('webcam read:',time1-time0)

    device_input = cuda.mem_alloc(int(input_size))
    device_output = cuda.mem_alloc(int(output_size))
    image = frame
    image_resize = cv2.resize(image,(320,240))
    input_image = ((image_resize - image_mean).astype(np.float32)/image_std)
    input_data[0] = input_image.transpose(2,0,1)
    cuda.memcpy_htod(device_input,input_data)
    context.execute(1, [int(device_input), int(device_output)])
    cuda.memcpy_dtoh(output_data, device_output)
    conf = output_data[0,:,5]
    mask = conf > 0.8
    conf = conf[mask]
    conf = np.sort(conf)
    
    bbox_masked = output_data[0,mask,:4]
    bbox_masked = bbox_masked[np.argsort(-conf)]
    if conf.shape[0] != 0:
        boxes = np.expand_dims(bbox_masked[0,:],0)  
    else:
        boxes = np.array([])
    for i in range(boxes.shape[0]):
        box = boxes[0, :]
        box = np.clip(box, 0, 1)
        print('x1 {} y1 {}'.format(box[0],box[1]))
        print('x2 {} y2 {}'.format(box[2],box[3]))
        x1, y1 = int(box[0]*1920),int(box[1]*1080)
        x2, y2 = int(box[2]*1920),int(box[3]*1080)
        image = cv2.rectangle(image,(x1, y1),(x2, y2),(0,0,255))
    time2 = monotonic()
    print('face detection infer: {}ms'.format(int((time2-time1)*1000)))
    print('FPS: {}'.format(int((1/(time2-time1)))))
    cv2.imshow("test", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


