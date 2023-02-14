"""
This code is used to batch detect images in a folder.
"""
import argparse
import cv2
from vision.ssd.config.fd_config import define_img_size
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

parser = argparse.ArgumentParser(
    description='detect_imgs')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=320, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.9, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1500, type=int,
                    help='nms candidate size')
parser.add_argument('--test_device', default="cuda", type=str,
                    help='cuda:0 or cpu')
args = parser.parse_args()
define_img_size(args.input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'



test_device = args.test_device

model_path = "models/RFB-Epoch-30-Loss-1.3682626485824585.pth"
net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=test_device)
predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)

net.load(model_path)

sum = 0
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, args.candidate_size / 2, args.threshold)
    sum += boxes.size(0)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        image = cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255))


    cv2.imshow("test", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()




