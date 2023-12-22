import os
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUi
from PyQt5 import uic
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap, QImage


import cv2
import numpy as np           

from cgitb import enable
import sys
import os

import multiprocessing

############ Model Lib ###############
import os
import platform
import sys
from pathlib import Path
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
########


qtCreatorFile = "design.ui"  # Enter file here.
global ImageFile
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

# for video capturing
e = multiprocessing.Event()
p = None

################# Load YOLO Model
weights = ROOT / 'final_model.pt'  # model path or triton URL
source = ROOT / 'data/images'  # file/dir/URL/glob/screen/0(webcam)
data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
imgsz = (640, 640)  # inference size (height, width)
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
view_img=False  # show results
save_txt=False  # save results to *.txt
save_conf=False  # save confidences in --save-txt labels
save_crop=False  # save cropped prediction boxes
nosave=False  # do not save images/videos
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
visualize=False  # visualize features
update=False  # update all models
project=ROOT / 'runs/detect'  # save results to project/name
name='exp'  # save results to project/name
exist_ok=False  # existing project/name ok, do not increment
line_thickness=2  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
half=False  # use FP16 half-precision inference
dnn=False  # use OpenCV DNN for ONNX inference
vid_stride=1  # video frame-rate stride
save_img= False

# Load model
device = ''
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent=parent)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        #self.comboBox.addItem("Image")
        self.comboBox.addItem("Camera")
        self.comboBox.addItem("Video")
        self.Btn_Browse.clicked.connect(self.Browser)
        #self.Btn_Classifier.clicked.connect(self.classifier)
        self.Btn_Start.clicked.connect(self.startDetect)
        

    # Convert an opencv image to QPixmap
    def drawPrevImage(self, cvImg):
        # Convert image to QImage
        scene = QtWidgets.QGraphicsScene()
        pixmap = QImage(cvImg.data, cvImg.shape[1], cvImg.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap(pixmap)
        scene.addPixmap(pix)
        self.graphicsView_prev.setScene(scene)   # set the current viewed image


    # Convert an opencv image to QPixmap
    def drawResImage(self, cvImg):
        # Convert image to QImage
        scene = QtWidgets.QGraphicsScene()
        pixmap = QImage(cvImg.data, cvImg.shape[1], cvImg.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap(pixmap)
        scene.addPixmap(pix)
        self.graphicsView_res.setScene(scene)   # set the current viewed image
    
    def Browser(self):
        method = self.comboBox.currentText()
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        ImageFile = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image To Process", "","All Files (*);;Image Files(*.jpg *.gif)", options=options)
        if ImageFile:
            self.label_filename.setText(ImageFile[0])
            if(method != "Video"):
                org_img = cv2.imread(ImageFile[0]) 
                img_colored = cv2.resize(org_img ,(400,400))    
                self.drawPrevImage(img_colored)
    
    def Detect(self, imagePath):
        source = imagePath  # file/dir/URL/glob/screen/0(webcam)
        source = str(source)
        imgsz=(640, 640)  # inference size (height, width)
        visualize=False  # visualize features
        view_img=True  # show results

        # Dataloader
        bs = 1  # batch_size
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = False #increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                    
                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    #cv2.imshow(str(p), im0)
                    return im0
                    #cv2.waitKey(0)  # 1 millisecond

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    

    def videoDetect(self, videotype, filename):
        if videotype == "Camera":
            vidcap = cv2.VideoCapture(0)
        if videotype == "Video":
            vidcap = cv2.VideoCapture(filename)
        size = (400,400)
        while True: 
            success, image = vidcap.read()
            if not success:
                print("no camera detected")
                break

            if success:
                image = cv2.flip(image, 1)
                #h, w =  image.shape[1], image.shape[0]
                #image = cv2.resize(image,size)
                cv2.imwrite(ROOT / "frame.jpg", image) 
                framePath = ROOT / 'frame.jpg'
                orgframe = cv2.imread(framePath)
                self.drawPrevImage(orgframe)
                #image = cv2.resize(image,(h,w))
                image = self.Detect(framePath)
                self.drawResImage(image)
                cv2.waitKey(1)

    def startDetect(self):
        selected = self.comboBox.currentText()
        imgPath = self.label_filename.text()
        if selected == "Image":
            res = self.Detect(imgPath)
            self.drawResImage(res)
        if selected == "Camera":
            self.videoDetect(selected, "")
        if selected == "Video":
            self.videoDetect(selected, imgPath)
 
    def Close(self):
        self.destroy()
    
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())