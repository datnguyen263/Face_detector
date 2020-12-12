import sys
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import *
import numpy as np
from cv2 import cv2
import dlib
import os
import pickle

#UI Project
class Window(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("Facial recognition system")
        self.resize(600, 400)

        label = QLabel(self)
        pixmap = QPixmap('img.jpg')
        label.setPixmap(pixmap)
        self.layout.addWidget(label)

        button = QPushButton("Training")
        button.clicked.connect(self.training) 
        self.layout.addWidget(button)

        button = QPushButton("Camera")
        button.clicked.connect(self.cam)
        self.layout.addWidget(button)

    #Trích xuất đặc trưng khuôn mặt và train dựa trên data thu thập
    def training(self):
        #path đến folder chứa data
        self.b = QPlainTextEdit(self)
        path = './facedata'
        #Dùng các file pretrain
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        FACE_DESC = []
        FACE_NAME = []

        #Xử lí bộ nhãn
        for fn in os.listdir(path):
            if fn.endswith('.jpg'):
                img = cv2.imread(path + '\\' + fn)
                dets = detector(img,1)
                for k, d in enumerate(dets):
                    shape = sp(img, d)
                    face_desc = model.compute_face_descriptor(img, shape, 1)
                    FACE_DESC.append(face_desc) 
                    print('loading...   ' + fn)
                    FACE_NAME.append(fn[:fn.index('_')])
        #ghi liệu trích xuất vào trainset.pk
        pickle.dump((FACE_DESC, FACE_NAME), open('trainset.pk','wb'))

    #Xử dụng camera để đọc new data
    #Từ data ghi được từ trainset.pk để định danh
    def cam(self):
        face_derector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        FACE_DESC, FACE_NAME = pickle.load(open('trainset.pk', 'rb'))
        
        cap = cv2.VideoCapture(0)

        #frame được xử lí bằng thư viện dlib để xác định khuôn mặt từ camera gán vào faces
        #Sử dụng slide để cắt khung ảnh
        #Trích xuất trực tiếp khuôn mặt và so sánh đến trainset.pk
        #Sai số trong khoảng từ 0-0.5 sẽ được định danh và gán nhãn, nằm ngoài 0.5 sẽ không thể định danh
        while True:
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_derector.detectMultiScale(gray, 1.3, 5)
            rects = detector(frame,1)

            for (x, y , w, h) in faces:
                img = frame[y - 10: y + h + 10, x - 10: x + w + 10][:,:,::-1]
                dets = detector(img, 1)
                for k, d in enumerate(dets):
                    shape = sp(img, d)
                    face_desc0 = model.compute_face_descriptor(img, shape, 1)
                    d = []
                    for face_desc in FACE_DESC:
                        #tính sai số
                        d.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0))) 
                    d = np.array(d)
                    idx = np.argmin(d)
                    if d[idx] < 0.3:
                        name = FACE_NAME[idx]
                        #name + độ chính xác
                        name = name + "  " + str(round((face_desc[idx]/face_desc0[idx])*100)) + "%"
                        print(name)
                        cv2.putText(frame, name, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
                        cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, 'Khong the dinh danh', (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
                        cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.imshow('test', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
       
app = QApplication(sys.argv)
screen = Window()
screen.show()
sys.exit(app.exec_())