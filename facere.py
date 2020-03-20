# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'facere.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import os
import numpy as np
from PIL import Image
import pickle
import csv
students=[]
stu=[]
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(190, 110, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(190, 160, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(190, 220, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(190, 430, 75, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(190, 270, 75, 23))
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(190, 360, 75, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "capture face"))
        self.pushButton_2.setText(_translate("MainWindow", "train faces"))
        self.pushButton_3.setText(_translate("MainWindow", "recognize face"))
        self.pushButton_4.setText(_translate("MainWindow", "ok"))
        self.pushButton_5.setText(_translate("MainWindow", "exit"))
        self.pushButton.clicked.connect(self.face_capture)
        self.pushButton_2.clicked.connect(self.train_face)
        self.pushButton_3.clicked.connect(self.recognize_face)
        self.pushButton_4.clicked.connect(self.face_capture)
       # self.pushButton_5.clicked.connect(self.exit)
        #self.pushButton_5.clicked.connect(lambda: os.system(cmd))

    def face_capture(self):
        id=self.lineEdit.text()
        
        #students.append(id)
        #print(students)
        face_classifier = cv2.CascadeClassifier('C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_eye.xml')
        os.mkdir('C:/Users/HP/.spyder-py3/images/'+id)
        def face_extractor(img):

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)
    
    
            if faces is():
                return None

            for(x,y,w,h) in faces:
                cropped_face = img[y:y+h, x:x+w]
                return cropped_face




        cap = cv2.VideoCapture(0)
        count = 0
        

        while True:
            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count+=1
                face = cv2.resize(face_extractor(frame),(200,200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = 'C:/Users/HP/.spyder-py3/images/%s/'%(id)+id+str(count)+'.jpg'
                print( file_name_path)
                cv2.imwrite(file_name_path,face)
       
                cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('Face Cropper',face)
            else:
                print("Face not Found")
                pass

            if cv2.waitKey(1)==13 or count==100:
                break

        cap.release()
        cv2.destroyAllWindows()
        print('Colleting Samples Complete!!!')
    
    def train_face(self):
        import cv2
        import os
        import numpy as np
        from PIL import Image
        import pickle

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "images")

        face_cascade = cv2.CascadeClassifier('C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        recognize = cv2.face.LBPHFaceRecognizer_create()

        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []
        st=[]
        stu=[]
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(root).replace(" ", "-").lower()
			#print(label, path)
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                        id_ = label_ids[label]
                    for i in label_ids:
                        st.append(i)
                        stu=set(st)
                        #print(stu)
                
            
			#print(id_)
			#y_labels.append(label) # some number
			#x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
                    pil_image = Image.open(path).convert("L") # grayscale
                    size = (112, 92)
                    final_image = pil_image.resize(size, Image.ANTIALIAS)
                    image_array = np.array(final_image, "uint8")
			#print(image_array)
                    faces = face_cascade.detectMultiScale(image_array, minNeighbors=5)

                    for (x,y,w,h) in faces:
                        roi = image_array[y:y+h, x:x+w]
                        x_train.append(roi)
                        y_labels.append(id_)


#print(y_labels)
#print(x_train)

        with open("face-labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)

        recognize.train(x_train, np.array(y_labels))
        recognize.save("face-trainner.yml")
        print("model trainned successfully")
    
    def recognize_face(self):
        import calendar
        from datetime import date
        import datetime
        #print(date.today())
        dateandtime=[]
    
        date1=("date :"+str(date.today()))
        #print(date1)
        dateandtime.append(date1)

        born = datetime.datetime.strptime(str(date.today()), '%Y-%m-%d').weekday() 
        day=("day :"+calendar.day_name[born] )
        dateandtime.append(day)
        #print(dateandtime)
        #print(date1)
        datenday=[tuple(dateandtime)]
        print(datenday)
        face_cascade = cv2.CascadeClassifier('C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("face-trainner.yml")

        labels = {"person_name": 1}
        with open("face-labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v:k for k,v in og_labels.items()}

        cap = cv2.VideoCapture(0)
        csvrows=[tuple(["register number","name"])]
        csvfinal=[]
        final_name=()
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
            	#print(x,y,w,h)
                roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
                roi_color = frame[y:y+h, x:x+w]
        
            	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
                id_, conf = recognizer.predict(roi_gray)
                if conf>4 and conf<85:
            		#print(5: #id_)
            		#print(labels[id_])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                    final_name=name.split(",")
                    fin=tuple(final_name)
                    #print(fin)
                    csvrows.append(fin)
                    #print(csvrows)
                    csvfinal=list(set(csvrows))
                    print(csvfinal)
                    
                    
                img_item = "me.png"
                cv2.imwrite(img_item, roi_color)
        
                color = (255, 0, 0) #BGR 0-255 
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            	#subitems = smile_cascade.detectMultiScale(roi_gray)
            	#for (ex,ey,ew,eh) in subitems:
            	#	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            # Display the resulting frame
                      
            cv2.imshow('frame',frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        print(csvrows)
        print(set(stu)-set(csvfinal))
        with open('face1.csv', 'w',newline='') as file:
            writer = csv.writer(file)
            writer.writerows(datenday)
            for row in csvfinal :
                writer.writerow(row)
            file.close()
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

        def exit(self)	:
            while(1):
                if 0xFF == ord('c'):
                    break				






if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
