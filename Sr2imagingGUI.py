# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:34:01 2022

@author: Jonathan
"""

#%%

import sys #needed if you want to interact with the program in command line
# import PySpin as ps
import numpy as np
# from BasicCameraV1 import CameraSystem #class that sets up the camera
from time import perf_counter # used to compute update frequency

import pyqtgraph as pg #plotting module for pyqt

from PyQt5 import QtCore, QtGui, QtWidgets #GUI module
from PyQt5.QtCore import Qt

from PyQt5.QtGui import QIntValidator

from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QVBoxLayout, 
QHBoxLayout, QPushButton, QCheckBox, QSlider, QLineEdit)

import siglent_psu_api as siglent

from scipy.signal import convolve2d

#Crappy workaround. Used to suppress problem of numpy importing twice. Once manually, and once through PySpin
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#make sure our camera class can be imported, as long as the module is in same folder as this script
try:
    sys.path.index(os.getcwd())
except ValueError:
    sys.path.append(os.getcwd())

from Sr2CameraSetup import CameraSystem #class that sets up the camera

# import matplotlib.image
from skimage.io import imsave
from datetime import datetime


#%%

"""
To implement:
digital gain slider with number label
checkbox for averaging bg - done

slider for graphs vertical scale
checkbox for vertical graph
slider for kernel size
"""

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs) #Boilerplate stolen from the interwebs
        self.demo = False
        self.powerControl = False
        
        if not self.demo:
            self.camera = CameraSystem() #Starting camera using class from the setup script
            #Setting up the camera to suit our needs
            self.camera.acquisitionMode('cont')
            self.camera.exposureAuto('off')
            self.camera.gainAuto('off')
            self.expTime = 20000
            self.camera.exposure(self.expTime)
            self.camera.gain(43)
            self.camera.binningMode('average')
            self.horBin = 2
            self.vertBin = 2
            self.horSize = 2048//self.horBin
            self.vertSize = 1536//self.vertBin
            self.camera.binning(bins=[self.horBin,self.vertBin])
            self.camera.beginAcquisition()
        elif self.demo:
            self.expTime = 50
            self.horBin = 2
            self.vertBin = 2
            self.horSize = 2048//self.horBin
            self.vertSize = 1536//self.vertBin
            
            self.base_image = np.ones((self.horSize,self.vertSize))
            # kernel = np.ones((11,1))
            self.base_image[200:-200,150:170] = 130
            self.base_image[200:-200,-170:-150] = 130
            
            # test = main.base_image
            n = 251
            kernel = np.ones((n,1))/n
            # kernel[0,:]=1
            self.base_image = convolve2d(self.base_image,kernel,'same')
        
        # Live video UI setup
        self.videoFeed = pg.GraphicsLayoutWidget()
        self.video = self.videoFeed.addViewBox()
        self.video.setAspectLocked(True)
        
        self.img = pg.ImageItem(levels =(0,255))
        self.video.addItem(self.img)
        
        self.removeBackgroundBool = False
        self.divideBackgroundBool = False
        self.smoothBackgroundBool = False
        self.analyseBeamBool = False
        self.saveImageBool = False
        self.averageImagesBool = False
        
        self.backgroundImage = np.ones((self.horSize,self.vertSize))
        self.smoothValue = 15

        
        self.makeROI()
        
        
        
        layout = QGridLayout()
    
        
        self.timeStart = perf_counter()
        self.FPSBuffer = np.zeros(10)
        self.FPS = 0
        
        self.infoBox = QtWidgets.QHBoxLayout()
        self.FPSlabel = QtWidgets.QLabel()
        self.FlourLabel1 = QtWidgets.QLabel()
        self.FlourLabel2 = QtWidgets.QLabel()
        self.infoBox.addWidget(self.FPSlabel)
        self.infoBox.addWidget(self.FlourLabel1)
        self.infoBox.addWidget(self.FlourLabel2)
        
        self.histPlots = self.makeHistWidget()
        
        self.buttonTakeBackground = QPushButton('Take Background')
        self.buttonTakeBackground.clicked.connect(self.takeBackground)
        
        self.checkboxRemoveBackground = QCheckBox('Remove Background',self)
        self.checkboxRemoveBackground.stateChanged.connect(self.checkBGstate)
        
        self.checkboxDivideBackground = QCheckBox('Divide Background (unchecked subtracts)',self)
        self.checkboxDivideBackground.stateChanged.connect(self.checkDivideState)
        
        # self.checkboxSmoothBackground = QCheckBox('Smooth Background',self)
        # self.checkboxSmoothBackground.stateChanged.connect(self.checkSmoothBGState)
        
        self.infoBox.addWidget(self.buttonTakeBackground)
        self.infoBox.addWidget(self.checkboxRemoveBackground)
        self.infoBox.addWidget(self.checkboxDivideBackground)
        # self.infoBox.addWidget(self.checkboxSmoothBackground)
        
        
        self.beamBox = QHBoxLayout()
        self.checkboxAnalyseBeam = QCheckBox('Analyse beam',self)
        self.checkboxAnalyseBeam.stateChanged.connect(self.checkAnalyseState)
        
        self.sliderBeam = QSlider(Qt.Horizontal)
        self.sliderBeam.setMinimum(10)
        self.sliderBeam.setMaximum(100)
        self.sliderBeam.setValue(90)
        self.sliderBeam.setTickPosition(QSlider.TicksBelow)
        self.sliderBeam.setTickInterval(5)
        self.sliderBeam.valueChanged.connect(self.sliderBeamChanged)
        self.labelBeamSlider = QtWidgets.QLabel(str(self.sliderBeam.value()) + '% of max (taken in red box)')
        self.beamCutoffValue = self.sliderBeam.value()/100
        
        self.beamBox.addWidget(self.checkboxAnalyseBeam)
        self.beamBox.addWidget(self.sliderBeam)
        self.beamBox.addWidget(self.labelBeamSlider)
        
        
        
        layout.addWidget(self.videoFeed,0,0,1,1)
        layout.addLayout(self.infoBox,1,0,1,2)
        layout.addLayout(self.histPlots,0,1,1,1)
        layout.addLayout(self.beamBox,2,0,1,1)

        #Now power control layout
        if self.powerControl:
            self.coilPSU1 = siglent.SIGLENT_PSU("10.90.61.242")
            self.coil1Box = QHBoxLayout()
            self.checkboxCoil1Power = QCheckBox('power coil 1',self)
            self.checkboxCoil1Power.stateChanged.connect(self.checkCoil1State)
            self.coil1PowerBool = False
            
            self.sliderCoil1 = QSlider(Qt.Horizontal)
            self.sliderCoil1.setMinimum(0)
            self.sliderCoil1.setMaximum(32)
            self.sliderCoil1.setValue(0)
            # self.sliderCoil1.setTickPosition(QSlider.TicksBelow)
            self.sliderCoil1.setTickInterval(1)
            self.sliderCoil1.valueChanged.connect(self.sliderCoil1Changed)
            self.labelCoil1Slider = QtWidgets.QLabel(str(self.sliderCoil1.value()/100) + 'A')
            
            self.coil1Box.addWidget(self.checkboxCoil1Power)
            self.coil1Box.addWidget(self.sliderCoil1)
            self.coil1Box.addWidget(self.labelCoil1Slider)
            
            layout.addLayout(self.coil1Box,3,0,1,1)
        
        #Save image setup
        self.buttonSaveImage = QPushButton('Save current image')
        self.buttonSaveImage.clicked.connect(self.saveImage)
        self.imageFileNameText = 'Image_'
        self.lineEditImageFileName = QLineEdit(self.imageFileNameText)
        self.lineEditImageFileName.textChanged.connect(self.imageFileNameChanged)
        
        self.saveImageBox = QtWidgets.QHBoxLayout()
        self.saveImageBox.addWidget(self.buttonSaveImage)
        self.saveImageBox.addWidget(self.lineEditImageFileName)
        
        layout.addLayout(self.saveImageBox,4,0,1,2)
        
        #Averaging setup
        self.checkboxAvgImage = QCheckBox('Average images',self)
        self.checkboxAvgImage.stateChanged.connect(self.checkAvgState)
        self.avgNumber = 5
        #probably have to be a string?
        self.lineEditAvgNumber = QLineEdit(str(self.avgNumber))
        self.validatorAvgNumber = QIntValidator()
        self.validatorAvgNumber.setRange(2,20)
        self.lineEditAvgNumber.setValidator(self.validatorAvgNumber)
        self.lineEditAvgNumber.textChanged.connect(self.avgNumberChanged)
        self.buttonSaveArray = QPushButton('Save array')
        self.buttonSaveArray.clicked.connect(self.saveArray)
        self.imageArray = np.zeros((self.horSize,self.vertSize,20)).astype('int')
        self.arrayFileNameText = 'Array_'
        self.lineEditArrayFileName = QLineEdit(self.arrayFileNameText)
        self.lineEditArrayFileName.textChanged.connect(self.arrayFileNameChanged)
        
        
        self.averageImageBox = QtWidgets.QHBoxLayout()
        self.averageImageBox.addWidget(self.checkboxAvgImage)
        self.averageImageBox.addWidget(self.lineEditAvgNumber)
        self.averageImageBox.addWidget(self.buttonSaveArray)
        self.averageImageBox.addWidget(self.lineEditArrayFileName)
        
        layout.addLayout(self.averageImageBox,5,0,1,2)
        
        window = QtWidgets.QWidget()
        window.setLayout(layout) 
    
        self.setCentralWidget(window)
        
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update) 
        self.timer.start()

        
    def update(self):
        #Take new image
        if self.demo:
            self.lastImage = self.base_image + np.random.randint(-10,10,size=(self.horSize,self.vertSize))
        else:
            self.lastImage = self.camera.getImage().reshape((self.horSize,self.vertSize),order='F')
            # And then casting to float to avoid overflow errors later
            self.lastImage = self.lastImage.astype('int')
        
        #update image array used in averaging
        self.imageArray = np.roll(self.imageArray,shift=1,axis=2)
        self.imageArray[:,:,0] = np.copy(self.lastImage)
        
        #analyze beam live for focusing beam
        if self.analyseBeamBool:
            self.beamMax = np.max(self.topDataBox)
            
            boolImage = self.lastImage > (self.beamMax*self.beamCutoffValue)
            dim = list(boolImage.shape)
            dim.append(3)
            im = np.zeros(dim)
            im[:,:,2] = boolImage*250
            # self.shit = np.zeros((list(boolImage.shape).append(3)))
            # self.img.setImage(self.boolImage*250, levels=(0,255) )
            self.img.setImage(im, levels=(0,255) )
        #Averaging
        elif self.averageImagesBool:
            self.avgImage = np.mean(self.imageArray[:,:,:self.avgNumber],axis=2)
            if (self.removeBackgroundBool and not self.divideBackgroundBool):
                self.displayImage = self.avgImage - self.backgroundImage
            elif (self.removeBackgroundBool and self.divideBackgroundBool):
                # self.lastImage = self.lastImage/(self.backgroundImage+0.001)*10
                self.displayImage = np.ones((self.horSize,self.vertSize))*255 - ((np.ones((self.horSize,self.vertSize))*255 - self.avgImage) / (np.ones((self.horSize,self.vertSize))*255 - self.backgroundImage + 0.001))*255
            else:
                self.displayImage = self.avgImage
        #just single images
        else:
            if (self.removeBackgroundBool and not self.divideBackgroundBool):
                self.displayImage = self.lastImage - self.backgroundImage
            elif (self.removeBackgroundBool and self.divideBackgroundBool):
                # self.lastImage = self.lastImage/(self.backgroundImage+0.001)*10
                self.displayImage = np.ones((self.horSize,self.vertSize))*255 - ((np.ones((self.horSize,self.vertSize))*255 - self.lastImage) / (np.ones((self.horSize,self.vertSize))*255 - self.backgroundImage + 0.001))*255
            else:
                self.displayImage = self.lastImage
        
        # now that we finally know which image to display, it is set to display
        self.img.setImage(self.displayImage, levels =(0,255))
        
        self.updateFPS()
        self.updateROI()
        self.updateHistData()
        if self.saveImageBool:
            self.saveImageBool = False
            imsave(self.fileNameImage+'.png', np.rot90(np.abs(self.displayImage).astype(np.uint8)))

    
    def updateFPS(self):
        self.timeStop = perf_counter()
        timeElapsed = self.timeStop - self.timeStart
        self.timeStart = np.copy(self.timeStop)
        self.FPSBuffer[0] = 1/timeElapsed
        self.FPSBuffer = np.roll(self.FPSBuffer,1)
        self.FPS = np.mean(self.FPSBuffer)
        self.FPSlabel.setText(f'FPS: {self.FPS:.2f}')
        
    def makeHistWidget(self):
        self.hist1Widget = pg.PlotWidget(title = "Top beam")
        self.hist1Widget.setRange(yRange=[0,100])
        self.hist1Plot = self.hist1Widget.plot(pen=pg.mkPen(width = 2, color = (255,255,255)))

        self.hist2Widget = pg.PlotWidget(title = "Bottom beam")
        self.hist2Widget.setRange(yRange=[0,100])
        self.hist2Plot = self.hist2Widget.plot(pen=pg.mkPen(width = 2, color = (255,255,255)))

        histWidget = QtWidgets.QVBoxLayout()
        histWidget.addWidget(self.hist1Widget)
        histWidget.addWidget(self.hist2Widget)
        
        return histWidget
        
    def updateHistData(self):
        topData = np.sum(self.topDataBox,axis=1)/self.expTime
        bottomData = np.sum(self.bottomDataBox,axis=1)/self.expTime
        # self.test = topData
        self.hist1Plot.setData(topData)
        self.hist2Plot.setData(bottomData)

        
        self.avgFlour1 = np.mean(topData[topData.size//2-5:topData.size//2+5])
        self.FlourLabel1.setText(f'center flour: {self.avgFlour1:.2f}')
        
        self.avgFlour2 = np.mean(bottomData[bottomData.size//2-5:bottomData.size//2+5])
        self.FlourLabel2.setText(f'center flour: {self.avgFlour2:.2f}')
        
    def updateROI(self):
        self.topDataBox = self.ROItop.getArrayRegion(self.lastImage, self.img)
        self.bottomDataBox = self.ROIbottom.getArrayRegion(self.lastImage, self.img)
        
        
        
    def makeROI(self):
        self.ROItop = pg.ROI([100,500],[100,100],pen = pg.mkPen(color='r',width=2))
        self.video.addItem(self.ROItop)
        self.ROItop.addScaleHandle([1,0.5], [0,0.5])
        self.ROItop.addScaleHandle([0.5,1], [0.5,0.5])
        self.ROItop.addTranslateHandle([0.5,0.5])
        
        self.ROIbottom = pg.ROI([100,100],[100,100],pen = pg.mkPen(color='y',width=2))
        self.video.addItem(self.ROIbottom)
        self.ROIbottom.addScaleHandle([1,0.5], [0,0.5])
        self.ROIbottom.addScaleHandle([0.5,1], [0.5,0.5])
        self.ROIbottom.addTranslateHandle([0.5,0.5])
             
        
    def takeBackground(self):
        if self.averageImagesBool:
            self.backgroundImage = np.mean(self.imageArray[:,:,:self.avgNumber],axis=2)
        else:
            self.backgroundImage = np.copy(self.lastImage)
        
    # def updateBG(self):
    #     if self.smoothBackgroundBool:
    #         self.backgroundImage = convolve2d(self.backgroundImageRaw,np.ones((self.smoothValue,self.smoothValue))/self.smoothValue**2,mode='same',boundary='fill')
    #     else:
    #         self.backgroundImage = self.backgroundImageRaw
        
    def checkBGstate(self,state):
        if state == Qt.Checked:
            self.removeBackgroundBool = True
        else:
            self.removeBackgroundBool = False

    def checkDivideState(self,state):
        if state == Qt.Checked:
            self.divideBackgroundBool = True
            # self.updateBG()
        else:
            self.divideBackgroundBool = False
            
    # def checkSmoothBGState(self,state):
    #     if state == Qt.Checked:
    #         self.smoothBackgroundBool = True
    #     else:
    #         self.smoothBackgroundBool = False
        
    def checkAnalyseState(self,state):
        if state == Qt.Checked:
            self.analyseBeamBool = True
        else:
            self.analyseBeamBool = False
            
    def checkCoil1State(self,state):
        if state == Qt.Checked:
            self.coil1PowerBool = True
            self.coilPSU1.output(siglent.CHANNEL.CH1, siglent.STATE.ON)
        else:
            self.coil1PowerBool = False
            self.coilPSU1.output(siglent.CHANNEL.CH1, siglent.STATE.OFF)
    
    def sliderBeamChanged(self):
        self.beamCutoffValue = self.sliderBeam.value()/100
        self.labelBeamSlider.setText(str(self.sliderBeam.value()) + '% of max (taken in red box)')
    
    def sliderCoil1Changed(self):
        self.coil1CurrentValue = self.sliderCoil1.value()/10
        self.labelCoil1Slider.setText(str(self.sliderCoil1.value()/10) + 'A')
        self.coilPSU1.set(siglent.CHANNEL.CH1, siglent.PARAMETER.CURRENT, self.coil1CurrentValue)
        
    def saveImage(self):
        self.saveImageBool = True
        self.fileNameImage = self.imageFileNameText + datetime.now().strftime("%d-%m-%Y_%H_%M_%S_%f")
    
    def imageFileNameChanged(self,text):
        self.imageFileNameText = text
    
    def arrayFileNameChanged(self,text):
        self.arrayFileNameText = text
        
    def checkAvgState(self,state):
        if state == Qt.Checked:
            self.averageImagesBool = True
        else:
            self.averageImagesBool = False
    
    def avgNumberChanged(self,number):
        self.avgNumber = int(number)
    
    def saveArray(self):
        #Save self.avgImage
        self.fileNameArray = self.arrayFileNameText + datetime.now().strftime("%d-%m-%Y_%H_%M_%S_%f")
        np.save(self.fileNameArray,self.imageArray[:,:,:self.avgNumber])
    
    def closeEvent(self, evt):
        self.timer.stop()
        if not self.demo:
            self.camera.endAcquisition()
            self.camera.stop()
        # self.cam.release_instance()
        # del self.camera
        



#%%
app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())



#%%
if False:
    test = main.base_image
    kernel = np.ones((11,1))/11
    # kernel[0,:]=1
    test2 = convolve2d(test,kernel,'same')