import cv2
from PersonData import * 
from assignValues import *
from threading import Thread, Event
from yolo import YOLO
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort import preprocessing
from PIL import Image
import math  
from kalman import *
import random


class ReusableThread(Thread):
    """
    This class provides code for a restartale / reusable thread

    join() will only wait for one (target)functioncall to finish
    finish() will finish the whole thread (after that, it's not restartable anymore)
        
    """

    def __init__(self):
        self._startSignal = Event()
        self._oneRunFinished = Event()
        self._finishIndicator = False
        self.first=True

        Thread.__init__(self)

    def restart(self,frame,cameras,index,frame_index):
        """make sure to always call join() before restarting"""
        self.frame=frame
        self.cameras=cameras
        self.index=index
        self.frame_index=frame_index
        self._startSignal.set()

    def run(self):
        """ This class will reprocess the object "processObject" forever.
        Through the change of data inside processObject and start signals
        we can reuse the thread's resources"""

        self.restart(None,None,None,None)
        while(True):    
            # wait until we should process
            self._startSignal.wait()

            self._startSignal.clear()

            if(self._finishIndicator):# check, if we want to stop
                self._oneRunFinished.set()
                return
            
            if(self.first):
                self.yolo=YOLO()
                self.first=False
            else:
                #changing image from bgr to rgb
                image = Image.fromarray(self.frame[...,::-1])  # bgr to rgb
                #running yolo
                boxes, confidence, classes = self.yolo.detect_image(image)

                #Getting bounding boxes from image data

                detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                            zip(boxes, confidence, classes)]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
                detected_outputs = [detections[i] for i in indices]

                #writing person detection accuracy and putting a bounding boxes around people
                #for det in detections[self.index]:
                #    bbox = det.to_tlbr()
                #    score = "%.2f" % round(det.confidence * 100, 2) + "%"
                    #cv2.rectangle(frame[self.index], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                    #if len(classes) > 0:
                        #cls = det.cls
                        #cv2.putText(frame[self.index], str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                        #            1e-3 * frame[self.index].shape[0], (0, 255, 0), 1)

                #changing rgb image data to hsv for hsv histogram
                hsvImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

                #the local hungarian matrix
                hungarianmatrix=[]
                #the local hungarian matrix data indexes
                hungarianDataIndex=[]
                #index for hungerian matrix
                indexx=0
                #checking the number of previous person data stored
                nodata=len(self.cameras[self.index].PersonData);
                #Setting all person updated variable to false
                for ind in range(nodata):
                    if(self.cameras[self.index].PersonData[ind].isDisabled):
                        continue;
                    self.cameras[self.index].PersonData[ind].kf.predict();
                    self.cameras[self.index].PersonData[ind].updated=False;
                    score=self.frame_index-self.cameras[self.index].PersonData[ind].lastFrame;
                    kalman_pos=self.cameras[self.index].PersonData[ind].kf.x
                    #cv2.putText(frame[self.index],str(self.cameras[self.index].PersonData[ind].localPersonIndex) ,(int(self.cameras[self.index].PersonData[ind].top+kalman_pos[0][0]*(score)), int(self.cameras[self.index].PersonData[ind].middle+kalman_pos[1][0]*(score))),0, 1e-3 * frame[self.index].shape[0], (0,0,255),1)
                    ypos=self.cameras[self.index].PersonData[ind].top-self.cameras[self.index].PersonData[ind].kf.x[0][0]*(score)
                    xpos=self.cameras[self.index].PersonData[ind].middle-self.cameras[self.index].PersonData[ind].kf.x[2][0]*(score)
                    if(xpos<0 or xpos>self.frame.shape[0] or ypos<0 or ypos>self.frame.shape[1] or self.cameras[self.index].PersonData[ind].totalFrames<5):
                        score+=80
                    if(score>=90):
                        self.cameras[self.index].PersonData[ind].isDisabled=True
                        continue;
                    hungarianDataIndex.append(ind)

                    

                #iterating through current detected_outputs
                for det in detected_outputs:
                    #getting top, left, bottom and right co-ordinated from bounding boxes
                    bbox = det.to_tlbr()
                    #checking if there was no previous person data, initializing all found out persons directly to camera's person data variable
                    if(nodata==0):
                        persondata=PersonData()
                        #persondata.color=[int(random.randint(0,255)),int(random.randint(0,255)),int(random.randint(0,255))]
                        persondata.top=(bbox[0]+bbox[2])/2
                        persondata.middle=bbox[1]
                        persondata.left=bbox[1]
                        persondata.positions.append([persondata.top,persondata.middle])
                        persondata.lastPosition=bbox
                        persondata.localPersonIndex=self.cameras[self.index].localPersonCount;
                        #persondata.kf=KalmanFilter([[bbox[0]],[0],[persondata.positions[0][1]],[0]],0.25)
                        persondata.kf=KalmanFilter([[0],[0],[0],[0]],0.07)
                        persondata.globalPersonIndex=-1;
                        #localgloballink.append([globalPersonCount,index,persondata.localPersonIndex])
                        #globalPersonCount=globalPersonCount+1
                        self.cameras[self.index].localPersonCount=self.cameras[self.index].localPersonCount+1;
                        hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                        hist=cv2.calcHist([hsvCroppedImage], [0,1], None, [180,256], [0,180,0,256])
                        persondata.histogram_h = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
                        #persondata.histogram_h = cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180]).flatten()
                        #dividing the hisogram by area so that person bounding box area wont alter histogram values
                        #persondata.histogram_h = np.divide(np.subtract(persondata.histogram_h,persondata.histogram_h.min()),persondata.histogram_h.max()-persondata.histogram_h.min())
                        #adding newly created person object to camera
                        self.cameras[self.index].PersonData.append(persondata)
                    else:
                        hungarianmatrix.append([])
                        #getting current hsv value from current frame
                        hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                        hist=cv2.calcHist([hsvCroppedImage], [0,1], None, [180,256], [0,180,0,256])
                        histogram_h = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
                        #histogram_h = cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180]).flatten()
                        #histogram_h = np.divide(np.subtract(histogram_h,histogram_h.min()),histogram_h.max()-histogram_h.min())

                        for z in range(len(self.cameras[self.index].PersonData)):
                            if(self.cameras[self.index].PersonData[z].isDisabled):
                                continue;
                            kalman_pos=self.cameras[self.index].PersonData[z].kf.x
                            #cov = np.cov(np.asarray(self.cameras[self.index].PersonData[z].positions).T)
                            #mahal=(distance.mahalanobis([self.cameras[self.index].PersonData[z].kf.calulatedmean[0],self.cameras[self.index].PersonData[z].kf.calulatedmean[2]],[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2],cov))/ self.frame.shape[0]
                            #mahal=math.sqrt((self.cameras[self.index].PersonData[z].positions[postions][0]-(bbox[0]+bbox[2])/2)**2+(self.cameras[self.index].PersonData[z].positions[postions][1]-(bbox[1]+bbox[3])/2)**2)/ (self.frame.shape[0]*3)
                            
                            #if(len(self.cameras[self.index].PersonData[z].positions)>4):
                            #    mahal=mahalanobis([bbox[0],(bbox[1]+bbox[3])/2],
                            #    self.cameras[self.index].PersonData[z].positions+[[self.cameras[self.index].PersonData[z].top+kalman_pos[0][0]*(self.frame_index- self.cameras[self.index].PersonData[z].lastFrame),
                            #    self.cameras[self.index].PersonData[z].middle+kalman_pos[2][0]*(self.frame_index- self.cameras[self.index].PersonData[z].lastFrame)]])/ (self.frame.shape[0]*5)
                            #    mahal+=math.sqrt(((self.cameras[self.index].PersonData[z].top+kalman_pos[0][0]*(self.frame_index- self.cameras[self.index].PersonData[z].lastFrame)-bbox[0])**4+(self.cameras[self.index].PersonData[z].middle+kalman_pos[2][0]*(self.frame_index- self.cameras[self.index].PersonData[z].lastFrame)-(bbox[1]+bbox[3])/2)**4))/ (self.frame.shape[0]*5)
                            #else:
                            mahal=math.pow(((self.cameras[self.index].PersonData[z].top+kalman_pos[0][0]*(self.frame_index- self.cameras[self.index].PersonData[z].lastFrame)-(bbox[0]+bbox[2])/2)**2+(self.cameras[self.index].PersonData[z].middle+kalman_pos[2][0]*(self.frame_index- self.cameras[self.index].PersonData[z].lastFrame)-bbox[1])**2),0.71)/ ((self.frame.shape[0]+self.frame.shape[1]))
                            #mahal+=math.sqrt(((self.cameras[self.index].PersonData[z].top-(bbox[0]+bbox[2])/2)**4+(self.cameras[self.index].PersonData[z].middle-bbox[1])**4))/ ((self.frame.shape[0]+self.frame.shape[1])*4)
                            #mahal+=math.pow(((self.cameras[self.index].PersonData[z].top-(bbox[0]+bbox[2])/2)**2+(self.cameras[self.index].PersonData[z].middle-bbox[1])**2),2)/ ((self.frame.shape[0]+self.frame.shape[1]))

                            #mahal=math.sqrt(((self.cameras[self.index].PersonData[z].top+kalman_pos[0][0]*(self.frame_index- self.cameras[self.index].PersonData[z].lastFrame)-bbox[0])**2+(self.cameras[self.index].PersonData[z].middle+kalman_pos[2][0]*(self.frame_index- self.cameras[self.index].PersonData[z].lastFrame)-(bbox[1]+bbox[3])/2)**2))/ (self.frame.shape[0])
                            #mahal=math.sqrt((self.cameras[self.index].PersonData[z].kf.calulatedmean[0]-(bbox[0]+bbox[2])/2)**2+(self.cameras[self.index].PersonData[z].kf.calulatedmean[1]-(bbox[1]+bbox[3])/2)**2)/ self.frame.shape[0]
                            #mahal=getMahalanbolisDist(self.cameras[self.index].PersonData[z].positions,[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
                            #mahal=(np.sum(np.absolute(np.subtract(histogram_h,self.cameras[self.index].PersonData[z].histogram_h))))/(bbox[1]+bbox[3])/10
                            #mahal=(ks_2samp(histogram_h,self.cameras[self.index].PersonData[z].histogram_h))[1]
                            #mahal=(distance.cosine(histogram_h,self.cameras[self.index].PersonData[z].histogram_h))*2 # is the best fit
                            mahal+=cv2.compareHist(histogram_h, self.cameras[self.index].PersonData[z].histogram_h, cv2.HISTCMP_BHATTACHARYYA)**4*1.5
                            #mahal+=(ttest_ind(histogram_h,self.cameras[self.index].PersonData[z].histogram_h))[1]
                            #mahal*=(1000/(1000+self.cameras[self.index].PersonData[z].totalFrames))
                            if(self.cameras[self.index].PersonData[z].totalFrames<5):
                                mahal+=(5-self.cameras[self.index].PersonData[z].totalFrames)/40
                            hungarianmatrix[indexx].append(mahal)
                        indexx=indexx+1
                #print(hungarianmatrix)
                if(nodata!=0):
                    row_ind=[]
                    col_ind=[]
                    if(len(hungarianmatrix)!=0 and len(hungarianmatrix[0])!=0):
                        row_ind, col_ind=assignValues(hungarianmatrix)
                    indexx=0;
                    for pos in range(len(col_ind)):
                        if(hungarianmatrix[row_ind[pos]][col_ind[pos]]<2-detected_outputs[row_ind[pos]].confidence):
                            bbox=detected_outputs[row_ind[pos]].to_tlbr()
                            detected_outputs[row_ind[pos]].localProcessed=True
                            self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].updated=True
                            lastTop=self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].top
                            lastMiddle=self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].middle
                            self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].top=(bbox[0]+bbox[2])/2
                            self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].left=bbox[1]
                            self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].middle=bbox[1]
                            self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].totalFrames+=1
                            #self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.update([[bbox[0]],[0],[(bbox[1]+bbox[3])/2],[0]])
                            #self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.update([[(bbox[0]-lastTop)/(0.1*(self.frame_index-self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))],[0],[(bbox[1]-lastLeft)/(0.1*(self.frame_index-self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))],[0]])
                            vy=((bbox[0]+bbox[2])/2-lastTop)/((self.frame_index-self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))
                            vx=(bbox[1]-lastMiddle)/((self.frame_index-self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))
                            toadd=(detected_outputs[row_ind[pos]].confidence-0.5)**2
                            #if(self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].totalFrames<5):
                            #    toadd=0.5
                            #self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.update([[vy*part+self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.x[0][0]*(1-part)],[0.5],[vx*part+self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.x[2][0]*(1-part)],[0.5]])
                            self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.update([[vy],[0],[vx],[0]])
                            #self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.u=[[(bbox[0]-lastTop)/(0.1*(self.frame_index-self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))],[0.5],[(bbox[1]-lastLeft)/(0.1*(self.frame_index-self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))],[0.5]]
                            self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame=self.frame_index
                            self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].lastPosition=bbox
                            self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].positions.append([self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].top,self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].middle])
                            hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

                            #hist=np.subtract(np.divide(cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180]).flatten(),(((bbox[3]-bbox[1])*(bbox[2]-bbox[0])))),self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h)
                            #self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h = np.add(np.multiply(hist.max()-hist,toadd),np.multiply(self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h,1-toadd))
                            #hist=cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180]).flatten()
                            #hist=np.divide(np.subtract(hist,hist.min()),hist.max()-hist.min())
                            hist=cv2.calcHist([hsvCroppedImage], [0,1], None, [180,256], [0,180,0,256])
                            #self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
                            self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h = np.add(np.multiply(hist,toadd),np.multiply(self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h,1-toadd))
                            #self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h=np.divide(np.subtract(self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h,self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h.min()),self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h.max()-self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h.min())
                            if(len(self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].positions)>6):
                                self.cameras[self.index].PersonData[hungarianDataIndex[col_ind[pos]]].positions.pop(0);

                    for pos in range(len(detected_outputs)):
                        if(hasattr(detected_outputs[pos], 'localProcessed')==False):
                            bbox = detected_outputs[pos].to_tlbr()
                            #if(bbox[1]>hsvImage.shape[0]):
                            #    continue
                            ndata=PersonData()
                            ndata.top=(bbox[0]+bbox[2])/2
                            ndata.left=bbox[1]
                            ndata.middle=bbox[1]
                            ndata.positions.append([ndata.top,ndata.middle])
                            ndata.color=[int(random.randint(0,255)),int(random.randint(0,255)),int(random.randint(0,255))]
                            ndata.localPersonIndex=self.cameras[self.index].localPersonCount
                            ndata.lastPosition=bbox
                            ndata.lastFrame=self.frame_index

                            ndata.kf=KalmanFilter([[0],[0],[0],[0]],0.07)
                            #ndata.kf=KalmanFilter([[bbox[0]],[0],[(bbox[1]+bbox[3])/2],[0]],0.25)
                            self.cameras[self.index].localPersonCount=self.cameras[self.index].localPersonCount+1
                            #localgloballink.append([globalPersonCount,index,ndata.localPersonIndex])
                            #ndata.globalPersonIndex=globalPersonCount
                            #globalPersonCount=globalPersonCount+1
                            hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

                            hist=cv2.calcHist([hsvCroppedImage], [0,1], None, [180,256], [0,180,0,256])
                            ndata.histogram_h = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

                            #ndata.histogram_h = cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180]).flatten()
                            #ndata.histogram_h = np.divide(np.subtract(ndata.histogram_h,ndata.histogram_h.min()),ndata.histogram_h.max()-ndata.histogram_h.min())
                            self.cameras[self.index].PersonData.append(ndata)

            # notify about the run's end
            self._oneRunFinished.set()

    def join(self):
        """ This join will only wait for one single run (target functioncall) to be finished"""
        self._oneRunFinished.wait()
        self._oneRunFinished.clear()

    def finish(self):
        self._finishIndicator = True
        self.restart()
        self.join()