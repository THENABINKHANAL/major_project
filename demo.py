#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from yolo import YOLO
import math  
from scipy.stats import ks_2samp
from win32api import GetSystemMetrics
import scipy as sp
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import motmetrics as mm
acc = mm.MOTAccumulator(auto_id=True)
model = tf.keras.models.load_model('my_model.h5')
#from videocaptureasync import VideoCaptureAsync
import random

import csv

#gt=[];
#with open('gt.txt') as csv_file:
#    csv_reader = csv.reader(csv_file, delimiter=',')
#    line_count = 0
#    for row in csv_reader:
#        gt.append([])
#        for col in range(6):
#            if col<2:
#                gt[line_count].append(int(row[col]))
#            else:
#                gt[line_count].append(float(row[col]))
#        line_count=line_count+1

screenWidth=GetSystemMetrics(0)
screenHeight=GetSystemMetrics(1)

warnings.filterwarnings('ignore')
def fx(x, dt):
    # state transition function - predict next state based
    # on constant velocity model x = vt + x_0
    F = np.array([[1, dt, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, dt],
               [0, 0, 0, 1]], dtype=float)
    return np.dot(F, x)
    
def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos]
    return np.array([x[0], x[2]])

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data, axis=0)
    if not cov:
        cov = np.cov(np.matrix(data).T)
    if(sp.linalg.det(cov)==0):
        inv_covmat = sp.linalg.pinv(cov)
    else:
        inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal

class clique:
    


    # Function to check if the given set of vertices
    # in self.store array is a clique or not


    def is_clique(self,b):

        # Run a loop for all the set of edges
        # for the select vertex
        for i in range(1, b):
            for j in range(i + 1, b):

                # If any edge is missing
                if (self.graph[self.store[i]][self.store[j]] == 0):
                    return False;

        return True;

    # Function to print the clique


    def print_cli(self,n):
        self.output.append([])
        for i in range(1, n):
            self.output[len(self.output)-1].append(self.store[i])

    # Function to find all the cliques of size s


    def findCliques(self,i, l, s):

        # Check if any vertices from i+1 can be inserted
        for j in range(i + 1, self.n - (s - l) + 1):

            # If the degree of the self.graph is sufficient
            if (self.d[j] >= s - 1):

                # Add the vertex to self.store
                self.store[l] = j;

                # If the self.graph is not a clique of size k
                # then it cannot be a clique
                # by adding another edge
                if (self.is_clique(l + 1)):

                    # If the length of the clique is
                    # still less than the desired size
                    if (l < s):

                        # Recursion to add vertices
                        self.findCliques(j, l + 1, s);

                    # Size is met
                    else:
                        self.print_cli(l + 1);


    # Driver code
    def __init__(self,edges,k,n):

        MAX = 100;

        # Stores the vertices
        self.store = [0] * MAX;

        # Graph
        self.graph = np.zeros((MAX, MAX));
        self.n=n
        # Degree of the vertices
        self.d = [0] * MAX;

        self.output = []

        size = len(edges);

        for i in range(size):
            self.graph[edges[i][0]][edges[i][1]] = 1;
            self.graph[edges[i][1]][edges[i][0]] = 1;
            self.d[edges[i][0]] += 1;
            self.d[edges[i][1]] += 1;

        self.findCliques(0, 1, k)

    def output(self):
        return self.output

class cliques:
    def __init__(self,edges,k,n):
        self.data=[]
        while k!=1:
            output=clique(edges,k,n).output
            self.data[len(self.data):]=self.merge(output)
            for sdata in range(len(output)):
                for data in output[sdata]:
                    for edge in edges:
                        if(edge[0]==data or edge[1]==data):
                            edges.remove(edge)
            k=k-1
            if(len(edges) is 0):
                break;

    def getCliques(self):            
        return self.data

    def merge(self,lists, results=None):
        
        if results is None:
            results = []

        if not lists:
            return results

        first = lists[0]
        merged = []
        output = []

        for li in lists[1:]:
            for i in first:
                if i in li:
                    merged = merged + li
                    break
            else:
                output.append(li)

        merged = merged + first
        results.append(list(set(merged)))
        return self.merge(output, results)
        
def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
solve_cudnn_error()

class PersonData:
    def __init__(self):
        self.positions=[]
        self.middle=0
        self.top=0
        self.left=0
        self.localPersonIndex=0
        self.globalPersonIndex=0
        self.globalFoundOutPersonIndex=-1
        self.globalSameTimes=1
        self.prvglobalFoundOutPersonIndex=-1
        self.kf=None
        self.histogram_h=[]
        self.histogram_h2=[]
        self.lastPosition=[]
        self.color=None
        self.updated=True
        self.imgs=[]
        self.lastFrame=0
        self.globaldissimilarity=1
        self.totalFrames=0
        self.isDisabled=False
        
class GlobalPersonData:
    def __init__(self):
        self.histogram_h=[]
        self.personIndexes=[]
        self.personzindexinCameras=[]
        self.personImages=[]

class KalmanFilter:
    def __init__(self, x,std_meas):
        self.dt = 0.1
        self.A = np.array([[1, self.dt,0,0],
                            [0, 1,0,0],
                            [0,0,1,self.dt],
                            [0,0,0,1]])
        self.B = np.array([[(self.dt**2)/2,0,0,0],[0,self.dt,0,0],[0,0,(self.dt**2)/2,0],[0,0,0,self.dt]]) 
        self.H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.Q = np.array([[(self.dt**4)/4, (self.dt**3)/2,0,0],
                            [(self.dt**3)/2, self.dt**2,0,0],
                            [0,0,(self.dt**4)/4, (self.dt**3)/2],
                            [0,0,(self.dt**3)/2, self.dt**2]])
        self.R = [[std_meas**2,0,0,0],[0,std_meas**2,0,0],[0,0,std_meas**2,0],[0,0,0,std_meas**2]]
        self.P = np.eye(self.A.shape[1])
        self.x = x
        self.u = np.array([[0],[0],[0],[0]])
    def predict(self):
        # Ref :Eq.(9) and Eq.(10)

        # Update time state
        #self.x = np.dot(self.A, self.x)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x
    def update(self, z):
            # Ref :Eq.(11) , Eq.(11) and Eq.(13)
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Eq.(11)

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))  # Eq.(12)

        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P  # Eq.(13)

class Camera:
    def __init__(self):
        self.PersonData=[]
        self.localPersonCount=0

def assignValues(a):
    b=np.transpose(a)
    output=[]
    output2=[]
    output1=[]
    minValues=[]
    outputAdded=0
    for i in range(len(b)):
        output.append(-1)
        minValues.append([i,b[i].min()])
    minLength=min(len(b),len(b[0]))
    while outputAdded<minLength:
        minValues = sorted(minValues, key=lambda a_entry: a_entry[1])
        j=0;
        for k in range(len(minValues)):
            minpos=np.argmin(b[minValues[j][0]])
            if(minpos in output):
                b[minValues[j][0]][minpos]=100
                minValues[j][1]=b[minValues[j][0]].min()
                break;
            outputAdded+=1
            output[minValues[j][0]]=minpos
            minValues.pop(j)
    for i in range(len(output)):
        if(output[i]==-1):
            continue
        output1.append(i)
        output2.append(output[i])
    return output2,output1

    
    
    



def find_l2_norm(a,b):
    '''
    a : 2D numpy array
    b : 2D numpy array
    returns the L2_norm between each vector of a and each vector of b
    if a : 4 x 256 and b : 7 x 256
       output is 4 x 7 norm matrix

     '''
    try:
        a = a.reshape([1,256])
    except:
        pass
    try:
        b = b.reshape([1,256])
    except:
        pass
    dot_product = np.matmul(a, np.transpose(b))
    a = np.square(a)
    b = np.square(b)
    norm = np.sum((np.expand_dims(a,axis=1) + b), axis=2) - 2*dot_product + 1e-6
    norm = np.sqrt(norm)
    return norm

def test(query_img,image_list):
    '''
    query_img : numpy array of image of shape [128 x 64 x 3]
    image_list : numpy array of images of shape [n X 128 x 64 x 3]
    outputs : zip of distance between query_img and each image in image_list with images_list
              in ascending order
    '''
    image_list=np.array(image_list)
    # create feed-dict to feed new data
    query_img = query_img.reshape(1,128,64,3)

    # concatenate the query image and source images to form single tensor
    embeddings = np.vstack((query_img,image_list))
    

    #run the session the object and not the output
    output=model.predict(embeddings)[0]

    #first one is the embedding of the query image
    origin_img = output[0]

    #find the distance between the query image and source images
    #distance function must be same as the distance function used
    #during training. Here L2 norm is used
    # distances=[]
    # for i in range(1,output.shape[0]):
    #     distances.append(np.linalg.norm(origin_img-output[i]))
    distances = find_l2_norm(origin_img,output[1:])
    if distances.shape[0]==1:
        distances=distances.reshape([-1])

    #return distances
    return distances


def main(yolo):

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    tracking = False
    writeVideo_flag = True
    asyncVideo_flag = False
    #file path for videos input
    file_path = ['out_6.mp4']
    #file_path = ['vid_1.mp4','vid_2.mp4','vid_3.mp4','vid_4.mp4']
    #file_path = ['4p-c0.avi','4p-c1.avi','4p-c2.avi','4p-c3.avi']
    #file_path = ['terrace1-c0.avi','terrace1-c1.avi','terrace1-c2.avi','terrace1-c3.avi']
    #calulating number of row and columns based on number of videos input
    cols=math.ceil(math.sqrt(len(file_path)))
    rows=math.ceil(len(file_path)/cols)
    #calulating single video hwight and width based on number of rows/cols and screen width/height
    singleHeight=int(screenHeight/rows)
    singleWidth=int(screenWidth/cols)
    #out_image sent to the screen and file written
    out_image=np.zeros((screenHeight,screenWidth,3), np.uint8)

    #if asyncVideo_flag :
    #    video_capture = VideoCaptureAsync(file_path)
    #else:
    #    video_capture = cv2.VideoCapture(file_path)

    #videos reference to get index later
    video_captures = []
    #number of videos/cameras fed. Used to save person data
    cameras=[]
    #previous time kalman filter was processed
    #prvTimes=[]
    #array to link local index to global person index
    localgloballink=[]
    #number of images saved after a person has been tracked in a single camera
    #imgsSaved=1

    #initializing cameras and video_capture variables
    for i in range(len(file_path)):
        video_captures.append(cv2.VideoCapture(file_path[i]))
        cameras.append(Camera())
        #prvTimes.append(time.time())

    #for h in range(400):
    #    for i in range(len(video_captures)):
    #        video_captures[i].read();
    #if asyncVideo_flag:
    #    video_capture.start()

    globalPersonData=[]
    if writeVideo_flag:
        #if asyncVideo_flag:
        #    w = int(video_capture.cap.get(3))
        #    h = int(video_capture.cap.get(4))
        #else:
        #setting width and height of video file written
        w = screenWidth
        h = screenHeight
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
    #number of frames processed till now
    frame_index = 0

    #fps
    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    #frames found in the current run
    frame=[]
    #initializing the frame variables
    for file in file_path:
        frame.append(None)

    #the global person count
    globalPersonCount=1
    #frame count for testing with motmetrics
    curFrame=1;
    #global person index
    gtIndex=0;
    #variable to count the current images saved
    cur_save_count=0

    prvGlobalIndexData=[]

    #countframe=0;
    while True:
        cur_save_count=cur_save_count+1
        #image saved in current run
        #allimages=[]
        
        for index in range(len(file_path)):
            #getting current time for kalman filter
            #cur=time.time()
            #reading a frame from video
            #while(countframe<500):
            #    ret, frame[index] = video_captures[index].read()  # frame shape 640*480*3
            #    countframe+=1
            ret, frame[index] = video_captures[index].read()  # frame shape 640*480*3
            if ret != True:
                 break
            #getting current time for file output
            t1 = time.time()
            #changing image from bgr to rgb
            image = Image.fromarray(frame[index][...,::-1])  # bgr to rgb
            #running yolo
            boxes, confidence, classes = yolo.detect_image(image)
            
            #Getting bounding boxes from image data
            if tracking:
                features = encoder(frame[index], boxes)

                detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                              zip(boxes, confidence, classes, features)]
            else:
                detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                              zip(boxes, confidence, classes)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            #Got bounding boxes from image data
            
            #writing person detection accuracy and putting a bounding boxes around people
            for det in detections:
                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2) + "%"
                #cv2.rectangle(frame[index], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                if len(classes) > 0:
                    cls = det.cls
                    #cv2.putText(frame[index], str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                    #            1e-3 * frame[index].shape[0], (0, 255, 0), 1)

            #changing rgb image data to hsv for hsv histogram
            hsvImage = cv2.cvtColor(frame[index], cv2.COLOR_BGR2HSV)

            #the local hungarian matrix
            hungarianmatrix=[]
            #the local hungarian matrix data indexes
            hungarianDataIndex=[]
            #index for hungerian matrix
            indexx=0
            #checking the number of previous person data stored
            nodata=len(cameras[index].PersonData);
            #Setting all person updated variable to false
            for ind in range(nodata):
                if(cameras[index].PersonData[ind].isDisabled):
                    continue;
                cameras[index].PersonData[ind].kf.predict();
                cameras[index].PersonData[ind].updated=False;
                score=frame_index-cameras[index].PersonData[ind].lastFrame;
                kalman_pos=cameras[index].PersonData[ind].kf.x
                #cv2.putText(frame[index],str(cameras[index].PersonData[ind].localPersonIndex) ,(int(cameras[index].PersonData[ind].top+kalman_pos[0][0]*(score)), int(cameras[index].PersonData[ind].middle+kalman_pos[1][0]*(score))),0, 1e-3 * frame[index].shape[0], (0,0,255),1)
                ypos=cameras[index].PersonData[ind].top-cameras[index].PersonData[ind].kf.x[0][0]*(score)
                xpos=cameras[index].PersonData[ind].middle-cameras[index].PersonData[ind].kf.x[2][0]*(score)
                if(xpos<0 or xpos>frame[index].shape[0] or ypos<0 or ypos>frame[index].shape[1] or cameras[index].PersonData[ind].totalFrames<5):
                    score+=80
                if(score>=90):
                    cameras[index].PersonData[ind].isDisabled=True
                    continue;
                hungarianDataIndex.append(ind)

                

            #iterating through current detections
            for det in detections:
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
                    persondata.localPersonIndex=cameras[index].localPersonCount;
                    #persondata.kf=KalmanFilter([[bbox[0]],[0],[persondata.positions[0][1]],[0]],0.25)
                    persondata.kf=KalmanFilter([[0],[0],[0],[0]],0.07)
                    persondata.globalPersonIndex=globalPersonCount;
                    localgloballink.append([globalPersonCount,index,persondata.localPersonIndex])
                    globalPersonCount=globalPersonCount+1
                    cameras[index].localPersonCount=cameras[index].localPersonCount+1;
                    hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                    hist=cv2.calcHist([hsvCroppedImage], [0,1], None, [180,256], [0,180,0,256])
                    persondata.histogram_h = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
                    #persondata.histogram_h = cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180]).flatten()
                    #dividing the hisogram by area so that person bounding box area wont alter histogram values
                    #persondata.histogram_h = np.divide(np.subtract(persondata.histogram_h,persondata.histogram_h.min()),persondata.histogram_h.max()-persondata.histogram_h.min())
                    #adding newly created person object to camera
                    cameras[index].PersonData.append(persondata)
                else:
                    hungarianmatrix.append([])
                    #getting current hsv value from current frame
                    hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                    hist=cv2.calcHist([hsvCroppedImage], [0,1], None, [180,256], [0,180,0,256])
                    histogram_h = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
                    #histogram_h = cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180]).flatten()
                    #histogram_h = np.divide(np.subtract(histogram_h,histogram_h.min()),histogram_h.max()-histogram_h.min())

                    for z in range(len(cameras[index].PersonData)):
                        if(cameras[index].PersonData[z].isDisabled):
                            continue;
                        kalman_pos=cameras[index].PersonData[z].kf.x
                        #cov = np.cov(np.asarray(cameras[index].PersonData[z].positions).T)
                        #mahal=(distance.mahalanobis([cameras[index].PersonData[z].kf.calulatedmean[0],cameras[index].PersonData[z].kf.calulatedmean[2]],[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2],cov))/ frame[index].shape[0]
                        #mahal=math.sqrt((cameras[index].PersonData[z].positions[postions][0]-(bbox[0]+bbox[2])/2)**2+(cameras[index].PersonData[z].positions[postions][1]-(bbox[1]+bbox[3])/2)**2)/ (frame[index].shape[0]*3)
                        
                        #if(len(cameras[index].PersonData[z].positions)>4):
                        #    mahal=mahalanobis([bbox[0],(bbox[1]+bbox[3])/2],
                        #    cameras[index].PersonData[z].positions+[[cameras[index].PersonData[z].top+kalman_pos[0][0]*(frame_index- cameras[index].PersonData[z].lastFrame),
                        #    cameras[index].PersonData[z].middle+kalman_pos[2][0]*(frame_index- cameras[index].PersonData[z].lastFrame)]])/ (frame[index].shape[0]*5)
                        #    mahal+=math.sqrt(((cameras[index].PersonData[z].top+kalman_pos[0][0]*(frame_index- cameras[index].PersonData[z].lastFrame)-bbox[0])**4+(cameras[index].PersonData[z].middle+kalman_pos[2][0]*(frame_index- cameras[index].PersonData[z].lastFrame)-(bbox[1]+bbox[3])/2)**4))/ (frame[index].shape[0]*5)
                        #else:
                        mahal=math.pow(((cameras[index].PersonData[z].top+kalman_pos[0][0]*(frame_index- cameras[index].PersonData[z].lastFrame)-(bbox[0]+bbox[2])/2)**2+(cameras[index].PersonData[z].middle+kalman_pos[2][0]*(frame_index- cameras[index].PersonData[z].lastFrame)-bbox[1])**2),0.71)/ ((frame[index].shape[0]+frame[index].shape[1]))
                        #mahal+=math.sqrt(((cameras[index].PersonData[z].top-(bbox[0]+bbox[2])/2)**4+(cameras[index].PersonData[z].middle-bbox[1])**4))/ ((frame[index].shape[0]+frame[index].shape[1])*4)
                        #mahal+=math.pow(((cameras[index].PersonData[z].top-(bbox[0]+bbox[2])/2)**2+(cameras[index].PersonData[z].middle-bbox[1])**2),2)/ ((frame[index].shape[0]+frame[index].shape[1]))

                        #mahal=math.sqrt(((cameras[index].PersonData[z].top+kalman_pos[0][0]*(frame_index- cameras[index].PersonData[z].lastFrame)-bbox[0])**2+(cameras[index].PersonData[z].middle+kalman_pos[2][0]*(frame_index- cameras[index].PersonData[z].lastFrame)-(bbox[1]+bbox[3])/2)**2))/ (frame[index].shape[0])
                        #mahal=math.sqrt((cameras[index].PersonData[z].kf.calulatedmean[0]-(bbox[0]+bbox[2])/2)**2+(cameras[index].PersonData[z].kf.calulatedmean[1]-(bbox[1]+bbox[3])/2)**2)/ frame[index].shape[0]
                        #mahal=getMahalanbolisDist(cameras[index].PersonData[z].positions,[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
                        #mahal=(np.sum(np.absolute(np.subtract(histogram_h,cameras[index].PersonData[z].histogram_h))))/(bbox[1]+bbox[3])/10
                        #mahal=(ks_2samp(histogram_h,cameras[index].PersonData[z].histogram_h))[1]
                        #mahal=(distance.cosine(histogram_h,cameras[index].PersonData[z].histogram_h))*2 # is the best fit
                        mahal+=cv2.compareHist(histogram_h, cameras[index].PersonData[z].histogram_h, cv2.HISTCMP_BHATTACHARYYA)**4*1.5
                        #mahal+=(ttest_ind(histogram_h,cameras[index].PersonData[z].histogram_h))[1]
                        #mahal*=(1000/(1000+cameras[index].PersonData[z].totalFrames))
                        if(cameras[index].PersonData[z].totalFrames<5):
                            mahal+=(5-cameras[index].PersonData[z].totalFrames)/40
                        hungarianmatrix[indexx].append(mahal)
                    indexx=indexx+1
            #print(hungarianmatrix)
            if(nodata!=0):
                row_ind=[]
                col_ind=[]
                if(hungarianmatrix!=[]):
                    row_ind, col_ind=assignValues(hungarianmatrix)
                indexx=0;
                for pos in range(len(col_ind)):
                    if(hungarianmatrix[row_ind[pos]][col_ind[pos]]<2-detections[row_ind[pos]].confidence):
                        bbox=detections[row_ind[pos]].to_tlbr()
                        detections[row_ind[pos]].localProcessed=True
                        cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].updated=True
                        lastTop=cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].top
                        lastMiddle=cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].middle
                        cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].top=(bbox[0]+bbox[2])/2
                        cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].left=bbox[1]
                        cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].middle=bbox[1]
                        cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].totalFrames+=1
                        #cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.update([[bbox[0]],[0],[(bbox[1]+bbox[3])/2],[0]])
                        #cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.update([[(bbox[0]-lastTop)/(0.1*(frame_index-cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))],[0],[(bbox[1]-lastLeft)/(0.1*(frame_index-cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))],[0]])
                        vy=((bbox[0]+bbox[2])/2-lastTop)/((frame_index-cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))
                        vx=(bbox[1]-lastMiddle)/((frame_index-cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))
                        toadd=(detections[row_ind[pos]].confidence-0.5)**2
                        #if(cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].totalFrames<5):
                        #    toadd=0.5
                        #cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.update([[vy*part+cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.x[0][0]*(1-part)],[0.5],[vx*part+cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.x[2][0]*(1-part)],[0.5]])
                        cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.update([[vy],[0],[vx],[0]])
                        #cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].kf.u=[[(bbox[0]-lastTop)/(0.1*(frame_index-cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))],[0.5],[(bbox[1]-lastLeft)/(0.1*(frame_index-cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame))],[0.5]]
                        cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].lastFrame=frame_index
                        cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].lastPosition=bbox
                        cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].positions.append([cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].top,cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].middle])
                        hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

                        #hist=np.subtract(np.divide(cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180]).flatten(),(((bbox[3]-bbox[1])*(bbox[2]-bbox[0])))),cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h)
                        #cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h = np.add(np.multiply(hist.max()-hist,toadd),np.multiply(cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h,1-toadd))
                        #hist=cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180]).flatten()
                        #hist=np.divide(np.subtract(hist,hist.min()),hist.max()-hist.min())
                        hist=cv2.calcHist([hsvCroppedImage], [0,1], None, [180,256], [0,180,0,256])
                        #cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
                        cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h = np.add(np.multiply(hist,toadd),np.multiply(cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h,1-toadd))
                        #cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h=np.divide(np.subtract(cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h,cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h.min()),cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h.max()-cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].histogram_h.min())
                        if(len(cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].positions)>6):
                            cameras[index].PersonData[hungarianDataIndex[col_ind[pos]]].positions.pop(0);

                for pos in range(len(detections)):
                    if(hasattr(detections[pos], 'localProcessed')==False):
                        bbox = detections[pos].to_tlbr()
                        #if(bbox[1]>hsvImage.shape[0]):
                        #    continue
                        ndata=PersonData()
                        ndata.top=(bbox[0]+bbox[2])/2
                        ndata.left=bbox[1]
                        ndata.middle=bbox[1]
                        ndata.positions.append([ndata.top,ndata.middle])
                        ndata.color=[int(random.randint(0,255)),int(random.randint(0,255)),int(random.randint(0,255))]
                        ndata.localPersonIndex=cameras[index].localPersonCount
                        ndata.lastPosition=bbox
                        ndata.lastFrame=frame_index

                        ndata.kf=KalmanFilter([[0],[0],[0],[0]],0.07)
                        #ndata.kf=KalmanFilter([[bbox[0]],[0],[(bbox[1]+bbox[3])/2],[0]],0.25)
                        cameras[index].localPersonCount=cameras[index].localPersonCount+1
                        localgloballink.append([globalPersonCount,index,ndata.localPersonIndex])
                        ndata.globalPersonIndex=globalPersonCount
                        globalPersonCount=globalPersonCount+1
                        hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

                        hist=cv2.calcHist([hsvCroppedImage], [0,1], None, [180,256], [0,180,0,256])
                        ndata.histogram_h = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

                        #ndata.histogram_h = cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180]).flatten()
                        #ndata.histogram_h = np.divide(np.subtract(ndata.histogram_h,ndata.histogram_h.min()),ndata.histogram_h.max()-ndata.histogram_h.min())
                        cameras[index].PersonData.append(ndata)

            #allimages.append([])
            #if(len(file_path))!=1:
            #    for pdata in cameras[index].PersonData:
            #        if(pdata.updated):
            #            nimg=cv2.resize(frame[index][int(pdata.lastPosition[1]):int(pdata.lastPosition[3]),int(pdata.lastPosition[0]):int(pdata.lastPosition[2])], (64,128
            #            ), interpolation = cv2.INTER_AREA)
            #            #allimages[len(allimages)-1].append(np.array(nimg))
            #            pdata.imgs.append(nimg);
            #            #cv2.imwrite('color_img'+str(frame_index)+str(pdata.globalPersonIndex) +'.jpg', nimg)
            #            if(len(pdata.imgs)==imgsSaved+1):
            #                pdata.imgs.pop(0)
            #nabin's code ends

            if tracking:
                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame[index], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    cv2.putText(frame[index], "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                                1e-3 * frame[index].shape[0], (0, 255, 0), 1)

            #if(len(cameras)==2):
            #globalHungarian=[]
            #    for fdata in range(len(cameras[0].PersonData)):
            #        globalHungarian.append([])
            #        for pdata in cameras[1].PersonData:
            #            globalHungarian[fdata].append(np.sum(np.absolute(np.subtract(pdata.histogram_h,cameras[0].PersonData[fdata].histogram_h))))
            #    
            #    row_ind, col_ind = linear_sum_assignment(globalHungarian)
            #    for row in range(len(row_ind)):
            #        cv2.putText(frame[0], chr(ord('a')+row),(int(cameras[0].PersonData[row_ind[row]].positions[len(cameras[0].PersonData[row_ind[row]].positions)-1][0]), int(cameras[0].PersonData[row_ind[row]].positions[len(cameras[0].PersonData[row_ind[row]].positions)-1][1])),0, 5e-3 * 200, (0,255,0),2)
            #        cv2.putText(frame[1], chr(ord('a')+row),(int(cameras[1].PersonData[col_ind[row]].positions[len(cameras[1].PersonData[col_ind[row]].positions)-1][0]), int(cameras[1].PersonData[col_ind[row]].positions[len(cameras[1].PersonData[col_ind[row]].positions)-1][1])),0, 5e-3 * 200, (0,255,0),2)

        if(len(cameras)==1):
            #hypos=[];
            #hyposPos=[];
            for person in cameras[0].PersonData:
                if(person.updated==True):
                    cv2.putText(frame[0],str(person.localPersonIndex) ,(int(person.top), int(person.middle)),0, 1e-3 * frame[index].shape[0], (0,255,0),1)
                        cv2.rectangle(frame[cam], (int(person.lastPosition[0]), int(person.lastPosition[1])), (int(person.lastPosition[2]), int(person.lastPosition[3])), (255, 0, 0), 2)

                #if(person.updated==True):
                #    hypos.append(person.localPersonIndex+1)
                #    hyposPos.append([person.top,person.left])
            #gts=[]
            #gtsPos=[]
            #while gtIndex<len(gt) and gt[gtIndex][0]==curFrame:
            #    gts.append(gt[gtIndex][1])
            #    gtsPos.append([gt[gtIndex][2],gt[gtIndex][3]])
            #    gtIndex=gtIndex+1
            #curFrame=curFrame+1
            #dis=mm.distances.norm2squared_matrix(np.array(gtsPos), np.array(hyposPos))
            #acc.update(gts,hypos,dis)

        else:
            if len(globalPersonData)!=0:
                for singleglobalPersonData in globalPersonData:
                    singleglobalPersonData.personIndexes=[]
                for k in range(len(cameras)):
                    globalHungarian=[]
                    rowsIndexes=[]
                    for i in range(len(cameras[k].PersonData)):
                        if(cameras[k].PersonData[i].isDisabled):
                            continue;
                        rowsIndexes.append(i)
                        globalHungarian.append([])
                        bbox=cameras[k].PersonData[i].lastPosition
                        img=cv2.resize(frame[index][int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])], (64,128), interpolation = cv2.INTER_AREA)
                        for j in range(len(globalPersonData)):
                            if(globalPersonData[j].personzindexinCameras[k]==i):
                                #if(cameras[k].PersonData[i].globaldissimilarity<0.3):
                                decrement=(1/(1+5/cameras[k].PersonData[i].globalSameTimes))*0.4/((1+cameras[k].PersonData[i].globaldissimilarity)**4)
                                #print("decrement ",decrement)
                                #globalHungarian[len(globalHungarian)-1].append(cv2.compareHist(cameras[k].PersonData[i].histogram_h, globalPersonData[j].histogram_h, cv2.HISTCMP_BHATTACHARYYA)**2-decrement)
                                #val=test(img,globalPersonData[j].personImages)
                                #globalHungarian[len(globalHungarian)-1].append(cv2.compareHist(cameras[k].PersonData[i].histogram_h, globalPersonData[j].histogram_h, cv2.HISTCMP_BHATTACHARYYA)**2*0.5+np.sum(val)/(len(globalPersonData[j].personImages)*1.2)*0.5-decrement)
                                globalHungarian[len(globalHungarian)-1].append(cv2.compareHist(cameras[k].PersonData[i].histogram_h, globalPersonData[j].histogram_h, cv2.HISTCMP_BHATTACHARYYA)**2*2-decrement)
                                #else:                           
                                #    globalHungarian[len(globalHungarian)-1].append(cv2.compareHist(cameras[k].PersonData[i].histogram_h, globalPersonData[j].histogram_h, cv2.HISTCMP_BHATTACHARYYA)**2-(1/(1+1/cameras[k].PersonData[i].globalSameTimes))*0.1)
                            else:
                                #val=test(img,globalPersonData[j].personImages)
                                #globalHungarian[len(globalHungarian)-1].append(cv2.compareHist(cameras[k].PersonData[i].histogram_h, globalPersonData[j].histogram_h, cv2.HISTCMP_BHATTACHARYYA)**2*0.5+np.sum(val)/(len(globalPersonData[j].personImages)*1.2)*0.5)
                                globalHungarian[len(globalHungarian)-1].append(cv2.compareHist(cameras[k].PersonData[i].histogram_h, globalPersonData[j].histogram_h, cv2.HISTCMP_BHATTACHARYYA)**2*2)
                    for i in range(len(cameras[k].PersonData)):
                            cameras[k].PersonData[i].globalFoundOutPersonIndex=-1

                    if(len(globalHungarian)!=0):
                        row_ind, col_ind = assignValues(globalHungarian)
                        #print(globalHungarian);
                        for pos in range(len(row_ind)):
                            if(globalHungarian[row_ind[pos]][col_ind[pos]]<0.6):
                                if(cameras[k].PersonData[rowsIndexes[row_ind[pos]]].prvglobalFoundOutPersonIndex==col_ind[pos]):
                                    cameras[k].PersonData[rowsIndexes[row_ind[pos]]].globalSameTimes+=1
                                else:
                                    cameras[k].PersonData[rowsIndexes[row_ind[pos]]].globalSameTimes=1;

                                globalPersonData[col_ind[pos]].personzindexinCameras[k]=rowsIndexes[row_ind[pos]]
                                cameras[k].PersonData[rowsIndexes[row_ind[pos]]].globalFoundOutPersonIndex=col_ind[pos]
                                cameras[k].PersonData[rowsIndexes[row_ind[pos]]].prvglobalFoundOutPersonIndex=col_ind[pos]
                                #part=(0.5-globalHungarian[row_ind[pos]][col_ind[pos]])**2
                                #if(part>0.2):
                                #    globalPersonData[col_ind[pos]].histogram_h=np.add(np.multiply(globalPersonData[col_ind[pos]].histogram_h,1-part),np.multiply(cameras[k].PersonData[rowsIndexes[row_ind[pos]]].histogram_h,part))
                                globalPersonData[col_ind[pos]].personIndexes.append([k,rowsIndexes[row_ind[pos]]])
                    
                    for singleglobalPersonData in globalPersonData:
                        matrix=np.full((len(singleglobalPersonData.personIndexes),len(singleglobalPersonData.personIndexes)),0.0)
                        for i in range(len(singleglobalPersonData.personIndexes)):
                            for j in range(i+1,len(singleglobalPersonData.personIndexes)):
                                dissimilarity=cv2.compareHist(cameras[singleglobalPersonData.personIndexes[i][0]].PersonData[singleglobalPersonData.personIndexes[i][1]].histogram_h, cameras[singleglobalPersonData.personIndexes[j][0]].PersonData[singleglobalPersonData.personIndexes[j][1]].histogram_h, cv2.HISTCMP_BHATTACHARYYA)**2*1.5
                                matrix[i][j]=dissimilarity
                                matrix[j][i]=dissimilarity
                        dissimilaritySum=np.divide(np.sum(matrix,0),len(singleglobalPersonData.personIndexes))

                        for i in range(len(singleglobalPersonData.personIndexes)):
                            cameras[singleglobalPersonData.personIndexes[i][0]].PersonData[singleglobalPersonData.personIndexes[i][1]].globaldissimilarity=dissimilaritySum[i]
                            #print(dissimilaritySum[i])
                            #print("imgval",test(cameras[singleglobalPersonData.personIndexes[i][0]].PersonData[singleglobalPersonData.personIndexes[i][1]].imgs[0],[cameras[singleglobalPersonData.personIndexes[j][0]].PersonData[singleglobalPersonData.personIndexes[j][1]].imgs[0]]))
                            if(dissimilaritySum[i]<0.4):
                                #singleglobalPersonData.histogram_h=np.add(np.multiply(cameras[singleglobalPersonData.personIndexes[i][0]].PersonData[singleglobalPersonData.personIndexes[i][1]].histogram_h,0.5),np.multiply(cameras[singleglobalPersonData.personIndexes[j][0]].PersonData[singleglobalPersonData.personIndexes[j][1]].histogram_h,0.5))
                                singleglobalPersonData.histogram_h=np.add(np.multiply(singleglobalPersonData.histogram_h,0.8),np.multiply(cameras[singleglobalPersonData.personIndexes[i][0]].PersonData[singleglobalPersonData.personIndexes[i][1]].histogram_h,0.2))

            cur_save_count=0
            edges=[]
            globalHungarian=[]
            allfeatureVector=[]
            newcameradata=[]
            for i in range(len(cameras)):
                stackedimgages=[]
                for pdata in range(len(cameras[i].PersonData)):
                    if(cameras[i].PersonData[pdata].globalFoundOutPersonIndex!=-1 or cameras[i].PersonData[pdata].isDisabled or cameras[i].PersonData[pdata].totalFrames<5):
                        continue;
                    bbox=cameras[i].PersonData[pdata].lastPosition
                    stackedimgages.append(cv2.resize(frame[i][int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])], (64,128), interpolation = cv2.INTER_AREA))
                if(len(stackedimgages)!=0):
                    newcameradata.append(i)
                    m=np.array(stackedimgages)
                    allfeatureVector.append(model.predict(m)[0])
            if(len(allfeatureVector)>=2): 
                actuali=-1
                for i in range(len(cameras)):
                    if(i not in newcameradata):
                        continue;
                    actualj=-1
                    actuali+=1
                    for j in range(i+1,len(cameras)):
                        if(j not in newcameradata):
                            continue;
                        actualj+=1
                        
                        x=0
                        xindexes=[]
                        yindexes=[]
                        #stackedimgages=[]
                        #for pdata in range(len(cameras[j].PersonData)):
                        #    if(cameras[j].PersonData[pdata].globalFoundOutPersonIndex!=-1 or cameras[j].PersonData[pdata].isDisabled or cameras[j].PersonData[pdata].totalFrames<5):
                        #        continue;
                        #    bbox=cameras[j].PersonData[pdata].lastPosition
                        #    stackedimgages.append(cv2.resize(frame[j][int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])], (64,128), interpolation = cv2.INTER_AREA))
                        #for pos in range(imgsSaved):
                        #    stackedimgages.append([])
                        #    for person in cameras[j].PersonData:
                        #        if(person.updated==True and len(person.imgs)==imgsSaved):
                        #            stackedimgages[pos].append(person.imgs[pos])
                        globalHungarian=find_l2_norm(allfeatureVector[actuali],allfeatureVector[actualj])
                        for fdata in range(len(cameras[i].PersonData)):
                            if(cameras[i].PersonData[fdata].globalFoundOutPersonIndex!=-1 or cameras[i].PersonData[fdata].isDisabled or cameras[i].PersonData[fdata].totalFrames<5):
                                continue;
                            #if(cameras[i].PersonData[fdata].updated==False or len(cameras[i].PersonData[fdata].imgs)!=imgsSaved):
                            #if(cameras[i].PersonData[fdata].updated==False or len(cameras[i].PersonData[fdata].imgs)!=imgsSaved):
                            #    continue
                            xindexes.append(fdata)
                            #curclique=[]
                            #prvfoundout=-1
                            #if len(prvGlobalIndexData)!=0:
                            #    for single in prvGlobalIndexData:
                            #        if cameras[i].PersonData[fdata].globalPersonIndex in single:
                            #            curclique=single
                            #            prvfoundout=single[0]
                            #            break;
                            y=0
                            #triplet=test(cameras[i].PersonData[fdata].imgs[0],stackedimgages[0])
                            #for pos in range(1,imgsSaved):
                            #    triplet=np.add(triplet,test(cameras[i].PersonData[fdata].imgs[pos],stackedimgages[pos]))
                            #bbox=cameras[i].PersonData[fdata].lastPosition
                            #triplet=test(cv2.resize(frame[i][int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])], (64,128), interpolation = cv2.INTER_AREA),stackedimgages)
                            #globalHungarian.append([])
                            for pdata in range(len(cameras[j].PersonData)):
                                if(cameras[j].PersonData[pdata].globalFoundOutPersonIndex!=-1 or cameras[j].PersonData[pdata].isDisabled or cameras[j].PersonData[pdata].totalFrames<5):
                                    continue;
                                #if(cameras[j].PersonData[pdata].updated==False or len(cameras[j].PersonData[pdata].imgs)!=imgsSaved):
                                #    continue
                                #globalHungarian[x].append(triplet[y])
                                #val=(np.sum(np.absolute(np.subtract(cameras[j].PersonData[pdata].histogram_h,cameras[i].PersonData[fdata].histogram_h)))+triplet[y])/(0.9+1.4*imgsSaved)#hsv seems to be max 0.9, triplet max seems to be 1.2
                                #if cameras[j].PersonData[pdata].globalPersonIndex in curclique or cameras[j].PersonData[pdata].prvglobalFoundOutPersonIndex==prvfoundout:
                                #    val-=0.2
                                #globalHungarian[x].append(val)
                                #globalHungarian[x][y]=cv2.compareHist(cameras[j].PersonData[pdata].histogram_h, cameras[i].PersonData[fdata].histogram_h, cv2.HISTCMP_BHATTACHARYYA)**2*1.8
                                globalHungarian[x][y]=globalHungarian[x][y]*0.5+cv2.compareHist(cameras[j].PersonData[pdata].histogram_h, cameras[i].PersonData[fdata].histogram_h, cv2.HISTCMP_BHATTACHARYYA)**2
                                if(x==0):
                                    yindexes.append(pdata)
                                #globalHungarian[fdata].append(np.sum(np.absolute(np.subtract(cameras[j].PersonData[pdata].histogram_h,cameras[i].PersonData[fdata].histogram_h))))
                                #globalHungarian[fdata].append(triplet[pdata])
                                y=y+1
                            x=x+1
                        if(len(globalHungarian)!=0 and len(globalHungarian[0])!=0):
                            print(globalHungarian)
                            row_ind, col_ind = assignValues(globalHungarian)
                            for pos in range(len(row_ind)):
                                if(globalHungarian[row_ind[pos]][col_ind[pos]]<0.8):
                                    edges.append((cameras[i].PersonData[xindexes[row_ind[pos]]].globalPersonIndex,cameras[j].PersonData[yindexes[col_ind[pos]]].globalPersonIndex))
                
                Allcliques=cliques(edges,len(cameras),globalPersonCount).getCliques()

                for sclique in Allcliques:
                    globalPersonData.append(GlobalPersonData());
                    bbox=cameras[localgloballink[sclique[0]-1][1]].PersonData[localgloballink[sclique[0]-1][2]].lastPosition
                    globalPersonData[len(globalPersonData)-1].personImages.append(cv2.resize(frame[index][int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])], (64,128), interpolation = cv2.INTER_AREA))
                    globalPersonData[len(globalPersonData)-1].personzindexinCameras=np.full(len(cameras),-1)
                    cameras[localgloballink[sclique[0]-1][1]].PersonData[localgloballink[sclique[0]-1][2]].globalFoundOutPersonIndex=len(globalPersonData)-1
                    globalPersonData[len(globalPersonData)-1].histogram_h=cameras[localgloballink[sclique[0]-1][1]].PersonData[localgloballink[sclique[0]-1][2]].histogram_h
                    for i in range(1,len(sclique)):
                        bbox=cameras[localgloballink[sclique[i]-1][1]].PersonData[localgloballink[sclique[i]-1][2]].lastPosition
                        globalPersonData[len(globalPersonData)-1].personImages.append(cv2.resize(frame[index][int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])], (64,128), interpolation = cv2.INTER_AREA))
                        cameras[localgloballink[sclique[i]-1][1]].PersonData[localgloballink[sclique[i]-1][2]].globalFoundOutPersonIndex=len(globalPersonData)-1
                        globalPersonData[len(globalPersonData)-1].histogram_h=np.multiply(np.add(cameras[localgloballink[sclique[i]-1][1]].PersonData[localgloballink[sclique[i]-1][2]].histogram_h,globalPersonData[len(globalPersonData)-1].histogram_h),0.5)
                

            #for cam in cameras:
            #    for person in cam.PersonData:
            #        if len(person.imgs)!=imgsSaved or person.updated==False:
            #            continue
            #        isinclique=True
            #        for clique in Allcliques:
            #            if person.globalPersonIndex in clique:
            #                isinclique=False
            #                break
            #        if isinclique:
            #            Allcliques.append([person.globalPersonIndex])

                

            for cam in range(len(cameras)):
                for person in cameras[cam].PersonData:
                    if person.updated==True:
                        cv2.rectangle(frame[cam], (int(person.lastPosition[0]), int(person.lastPosition[1])), (int(person.lastPosition[2]), int(person.lastPosition[3])), (255, 0, 0), 2)
                        cv2.putText(frame[cam],str(person.globalFoundOutPersonIndex) ,(int(person.top),int(person.middle)+10),0, 1e-3 * frame[index].shape[0], (0,255,0),2)

            

        out_image.fill(0)
        vindex=0;
        for row in range(rows):
            for col in range(cols):
                if(vindex==len(file_path)):
                    break
                vidshape=frame[vindex].shape
                curvidheightratio=vidshape[0]/singleHeight
                curvidwidthratio=vidshape[1]/singleWidth

                if(curvidwidthratio<curvidheightratio):
                    #height is small
                    resizedwidth=int(vidshape[1]/vidshape[0]*singleHeight)
                    nimg=cv2.resize(frame[vindex], (resizedwidth,singleHeight), interpolation = cv2.INTER_AREA)
                    widthpos=int((singleWidth-resizedwidth)/2)+col*singleWidth
                    out_image[row*singleHeight:(row+1)*singleHeight,widthpos:widthpos+resizedwidth]=nimg
                else:
                    #width is small
                    resizedheight=int(vidshape[0]/vidshape[1]*singleWidth)
                    nimg=cv2.resize(frame[vindex], (singleWidth,resizedheight), interpolation = cv2.INTER_AREA)
                    heightpos=int(((singleHeight-resizedheight)/2)+row*singleHeight)
                    out_image[heightpos:heightpos+resizedheight,col*singleWidth:(col+1)*singleWidth]=nimg
                vindex=vindex+1

        #if(len(cameras)==1):
        #    mh = mm.metrics.create()
        #    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
        #    print(summary)
        cv2.imshow('', out_image)

        if writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            out.write(out_image)
        frame_index = frame_index + 1

        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1./(time.time()-t1))) / 2
            print("FPS = %f"%(fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
