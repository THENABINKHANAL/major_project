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

from win32api import GetSystemMetrics

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import motmetrics as mm
acc = mm.MOTAccumulator(auto_id=True)
model = tf.keras.models.load_model('my_model.h5')
#from videocaptureasync import VideoCaptureAsync
import random

import csv
gt=[];
with open('gt.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        gt.append([])
        for col in range(6):
            if col<2:
                gt[line_count].append(int(row[col]))
            else:
                gt[line_count].append(float(row[col]))
        line_count=line_count+1

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

def getMahalanbolisDist(data,x):
    m=np.mean(data,axis=0)
    xMm=x-m
    data=np.transpose(np.array(data))
    covM=np.cov(data,bias=False)
    det=np.linalg.det(covM)
    if(det==0.0):
        invConveM=covM
    else:
        invConveM=np.linalg.inv(covM)
    tem1=np.dot(xMm,invConveM)
    tem2=np.dot(tem1,np.transpose(xMm))
    return np.sqrt(tem2)

class KF:
    def __init__(self,initial_x,initial_y,initial_vx,initial_vy):
        self.dt = 0.05
        # create sigma points to use in the filter. This is standard for Gaussian processes
        self.points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)

        self.kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=self.dt, fx=fx, hx=hx, points=self.points)
        self.kf.x = np.array([initial_x, 0, initial_y, 0]) # initial state
        self.kf.P *= 0.1 # initial uncertainty
        self.z_std = 0.1
        self.kf.R = np.diag([self.z_std**2, self.z_std**2]) # 1 standard
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.01**2, block_size=2)

    def predict(self):
        return self.kf.predict();
    
    def update(self,meas_value):
        self.kf.update([meas_value[0],meas_value[1]]);


    @property
    def calulatedmean(self):
        return self.kf.x   
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
        


class PersonData:
    def __init__(self):
        self.positions=[]
        self.left=0
        self.top=0
        self.localPersonIndex=0
        self.globalPersonIndex=0
        self.globalFoundOutPersonIndex=-1
        self.prvglobalFoundOutPersonIndex=-1
        self.kf=None
        self.histogram_h=[]
        self.lastPosition=[]
        self.color=None
        self.updated=True
        self.imgs=[]
        

class Camera:
    def __init__(self):
        self.PersonData=[]
        self.localPersonCount=0

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

    file_path = ['vid_1.mp4','vid_2.mp4']
    #file_path = ['veed.mp4']
    cols=math.ceil(math.sqrt(len(file_path)))
    rows=math.ceil(len(file_path)/cols)
    singleHeight=int(screenHeight/rows)
    singleWidth=int(screenWidth/cols)
    out_image=np.zeros((screenHeight,screenWidth,3), np.uint8)

    #if asyncVideo_flag :
    #    video_capture = VideoCaptureAsync(file_path)
    #else:
    #    video_capture = cv2.VideoCapture(file_path)

    video_captures = []
    cameras=[]
    prvTimes=[]
    localgloballink=[]
    imgsSaved=1

    for i in range(len(file_path)):
        video_captures.append(cv2.VideoCapture(file_path[i]))
        cameras.append(Camera())
        prvTimes.append(time.time())

    #if asyncVideo_flag:
    #    video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
            h = int(video_capture.cap.get(4))
        else:
            w = screenWidth
            h = screenHeight
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    frame=[]
    globalPersonCount=1
    for file in file_path:
        frame.append(None)

    curFrame=1;
    gtIndex=0;

    while True:
        allimages=[]
        for index in range(len(file_path)):
            cur=time.time()
            ret, frame[index] = video_captures[index].read()  # frame shape 640*480*3
            if ret != True:
                 break

            t1 = time.time()

            image = Image.fromarray(frame[index][...,::-1])  # bgr to rgb
            boxes, confidence, classes = yolo.detect_image(image)

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

            for det in detections:
                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2) + "%"
                cv2.rectangle(frame[index], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                if len(classes) > 0:
                    cls = det.cls
                    cv2.putText(frame[index], str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                                1e-3 * frame[index].shape[0], (0, 255, 0), 1)

            #nabin's code
            hsvImage = cv2.cvtColor(frame[index], cv2.COLOR_BGR2HSV)

            hungarianmatrix=[]
            indexx=0
            if(len(cameras[index].PersonData)>0):
                diff=cur-prvTimes[index]
                times=int(diff/0.05)
                prvTimes[index]=cur
                for data in cameras[index].PersonData:
                    if(data.kf!=None):
                        for i in range(times):
                            data.kf.predict()
            nodata=len(cameras[index].PersonData);
            for z in range(len(cameras[index].PersonData)):
                cameras[index].PersonData[z].updated=False;
            for det in detections:
                bbox = det.to_tlbr()
                if(nodata==0):
                    persondata=PersonData()
                    persondata.color=[int(random.randint(0,255)),int(random.randint(0,255)),int(random.randint(0,255))]
                    persondata.positions.append([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
                    persondata.positions.append([(bbox[0]+bbox[2])/2+0.1,(bbox[1]+bbox[3])/2+0.1])
                    persondata.top=bbox[0]
                    persondata.left=bbox[1]
                    persondata.lastPosition=bbox
                    persondata.localPersonIndex=cameras[index].localPersonCount;
                    persondata.kf=KF(persondata.positions[0][0],persondata.positions[0][1],0,0)
                    persondata.globalPersonIndex=globalPersonCount;
                    localgloballink.append([globalPersonCount,index,persondata.localPersonIndex])
                    globalPersonCount=globalPersonCount+1
                    cameras[index].localPersonCount=cameras[index].localPersonCount+1;
                    hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                    persondata.histogram_h = cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180])
                    persondata.histogram_h = np.divide(persondata.histogram_h,((bbox[3]-bbox[1])*(bbox[2]-bbox[0])))
                    cameras[index].PersonData.append(persondata)
                else:
                    hungarianmatrix.append([])
                    hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                    histogram_h = cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180])
                    histogram_h = np.divide(histogram_h,((bbox[3]-bbox[1])*(bbox[2]-bbox[0])))

                    for z in range(len(cameras[index].PersonData)):
                        postions=len(cameras[index].PersonData[z].positions)-1
                        cov = np.cov(np.asarray(cameras[index].PersonData[z].positions).T)
                        if(cameras[index].PersonData[z].kf==None):
                            mahal=distance.mahalanobis([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2],cameras[index].PersonData[z].positions[postions],cov)
                        else:
                            #mahal=distance.mahalanobis([cameras[index].PersonData[z].kf.calulatedmean[0],cameras[index].PersonData[z].kf.calulatedmean[2]],[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2],cov)
                            mahal=math.sqrt((cameras[index].PersonData[z].kf.calulatedmean[0]-(bbox[0]+bbox[2])/2)**2+(cameras[index].PersonData[z].kf.calulatedmean[2]-(bbox[1]+bbox[3])/2)**2)
                            #mahal=getMahalanbolisDist(cameras[index].PersonData[z].positions,[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
                            mahal+=(np.sum(np.absolute(np.subtract(histogram_h,cameras[index].PersonData[z].histogram_h))))
                        hungarianmatrix[indexx].append(mahal)
                    indexx=indexx+1
            if(nodata!=0):
                row_ind=[]
                col_ind=[]
                if(hungarianmatrix!=[]):
                    row_ind, col_ind = linear_sum_assignment(hungarianmatrix)
                indexx=0;
                for pos in range(len(col_ind)):
                    if(hungarianmatrix[row_ind[pos]][col_ind[pos]]<100):
                        bbox=detections[row_ind[pos]].to_tlbr()
                        detections[row_ind[pos]].localProcessed=True
                        cameras[index].PersonData[col_ind[pos]].updated=True
                        cameras[index].PersonData[col_ind[pos]].top=bbox[0]
                        cameras[index].PersonData[col_ind[pos]].left=bbox[1]
                        cameras[index].PersonData[col_ind[pos]].kf.update([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
                        cameras[index].PersonData[col_ind[pos]].lastPosition=bbox
                        cameras[index].PersonData[col_ind[pos]].positions.append([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
                        hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                        cameras[index].PersonData[col_ind[pos]].histogram_h = np.add(np.multiply(cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180]),0.5*1/(((bbox[3]-bbox[1])*(bbox[2]-bbox[0])))),np.multiply(cameras[index].PersonData[col_ind[pos]].histogram_h,0.25))
                        if(len(cameras[index].PersonData[col_ind[pos]].positions)>6):
                            cameras[index].PersonData[col_ind[pos]].positions.pop(0);

                for pos in range(len(detections)):
                    if(hasattr(detections[pos], 'localProcessed')==False):
                        bbox = detections[pos].to_tlbr()
                        #if(bbox[1]>hsvImage.shape[0]):
                        #    continue
                        ndata=PersonData()
                        ndata.top=bbox[0]
                        ndata.left=bbox[1]
                        ndata.positions.append([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
                        ndata.positions.append([(bbox[0]+bbox[2])/2+0.1,(bbox[1]+bbox[3])/2+0.1])
                        ndata.kf=KF((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2,0,0)
                        ndata.color=[int(random.randint(0,255)),int(random.randint(0,255)),int(random.randint(0,255))]
                        ndata.localPersonIndex=cameras[index].localPersonCount
                        ndata.lastPosition=bbox

                        ndata.kf=KF(ndata.positions[0][0],ndata.positions[0][1],0,0)
                        cameras[index].localPersonCount=cameras[index].localPersonCount+1
                        localgloballink.append([globalPersonCount,index,ndata.localPersonIndex])
                        ndata.globalPersonIndex=globalPersonCount;
                        globalPersonCount=globalPersonCount+1
                        hsvCroppedImage=hsvImage[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                        ndata.histogram_h = cv2.calcHist([hsvCroppedImage],[0],None,[180],[0,180])
                        ndata.histogram_h = np.divide(ndata.histogram_h,((bbox[3]-bbox[1])*(bbox[2]-bbox[0])))

                        cameras[index].PersonData.append(ndata)

            #allimages.append([])
            for pdata in cameras[index].PersonData:
                if(pdata.updated):
                    nimg=cv2.resize(frame[index][int(pdata.lastPosition[1]):int(pdata.lastPosition[3]),int(pdata.lastPosition[0]):int(pdata.lastPosition[2])], (64,128
                    ), interpolation = cv2.INTER_AREA)
                    #allimages[len(allimages)-1].append(np.array(nimg))
                    pdata.imgs.append(nimg);
                    if(len(pdata.imgs)==imgsSaved+1):
                        pdata.imgs.pop(0)
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
            hypos=[];
            hyposPos=[];
            for person in cameras[0].PersonData:
                cv2.putText(frame[0],str(person.localPersonIndex) ,(int(person.positions[len(person.positions)-1][0]), int(person.positions[len(person.positions)-1][1])),0, 1e-3 * frame[index].shape[0], (0,255,0),1)
                if(person.updated==True):
                    hypos.append(person.localPersonIndex+1)
                    hyposPos.append([person.top,person.left])
            gts=[]
            gtsPos=[]
            while gt[gtIndex][0]==curFrame and gtIndex<len(gt):
                gts.append(gt[gtIndex][1])
                gtsPos.append([gt[gtIndex][2],gt[gtIndex][3]])
                gtIndex=gtIndex+1
            curFrame=curFrame+1
            dis=mm.distances.norm2squared_matrix(np.array(gtsPos), np.array(hyposPos))
            acc.update(gts,hypos,dis)

        else:
            edges=[]
            globalHungarian=[]
            for i in range(len(cameras)):
                for j in range(i+1,len(cameras)):
                    globalHungarian=[]
                    x=0
                    xindexes=[]
                    yindexes=[]
                    stackedimgages=[]
                    for pos in range(imgsSaved):
                        stackedimgages.append([])
                        for person in cameras[j].PersonData:
                            if(person.updated==True and len(person.imgs)==imgsSaved):
                                stackedimgages[pos].append(person.imgs[pos])
                    for fdata in range(len(cameras[i].PersonData)):
                        if(cameras[i].PersonData[fdata].updated==False or len(cameras[i].PersonData[fdata].imgs)!=imgsSaved):
                            continue
                        xindexes.append(fdata)
                        y=0
                        triplet=test(cameras[i].PersonData[fdata].imgs[0],stackedimgages[0])
                        for pos in range(1,imgsSaved):
                            triplet=np.add(triplet,test(cameras[i].PersonData[fdata].imgs[pos],stackedimgages[pos]))
                        globalHungarian.append([])
                        for pdata in range(len(cameras[j].PersonData)):
                            if(cameras[j].PersonData[pdata].updated==False or len(cameras[j].PersonData[pdata].imgs)!=imgsSaved):
                                continue
                            globalHungarian[x].append(np.sum(np.absolute(np.subtract(cameras[j].PersonData[pdata].histogram_h,cameras[i].PersonData[fdata].histogram_h)))*2+triplet[y])
                            if(x==0):
                                yindexes.append(pdata)
                            #globalHungarian[fdata].append(np.sum(np.absolute(np.subtract(cameras[j].PersonData[pdata].histogram_h,cameras[i].PersonData[fdata].histogram_h))))
                            #globalHungarian[fdata].append(triplet[pdata])
                            y=y+1
                        x=x+1
                    if(len(globalHungarian)!=0):
                        row_ind, col_ind = linear_sum_assignment(globalHungarian)
                        print(globalHungarian)
                        for pos in range(len(row_ind)):
                            if(globalHungarian[row_ind[pos]][col_ind[pos]]<3.2):
                                edges.append((cameras[i].PersonData[xindexes[row_ind[pos]]].globalPersonIndex,cameras[j].PersonData[yindexes[col_ind[pos]]].globalPersonIndex))
            
            Allcliques=cliques(edges,len(cameras),globalPersonCount).getCliques()

            for cam in cameras:
                for person in cam.PersonData:
                    isinclique=True
                    for clique in Allcliques:

                        if person.globalPersonIndex in clique:
                            isinclique=False
                            break
                    if isinclique:
                        Allcliques.append([person.globalPersonIndex])

            for sclique in Allcliques:
                indexes=[]
                cur=min(sclique)

                
                for i in range(len(sclique)):
                    isInclique=False
                    prvIndex=cameras[localgloballink[sclique[i]-1][1]].PersonData[localgloballink[sclique[i]-1][2]].prvglobalFoundOutPersonIndex
                    if prvIndex==-1:
                        isInclique=True
                    else:
                        for snclique in Allcliques:
                            if prvIndex in snclique:
                                isInclique=True
                                break;
                    if isInclique==True:
                        cameras[localgloballink[sclique[i]-1][1]].PersonData[localgloballink[sclique[i]-1][2]].globalFoundOutPersonIndex=cur
                    else:
                        cameras[localgloballink[sclique[i]-1][1]].PersonData[localgloballink[sclique[i]-1][2]].globalFoundOutPersonIndex=prvIndex

            for cam in range(len(cameras)):
                for person in cameras[cam].PersonData:
                    if person.updated==True:
                        cv2.putText(frame[cam],str(person.globalFoundOutPersonIndex) ,(int(person.positions[len(person.positions)-1][0]), int(person.positions[len(person.positions)-1][1])),0, 1e-3 * frame[index].shape[0], (0,255,0),2)
            
            for sclique in Allcliques:
                for i in range(len(sclique)):
                    cameras[localgloballink[sclique[i]-1][1]].PersonData[localgloballink[sclique[i]-1][2]].prvglobalFoundOutPersonIndex=cameras[localgloballink[sclique[i]-1][1]].PersonData[localgloballink[sclique[i]-1][2]].globalFoundOutPersonIndex
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
