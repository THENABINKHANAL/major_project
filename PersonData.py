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