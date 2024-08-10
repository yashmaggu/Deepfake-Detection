import glob
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import gc
from torch import nn
from torchvision import models
#torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #USE CUDA if gpu is represent
isGpuAvailable= torch.cuda.is_available()

#Model
class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))


#Prediction Function
im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    #cv2.imwrite('./2.png',image*255)
    return image

def predict(model,img,path = './'):
  #print(img)
  fmap,logits = model(img.to('cuda' if isGpuAvailable else "cpu"))
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  confidence = logits[:,int(prediction.item())].item()*100
 
  idx = np.argmax(logits.detach().cpu().numpy())
  bz, nc, h, w = fmap.shape
  out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  predict = out.reshape(h,w)
  predict = predict - np.min(predict)
  predict_img = predict / np.max(predict)
  predict_img = np.uint8(255*predict_img)
  out = cv2.resize(predict_img, (im_size,im_size))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  img = im_convert(img[:,-1,:,:,:])
  result = heatmap * 0.5 + img*0.8*255
  #cv2.imwrite('/content/1.png',result)
  result1 = heatmap * 0.5/255 + img*0.8
  r,g,b = cv2.split(result1)
  result1 = cv2.merge((r,g,b))
  #plt.imshow(result1)
  #plt.show()
  print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)
  gc.collect()
  if(isGpuAvailable):
    torch.cuda.empty_cache()
  return [int(prediction.item()),confidence]

class extract_face_video(Dataset): # extracts face from given video for prediction
    def __init__(self,video_names,sequence_length = 10,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        #a = int(100/self.count)
        #first_frame = np.random.randint(0,a)
        face_detector = MTCNN(device=device) #MTCNN for face detection
        for i,frame in enumerate(self.frame_extract(video_path)):
         
            try:
                face = face_detector.detect(frame)[0][0]
            except:
                continue
            try:              
                left,bottom,right,top=face
                left=int(left)
                bottom=int(bottom)
                right=int(right)
                top=int(top)
                frame=frame[bottom:top,left:right] 
                #print(frame)
            except:
                pass
           
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
              break
        #print("no of frames",len(frames))
        frames = torch.stack(frames)
        frames = frames[:self.count]
    
        return frames.unsqueeze(0)
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path)
      success = 1

      while success:
          success, image = vidObj.read()
          if success:
              yield image


class PredictModel():
    def __init__(self):
        #Code for making prediction
        self.im_size = 112
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

        self.train_transforms = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.Resize((im_size,im_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean,std)])
        self.model =Model(2).cuda() if isGpuAvailable else Model(2).cpu()
        self.path_to_model = '../trained-model.pt'
        self.model.load_state_dict(torch.load(self.path_to_model,map_location=device))
        self.model.eval()

    def doPrediction(self):
        path_to_videos= glob.glob("predict_videos/*.mp4")
        #print(path_to_videos)
        video_dataset = extract_face_video(path_to_videos,sequence_length = 10,transform = self.train_transforms)
        #print(video_dataset)
        
        prediction = predict(self.model,video_dataset[0],'./')

        if prediction[0] == 1:      
            print("REAL")
        else:
            print("FAKE")
        if isGpuAvailable:
            torch.cuda.empty_cache()
        gc.collect()
        return prediction[0],prediction[1] #y-hat and confidence




# predictModel=PredictModel()
# y_hat,confidence=predictModel.doPrediction()
# torch.cuda.empty_cache()
# gc.collect()
