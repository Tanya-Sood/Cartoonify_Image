import cv2
import numpy as np 
def reduce_colors(img,k):
  data=np.float32(img).reshape((-1,3))
  criteria=(cv2.TermCriteria_EPS +cv2.TermCriteria_MAX_ITER,20,0.001)
  ret,label,center= cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
  center=np.uint8(center)
  result=center[label.flatten()]
  result=result.reshape(img.shape)
  return result
def get_edges(img,line_size,blur_val):
  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  gray_blur=cv2.medianBlur(gray,blur_val)
  edges=cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_size,blur_val)
  return edges
img = cv2.imread("C:\\Users\\I Love My INDIA\\OneDrive\\Desktop\\Cartoonify image\\image.JPG")
#cv2.imshow('Image',img)
line_size=5
blur_val=3
edges=get_edges(img,line_size,blur_val)
k=12
img=reduce_colors(img,k)
blurred=cv2.bilateralFilter(img,d=5,sigmaColor=150,sigmaSpace=150)
cartoon=cv2.bitwise_and(img,img,mask=edges)
#cv2.imshow("Cartoon", cartoon)
cv2.imwrite("C:\\Users\\I Love My INDIA\\OneDrive\\Desktop\\Cartoonify image\\cartoon_image.JPG", cartoon)