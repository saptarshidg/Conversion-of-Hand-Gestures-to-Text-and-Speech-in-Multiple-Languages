import cv2
import numpy as np
import csv
from matplotlib import pyplot as plt

def captureImage():

	cam=cv2.VideoCapture(0)
	cv2.namedWindow("Preprocessing")
	img_counter=0

	while True:
		ret,frame=cam.read()
		cv2.imshow("test",frame)
		if not ret:
			break
		k=cv2.waitKey(1)

		if k%256==27:
			print("Escape hit.Closing..")
			break
		elif k%256==32:
			img_name="Image_{}.png".format(img_counter)
			print("Image file saved as ",img_name)
			cv2.imwrite(img_name,frame)
			img_counter+=1
	cam.release()
	cv2.destroyAllWindows()
	return img_name

def cropImage(imgName):
	im=cv2.imread(imgName)
	fromCenter=False
	r=cv2.selectROI(im,fromCenter)
	#r=cv2.selectROI(im)
	
	imgCrop=im[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
	cv2.imwrite("Cropped.png",imgCrop)
	#cv2.imshow("Image",imgCrop)
	#cv2.waitKey(0)
	return imgCrop

def rgbtoGray(blur):
	filtered=cv2.imread("Blurred.png",0)
	cv2.imwrite("Gray.png",filtered)
	#cv2.imshow("Grayscale",filtered)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return filtered

def binary():
	hand=cv2.imread("Gray.png")
	ret,the=cv2.threshold(hand,120,170,cv2.THRESH_BINARY_INV)
	cv2.imwrite("Binary.png",the)
	#cv2.imshow("Test",the)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return the

def cont(the):
	converted=cv2.cvtColor(the,cv2.COLOR_BGR2GRAY)
	_,contours,_=cv2.findContours(converted.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	hull=[cv2.convexHull(c) for c in contours]
	final=cv2.drawContours(the,hull,-1,(255,0,0))
	cv2.imwrite("Convex Hull.png",final)
	#cv2.imshow("Convex Hull.png",final)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

def blur(crop):
	blur=cv2.GaussianBlur(crop,(3,3),0)
	cv2.imwrite("Blurred.png",blur)
	#cv2.imshow("Blurred.png",blur)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return blur

def morphology(conv):
	kernel=np.ones((5,5),np.uint8)
	dilate=cv2.dilate(conv,kernel,iterations=1)
	erosion=cv2.erode(dilate,kernel,iterations=1)
	cv2.imwrite("Filter.png",erosion)
	cv2.imshow("Filter.png",erosion)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return erosion

def thinning(img):
	size=np.size(img)
	skel=np.zeros(img.shape,np.uint8)
	ret,img=cv2.threshold(img,127,255,0)
	element=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done=False
	
	while(not done):
		eroded=cv2.erode(img,element)
		temp=cv2.dilate(eroded,element)
		temp=cv2.subtract(img,temp)
		skel=cv2.bitwise_or(skel,temp)
		img=eroded.copy()

	cvs.imwrite("thinning.png",skel)
	cv2.imshow("skel",skel)
	cv2.waitKey(0)
	cv2.destroyWindows()

blurred=''

def main():
	name=captureImage()
	crop=cropImage(name)
	blurred=blur(crop)
	rgb=rgbtoGray(blurred)
	conv=binary()
	er=morphology(conv)
	edges = cv2.Canny(er,100,200)
	with open('test.txt', 'w') as f:                                    
   		writer = csv.writer(f, delimiter=',')
   		writer.writerows(edges)
	plt.subplot(121),plt.imshow(er,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

	plt.show()
	
main()