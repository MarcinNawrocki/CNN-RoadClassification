import google_streetview.api
import os.path
from PIL import Image


file_name = 'pp'
folder_name = 'pp'


loc=['52.4016388,16.9490621',
	 '54.3470944,18.6451204',
	 '54.3470085,18.6451353',
	 '54.3469229,18.6451517',]
n = len(loc)
for i in range(n):
	for j in range(4):
		if j == 0:
			head='33.25'
		elif j == 1:
			head='20.00'
		elif j == 2:
			head = '47.00'
		else:
			head='77.00'
		params = [{
			'size': '640x640', # max 640x640 pixels
			'location': loc[i],
			'heading': head,
			'pitch': '0.00',
			'key': '***********'
		}]

		results = google_streetview.api.results(params)
		results.download_links(folder_name)

		file_name_index = '_' + str(i) + '_' + str(j)
		name = file_name + file_name_index

		if os.path.isfile("/home/ostry/PycharmProjects/street/"+folder_name+"/gsv_0.jpg"):
			os.rename("/home/ostry/PycharmProjects/street/"+folder_name+"/gsv_0.jpg", "/home/ostry/PycharmProjects/street/"+folder_name+"/"+name+".jpg")

		os.remove("/home/ostry/PycharmProjects/street/"+folder_name+"/metadata.json")
		#im = Image.open("/home/ostry/PycharmProjects/street/"+folder_name+"/"+name+".jpg")

		#im = im.crop((100, 100, 560, 560))  # x1,y1,x2,y2_______640 px -> 460 px
		#im.save("/home/ostry/PycharmProjects/street/"+folder_name+"/"+name+".jpg", quality=95)

		procent = round(((i+1)*(j+1)/(n*(j+1)))*100)
		print(str(procent)+'%')

AIzaSyDIX0TDii1oCRpSM8ohU5Mu3prvwVA9OD8