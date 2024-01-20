### Introduction: 
 
Logo detection from images is an exciting field that involves the identification and recognition of logos in images. This technology has become increasingly important today, as companies use logos for brand recognition and marketing. The ability to automatically detect and recognize logos in images can be helpful in various applications, such as product search, image retrieval, and social media monitoring. 
The project of logo detection from images aims to develop an algorithm that can accurately and efficiently detect logos in a given image. This involves training a deep-learning model using a large dataset of images with different logos. The model is trained to recognize the unique features of each logo and distinguish them from other elements in the image. We have used the TensorFlow object detection model and YOLO v5 model in our project. 

### Dataset : 
After researching online, we found the dataset online from GitHub and Flickr dataset for our project. In total out of  21924, we have used 763 images. Out of 352 classes have used 4 classes for Tensorflow.   
We have used the images of companies like Adidas, Coca-cola, Samsung, McDonalds for TensorFlow. 
We used Flickr dataset containing around 712 images and around 27 classes for YOLO v5 model. 
We have used Adidas, Apple, BMW, Citroen, Cocacola, DHL, Fedex, Ferrari, Ford, Google, Heineken, HP, Intel, McDonalds, Mini, Nbc, Nike, Pepsi, Porsche, Puma, RedBull, Sprite, Starbucks, Texaco, Unicef, Vodafone, Yahoo. 
 
### Model Building: 

### Tensorflow: 

TensorFlow is a widely used platform for building logo detection models. Here are the general steps that we have used in a logo detection project: 
1.	Initially, we downloaded the dataset and did the annotations. We have generated the xml file with the class name and with the size of the logos that appear in an image. 
2.	After that, we generated the train.tfrecord file of all the classes by using a Python script. 
3.	Next, we generated the pbtext file which contains the id and class of all the images. 
4.	pbtxt is a text-based representation of a TensorFlow graph. It is used to describe the structure of a graph in a human-readable format and is often used in conjunction with the .pb binary format to store and load TensorFlow models. 
5.	The .pbtxt file contains a set of nodes, which represent operations in the graph, and edges, which represent the data flowing between the nodes. Each node has a name, a type, and a set of attributes that describe its behavior. 
6.	After doing all these steps we setup a folder structure for our project. 
7.	Then, we started to implement our model. 
8.	Initially, we cloned the TensorFlow models repository in our notebook. 
9.	After that, we have installed the Tensorflow object detection API. 
10.	Then, we mentioned the train_record_path and label_map path. 
11.	In an object detection project using TensorFlow, the training record path and label map path are necessary for training an accurate and effective model. 
12.	The training record path refers to the location of a dataset of labeled images in the format of TFRecord files, which contain serialized TensorFlow Example protocol buffer messages. The images are labeled with bounding boxes around the objects of interest, and each bounding box is associated with a class label. The training record path is used to load the training data into the model during the training process. 
13.	The label map path refers to a file that contains a mapping of class names to class IDs. Each object in the training data is assigned a class label, which is typically represented by a string name. 
14.	After that, we have used TensorFlow 2 version of the object detection training script model_main_tf2.py from the TensorFlow Object Detection API. 
15.	Then we used the below hyperparameter in the model:
 ```
 *batch_size = 16 refers to the number of images that will be processed in each training batch. The batch size is a crucial hyperparameter that affects the speed and accuracy of the training process.*  
 *num_steps = 8000 refers to the number of steps that the training process will run for. Each training step processes one batch of images, and the number of steps determines how many images will be processed during the training process.*  
 *num_eval_steps = 1000 refers to the number of steps that will be used to evaluate the model’s performance during the validation process. The validation process checks the model’s accuracy on a separate set of images that were not used during the training process.* 
 ```
16.	Then, at last, we trained our model using the test dataset but unfortunately, we were not able to get the accurate results; we got the result in an array format but not getting the accurate bounding box of the logo in the images. 

### Final Output of Tensorflow: 
We have got the below output in an array format for the images. 

![image](https://github.com/harshilbhavsar7/Logo-Detection-with-Flask-App/assets/60917314/6047633a-edb3-4aee-9f4e-2b7a52ef7db7)

### YOLO v5:
After implementing the project in TensorFlow, we tried to use a custom YOLO v5 model to train the dataset. 
Steps for YOLO v5 model implementation: 
* Initially, we have collected the dataset from image. ntua.gr website. 
* Next step was to do the annotations. So we have used LabelImg to do the annotations.  
*	We have segregated all the images to the team members to complete the annotations.  
*	Then we collected all the annotations in yolo format and saved all the files in .txt format with the coordinates. 
*	Next, we created the yaml filet with the below path and classes to use this file for training the model. 
  ```
 	train: /content/final_data_v1/images
 	val: /content/final_data_v1/images
 	test:  /content/final_data_v1/images  
  ```

# Classes
```
nc: 27  # number of classes names: 
['Adidas','Apple','BMW','Citroen','Cocacola','DHL','Fedex','Ferrari','Ford','Google','Heinek en','HP','Intel','McDonalds','Mini','Nbc','Nike','Pepsi','Porsche','Puma','RedBull','Sprite','S tarbucks','Texaco','Unicef','Vodafone','Yahoo']   
```

*	After that, we cloned the yolo model in our python notebook and tried to implement the YOLO v5 model. 
*	So to train the model, we wanted to choose the YOLO model. YOLO comes in below 5 size. 

![image](https://github.com/harshilbhavsar7/Logo-Detection-with-Flask-App/assets/60917314/f53f5722-da5e-4b89-92ed-97160934bc47)

As we can see in the above images. In the above diagram, there are a total of 5 sizes available in YOLO v5. We have used the YOLOv5s model for our project as we had a small dataset.  
*	n for extra small (nano) size model. 
*	s for small-size model. 
*	m for medium size model. 
*	l for large size model 
*	x for extra large size model 

As we had a very small dataset, we used a Small model for our project. 
*	After choosing our model, we used hyperparameters like SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias to train our model. 
*	We have used 50 epochs and an image size of 640 to train the custom YOLO v5 model.  
*	Finally, we have received the below accuracy in our model.

![image](https://github.com/harshilbhavsar7/Logo-Detection-with-Flask-App/assets/60917314/75de0d0e-b2b6-4cb3-995b-642a98931ca1)


*	We got mAP50 accuracy 92% and mAP50-95 accuracy 64%. 
*	mAP50 refers to the mean average precision at 50% overlap between predicted and ground truth bounding boxes in object detection tasks. A higher mAP50 score indicates better object detection performance, with a maximum value of 1 indicating perfect detection. The mAP50 metric is commonly used in benchmarking object detection models, and it is one of the key metrics used in the popular COCO dataset evaluation. 
*	mAP50-95 is a variation of the mean average precision metric used in object detection tasks that takes into account a range of overlap thresholds between predicted and ground truth bounding boxes. 
*	In traditional mAP50, the overlap threshold is fixed at 50%. In contrast, mAP50-95 calculates the average precision (AP) for overlap thresholds ranging from 50% to 95%, in 5% increments. The final mAP50-95 score is the average of the AP values across all overlap thresholds. 
*	Please take a look at the model summary below.

![image](https://github.com/harshilbhavsar7/Logo-Detection-with-Flask-App/assets/60917314/62ed4cd0-f0e6-415c-b2d6-08764facc9b4)

Graphs : 
  
  1. F1 – confidence curve

  ![image](https://github.com/harshilbhavsar7/Logo-Detection-with-Flask-App/assets/60917314/0bbc6919-6abc-4272-adf6-ed5264908723)

  2. Precision – Recall Curve  

  ![image](https://github.com/harshilbhavsar7/Logo-Detection-with-Flask-App/assets/60917314/58efc868-2fce-46b1-a18b-46c7000083b9)

  3. Precision confidence curve 

  ![image](https://github.com/harshilbhavsar7/Logo-Detection-with-Flask-App/assets/60917314/feca0f4e-743e-4361-b2cf-9dd78b013671)

  4. Confusion Matrix 

  ![image](https://github.com/harshilbhavsar7/Logo-Detection-with-Flask-App/assets/60917314/632b1802-9130-4706-8e2d-715f65b8db1c)

### Final output of YOLO v5 model : 

Please find below the some of the output snippets of our YOLO v5 model result. 

![image](https://github.com/harshilbhavsar7/Logo-Detection-with-Flask-App/assets/60917314/ada6cd29-f5bc-4a67-bb0f-21f95eebb754)

![image](https://github.com/harshilbhavsar7/Logo-Detection-with-Flask-App/assets/60917314/4dbcdf51-cb8e-44b7-8de6-c98ddfeaa217)

### Integrating Trained model with Flask:

The trained model can be consumed through the Flask web app to display its working. There are multiple scripts that were developed with the sole purpose of creating a sufficient web app architecture.

### Conclusion : 

To conclude, we have implemented the 2 models in our project one using TensorFlow and one custom YOLO v5 model. In TensorFlow we have not received the accurate result and it had many garbage classes, so we were not able to get the result in a image file. So we have used YOLO v5 custom model and we have got good accuracy in that model. 

### REFERENCES : 

[1]	Jocher, G. (2020, August 21). ultralytics/yolov5. GitHub. https://github.com/ultralytics/yolov5 
[2]	tensorflow. 	(2019, 	July 	15). 	tensorflow/models. 	GitHub. 
https://github.com/tensorflow/models/tree/master/research/object_detection  
[3]	Hui, J. (2019, April 3). mAP (mean Average Precision) for Object Detection. Medium. 
https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection45c121a31173  
[4]	262588213843476. (n.d.). Python script to create tfrecords from pascal VOC data set format (one class detection) for Object Detection API Tensorflow, where it divides dataset into (90% train.record and 10% test.record). Gist. Retrieved April 17, 2023, from https://gist.github.com/saghiralfasly/ee642af0616461145a9a82d7317fb1d6  
[5]	Y. Kalantidis Y. Avrithis, LG. P. T. van Z. (2011). Scalable Triangulation-based Logo Recognition. 
Image.ntua.gr. http://image.ntua.gr/iva/datasets/flickr_logos/  
