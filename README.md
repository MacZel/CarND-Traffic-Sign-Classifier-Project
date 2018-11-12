# **Traffic Sign Recognition** 

## Writeup

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[chart1]: ./readme_imgs/chart1.png "Labels occurence frequency distributions"
[chart2]: ./readme_imgs/chart2.png "Labels occurence frequency distributions combined descending"
[mean_std]: ./readme_imgs/mean_std.png "Mean & Standard Deviation of Training Set"
[bokeh]: ./readme_imgs/bokeh.PNG "Bokeh Interactive plot"
[valid_acc]: ./readme_imgs/valid_acc.png "Validation Accuracy"
[train_acc]: ./readme_imgs/train_acc.png "Training Accuracy"

[raw]: ./readme_imgs/raw.png "Raw training images"
[norm]: ./readme_imgs/norm.png "Normalized training images"
[gray]: ./readme_imgs/gray.png "Grayscaled training images"

[my_raw]: ./readme_imgs/my_raw.png "Raw found images"
[my_norm]: ./readme_imgs/my_norm.png "Normalized found images"
[my_gray]: ./readme_imgs/my_gray.png "Grayscaled found images"

[top_5_1]: ./readme_imgs/top_5_1.png "Mean & Standard Deviation of Training Set"
[top_5_2]: ./readme_imgs/top_5_2.png "Mean & Standard Deviation of Training Set"
[top_5_3]: ./readme_imgs/top_5_3.png "Mean & Standard Deviation of Training Set"
[top_5_4]: ./readme_imgs/top_5_4.png "Mean & Standard Deviation of Training Set"
[top_5_5]: ./readme_imgs/top_5_5.png "Mean & Standard Deviation of Training Set"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/MacZel/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
    34799
* The size of the validation set is ?
    4410
* The size of test set is ?
    12630
* The shape of a traffic sign image is ?
    (32, 32, 3)
* The number of unique classes/labels in the data set is ?
    43

Below is the random sample taken from training set, showing raw images.
![Raw training images][raw]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the labels occurence frequency distributions in all three datasets.

![Labels occurence frequency distributions][chart1]

Below is the labels occurency frequency distibutions histogram of all three datasets combined.

![Labels occurence frequency distributions combined descending][chart2]

As we can see from the first graph distributions are pretty much the same over all three sets.
The most popular signs across all three sets are:
* Speed limit (50km/h)
* Speed limit (30km/h)
* Yield
* Priority road
* Keep right

An interactive plot in Bokeh is available when running the jupyter notebook on local environment.
It enables to browse and switch through all datasets, and sort datasets in any order by label id or label frequency. There's also a hover tool enabling to lookup the random sample image from specified dataset.
Below is the image showing the tool:
![Bokeh Interactive plot][bokeh]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the image data because it is an important step ensuring each pixel has a similar data distribution. This step makes covergence faster while training the network. Data normalization is done by subtracting the mean from each pixel, and then dividing by the standard deviation.
For image inputs, pixel values should be positive, so data should be normalized to range [0,1] or [0,255]. I used [0,1] range.

Normalization step on training set is depicted below:
![Normalized training images][norm]

I did obtain a very close to zero mean over the whole training set (MEAN:  0.00766), and the approximately equal standard variation.
Below images depict mean image and standard deviation over training set.

![Mean & Standard Deviation of Training Set][mean_std]

As a last step, I grayscaled the image data.
Dimensional reduction is performed by collapsing the RGB channels into a single gray-scale channel.
Reducing other dimensions allows neural network performance to be invariant of those other dimensions, and make the training more controllable.

Below is shown grayscale applied to normalized sample from previous step:
![Grayscaled training images][gray]

Firstly I did also generate additional data by random translating, rotating, warping and applying gaussian noise.
This increased the training dataset by a factor of 4.
However I did not observe a significant increase in validation accuracy. Furthermore additional data did significantly slow the computation time.
Therfore I decided not to generate additional data and keep with the most simple solution.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x16 					|
| Flattening			| outputs 400									|
| Dropout				| 												|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Dropout				| 												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Dropout				| 												|
| Fully connected		| outputs 43  									|

This is a classic LeNet model with additional three dropout layers.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer as used in LeNet model.
Batch size: 128
Epochs: 80
Learning rate: rate = 0.000975
Sigma: 0.1
Mu: 0
Keep prob: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Below is the plot showing models learning process over all epochs, gained accuracy and the derivative of training and validation accuracy functions, showing tendency of a model to learn over epochs.

![Training Accuracy][train_acc]
![Validation Accuracy][valid_acc]

My final model results were:
* training set accuracy of 98.86
* validation set accuracy of  95.58
* test set accuracy of 93.26

* What architecture was chosen?
The first architecture that was tried was LeNet model. It was chosen because of its simplicity and good accuracy values obtained.
I did also play with model based on article:
[Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
but didn't manage to gain more on accuracy, so I abandoned this model, and came back to LeNet.
* What were some problems with the initial architecture?
First LeNet architecture was modified with the droput becase of large accuracy difference between training and validation sets. Probably due to overfitting. To make the model less prone to fall into redundancy the droputs were introduced between last three fully conected layers.
* Which parameters were tuned? How were they adjusted and why?
Epochs were drastically increased up to 80 epochs. As we can see on validation accuracy plot, the accuracy is dependent on number of epochs. After reaching 80 epochs I decided to stop training, since I did not observe learning of the model anymore.
Also I was playing with learning rate, and finally I did decrease it from 0.001 to 0.000975
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Raw found images][my_raw]

The second image might be difficult to classify because of the bad lightning conditions.
Last three images might also be difficult to classify because of skewness.

Below are shown the preprocessing steps (normalization & dimensional reduction aka grayscaling)
![Normalized found images][my_norm]
![Grayscaled found images][my_gray]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)  | Speed limit (60km/h)   						| 
| No entry     			| No entry 										|
| Stop					| Stop											|
| Priority Road	      	| Priority Road					 				|
| Go straight or left	| Go straight or left							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.26%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first, third and fourth images, the model is sure that this is a stop sign (probability of almost 100%). For the second and fifth image the prediction is correct but we can see that the the model is not so sure of the correctness of this classification.

| Probability [%]     	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100         			| Speed limit (60km/h)   						| 
| 36     				| No entry 										|
| 100					| Stop											|
| 99	      			| Priority Road					 				|
| 57				    | Go straight or left							|

Below are the visualizations of top 5 softmax probabilities for found images:
* [Mean & Standard Deviation of Training Set][top_5_1]
* [Mean & Standard Deviation of Training Set][top_5_2]
* [Mean & Standard Deviation of Training Set][top_5_3]
* [Mean & Standard Deviation of Training Set][top_5_4]
* [Mean & Standard Deviation of Training Set][top_5_5]

### Articles I made use of:
* ![Image Data Pre-Processing for Neural Networks](https://becominghuman.ai/image-data-pre-processing-for-neural-networks-498289068258)
* ![NanoNets : How to use Deep Learning when you have Limited Data](https://medium.com/nanonets/nanonets-how-to-use-deep-learning-when-you-have-limited-data-f68c0b512cab)
* ![Data Augmentation | How to use Deep Learning when you have Limited Data — Part 2](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced)
* ![Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
* ![A Gentle Introduction to Normality Tests in Python](https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)
* ![Data Augmentation Techniques in CNN using Tensorflow](https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9)
* ![Tensorflow Image: Augmentation on GPU](https://towardsdatascience.com/tensorflow-image-augmentation-on-gpu-bf0eaac4c967)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

TBD
