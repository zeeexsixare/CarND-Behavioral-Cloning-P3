
# **Behavioral Cloning** 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
First, the images were cropped by 70 pixels from the top, and 25 pixels from the bottom using the Cropping2D function. 

My model consists of the [NVIDIA convolution neural network](https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwjfjb2OovXbAhWj6YMKHZgKAcUQjhx6BAgBEAM&url=https%3A%2F%2Fdevblogs.nvidia.com%2Fdeep-learning-self-driving-cars%2Fcnn-architecture%2F&psig=AOvVaw1zTKhbjS11AX7TqLTGnKc5&ust=1530237783965306) with 5 convolutional layers and 3 fully connected layers.  Kernels were 5x5 for the first 3 layers, then 3x3 for the last 2 layers. 

The data was normalized using a Lambda layer to center between -0.5 and 0.5.  ReLU was used as the activation function on all the convolutional layers. 
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers between 0.3 and 0.5   to reduce overfitting.  Anything less tended to crash the car.  However, more epochs were required.  Batch Normalization was used to converge more quickly. 

The model was trained and validated on different data sets to ensure that the model was not overfitting.  I used a 80/20 split. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track without touching the curb.

#### 3. Model parameter tuning

The adam optimizer has been standard fare for TensorFlow and Keras, so that's what I used.  No learning rates were adjusted manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used only center lane driving and both left and right camera views were incorporated with a 0.1 correction factor.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to pick up a few tips from the Slack channel and to look at some of the other recent error messages encountered so that I would know how to respond.

My first step was to use the aforementioned NVIDIA model, as others have had significant success using this model.  

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes 
1. 24 filters with 5x5 kernel
2. 36 filters with 5x5 kernel
3. 48 filters with 5x5 kernel
4. 64 filters with 3x3 kernel
5. 64 filters with 3x3 kernel
6. Fully connected layer (100 dim)
7. Fully connected layer (50 dim)
8. Fully connected layer (10 dim)
9. Fully connected layer (1 dim) which is the steering angle

#### 3. Creation of the Training Set & Training Process

I did not use a generator because I trained on an AWS instance and did not need it because I never got any memory error issues at any point.

I first converted the color space from BGR to RGB because I knew that opencv opens it in BGR.

I first started using the provided Udacity data but then created my own data set when others had success creating their own.  I created my own small data set first with driving around the track at full speed manually to validate that the model was working properly.  This created about 2000 data points.  After initial missteps due to bugs in the code, I was able to stay on track until the tight curves.  I then re-drove a new data set for 1 lap at slow speed, totalling 6000 data points, which I thought might be enough.

I next added the flipped orientation when I noticed that the car was weaving excessively.  I also reversed the steering angle accordingly so as to generalize better.  This stabilized it on the straights.

I knew that I was having problems with overfit after only 3 epochs.  I added dropout at 10% on every layer to slowly introduce it.  I increased it incrementally until I reached a sweet spot of 0.3 to 0.5.  However, it kept on running off the road at the dirt road.  

I realized I needed to add corrective data points at the dirt road.  After adding and patching and rerunning the model many times, it kept on running off the road.

I re-reviewed the successful methods that others had used and realized that I only had about 6000 data points (12,000 when flipped was added).  Recovery data points totaled 2000ish data points, so I thought 8000 might be enough.  However, it was not.  Others reported success only after having 40k or more data points.  I then decided to add left and right camera images rather than re-drive or add more recovery data.  This worked well along with increasing dropout and increasing epochs to 10.  My validation loss tended to continue to drop after the 3rd epoch when previously it would plateau, and the model still wasn't overfitting.  This last step generated a robust model that successfully drove around the track without leaving the road!