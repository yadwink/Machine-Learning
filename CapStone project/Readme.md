# Fruits and vegetables classification

## Goal of the project

>  To solve an image classification problem in deep learning for fruits and vegetables dataset

## Dataset description
> The dataset can be found here: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition

> It contains images following image items:

    * Fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.
    * Vegetables: cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.

> These images are contained in three folders:

    * train (100 images each)
    * test (10 images each)
    * validation (10 images each)

 # Problem description

The manual classification of fruits and vegetables is often difficult. The reason is due to their wide variety and similarity in appearance i.e., colour and shape. Therefore, the manual classification can be challenging. However,an automated classifier can come in handy in future. Therefore, I trained different deep learning image classification models to identify or recognize the food items from the photos correctly.  

# Libraries required

    * Pandas
    * Numpy
    * Python
    * Tensorflow
    * Keras

    for further list of libraries, please refer to the requirement.txt file

# Setup

    * Clone the project repo and open it
    * To reproduce the results, first download the data, run train.py, create a virtual environment as described and install the dependencies (requirement.txt )

# EDA

> This analysis is under the *Notebook.ipynb* file
> I first created a list with the filepaths for training and testing datasets
> Then, I created a DataFrame with the filepath and the labels of the pictures
> In the next step, I extracted the basic information about the training dataset that showed that on average there are 86.53 images for each class in the training set and there are total of 2780 pictures with 36 different labels
> In the last step, I created a data frame with one label of each category

# Training different models

> The first model I trained was Mobilenet which showed loss of 0.1818  and accuracy of  0.9491
> Then I trained a Sequential model
> My third model was ResNet101 which showed loss of 1.0867 and accuracy of 0.8774. I further tested the accuracy of the ResNet101 on the test set which showed an accuracy score of 87.7%!!
> My last model was Xception which I trained and performed hypertuning on its parameters. This showed loss of 0.1901 and accuracy of 0.9551
> Then I performed the final model training based on hyperparmeters on training and validation datasets
> Then I evaluated the performance of all three models. The evaluation showed that the Mobilenet had the highest accuracy and therefore, I decided to go for Mobilenet as my final model and  use it for deployment


# Saving and loading the saved model

The model was saved as  fruits_vegetables_MobileNet.h5 and also using the tflite library as fruits_vegetables_MobileNet.tflite

# Putting the model into a Web Service and local deployment using Docker:

I extracted the final trained model as the python script *predict.py*. 

# Creating a Python virtual environment using Pipenv 

> I created a virtual environment using pipenv which created Pipfile and Pipfile.lock. These files contain library-related details, version names and hash files. 

> These files can easily be used when the project runs on another machine. These dependencies  can be easily installed using command pipenv install. This command will look upto  Pipfile and Pipfile.lock and they will install the libraries with specified version.


# Containerization in Docker 

> I created the *dockerfile* that allows to separate this or any project  from any system machine and enables it running smoothly on any machine using a container. I then created the docker image which contained following dependencies of the project

```
FROM python:3.11.1-slim

WORKDIR /app
COPY . /app/

RUN pip install Pillow
RUN pip install keras-image-helper
RUN pip install tflite

EXPOSE 5000

ENTRYPOINT [ "waitress-serve", "--listen=0.0.0.0:5000", "predict:app" ]

```
> Then I finally deployed my liver cirrhosis app inside a Docker container.

# Local deployment of the project in the following steps:

1. python *train.py* in the cmd

2. pip install flask

3. python *predict.py* 

4. pip install waitress

5. waitress-serve --listen=127.0.0.1:5000 predict:app

6. python predict-test.py in a new terminal

7. pip install pipenv

8. pipenv install numpy pandas matplotlib tensorflow scikit-learn flask requests tflite waitress

9. pipenv install --python 3.11

# Deploy the model 

* Install Docker on your system
* Install pipenv on your system
* pipenv install requirements 
* docker built -t capstone .
* docker run -it --rm -p 5000:5000 capstone
