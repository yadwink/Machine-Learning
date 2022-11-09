# Liver Cirrhosis Prediction  

 Cirrhosis is a late stage of scarring (fibrosis) of the liver caused by many forms of liver diseases and conditions, such as hepatitis and chronic alcoholism. It is a widespread issue especially in North America due to high intake of alcohol. 


# Data description

> This dataset was collected from the Mayo Clinic trial in primary biliary cirrhosis (PBC) of the liver conducted between 1974 and 1984. A total of 424 PBC patients went through  the randomized placebo-controlled trial of the drug D-penicillamine. The first 312 cases in the dataset participated in the randomized trial and contain largely complete data. The additional 112 cases did not participate in the clinical trial but consented to have basic measurements recorded and to be followed for survival. 

The dataset can be found here: https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset

## The data includes a bunch of diagnostics as follows:

1. ID: Patient ID

2. N_Days: number of days between registration and the earlier of death, transplantation, or study analysis time in July 1986

3. Status: status of the patient C (censored), CL (censored due to liver tx), or D (death)

4. Drug: type of drug D-penicillamine or placebo

5. Age: age in [days]

6. Sex: M (male) or F (female)

7. Ascites: presence of ascites N (No) or Y (Yes)
    Ascites is signiﬁcant scarring of the liver. It is a common complication of liver cirrhosis

8. Hepatomegaly: presence of hepatomegaly N (No) or Y (Yes)
    Hepatomegaly is enlarged liver. An enlarged liver could be an acute response to infection or the result of advanced chronic liver disease. In this case i.e., Liver Cirrhosis

9. Spiders: presence of spiders N (No) or Y (Yes)

10. Edema: presence of edema N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy)

11. Bilirubin: serum bilirubin in [mg/dl]

12. Cholesterol: serum cholesterol in [mg/dl]

13. Albumin: albumin in [gm/dl]

14. Copper: urine copper in [ug/day]

15. Alk_Phos: alkaline phosphatase in [U/liter]

16. SGOT: SGOT in [U/ml]

17. Triglycerides: triglicerides in [mg/dl]

18. Platelets: platelets per cubic [ml/1000]

19. Prothrombin: prothrombin time in seconds [s]

20. Stage: histologic stage of disease (1, 2, 3, or 4)

 # Problem description

 The machine learning model can be trained to predict the diagnosis of the patient whether or not they are diagnosed with Liver Cirrhosis based on the given other clinical test results.  

# EDA

> This analysis is under the *MidTermProject_YKaur.ipynb* file
> I first separated the categorical and numerical variables and checked the data for missing values. Then I imputed missing values in the numerical variables by mean and missing values in categorical variables by the mode.
> Then I converted the target variable 'stage' into a binary variable where 1 indicates liver cirrhosis and 0 shows no cirrhosis.
> Explored several variables and their relationship wit the target variable.

# Setting up the validation framework

I performed the train/test/validation split and deleted the target variable from the data frame.

# Feature importance analysis 

Feature importance analysis using mutual info score. The mutual info score showed that hepatomegaly, status, ascites, & Edema are some important features. Ascites is the main complication of cirrhosis. Thus, is relevant to to the stage of the liver cirrhosis. 
 
# Training different models

> Logistic regression model training produced AUC score of 0.714
> The decision tree classifier produced AUC score of 0.642
> After that I tuned the decision tree classifier and decided on max_depth = 4, min_samples_leaf = 15 and trained the decision tree classifier again which produced AUC score of 0.742
> At last, I trained a random forest classifier and tuned its parameters  and decided with n_estimators=50 max_depth=15, min_samples_leaf=50. The trained model gave final AUC score of 0.820 which is the highest and the best AUC score
> I finally selected the random forest classifier model and trained this model on the train and validation set which gave AUC score of 0.768

# Saving and loading the saved model

The model was saved to a  binary file using the Pickle library as model_rf.bin

# Putting the model into a Web Service and local deployment using Docker:

I extracted the final trained model as the python script *predict.py*. 

# Creating a Python virtual environment using Pipenv 

> I created a virtual environment using pipenv which created Pipfile and Pipfile.lock. These files contain library-related details, version names and hash files. 

> These files can easily be used when the project runs on another machine. These dependencies  can be easily installed using command pipenv install. This command will look upto  Pipfile and Pipfile.lock and they will install the libraries with specified version.

# Containerization in Docker 

> I created the *dockerfile* that allows to separate this or any project  from any system machine and enables it running smoothly on any machine using a container. I then created the docker image which contained following dependencies of the project

```
FROM python:3.9-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_rf.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]

```

> Then I finally deployed my liver cirrhosis app inside a Docker container.

# Local deployment of the project

1. python *train.py* in the cmd

2. pip install flask

3. python *predict.py* 

4. pip install waitress

5. waitress-serve --listen=127.0.0.1:9696 predict:app

6. python predict-test.py in a new terminal

7. pip install pipenv

8. pipenv install numpy scikit-learn==1.0.2 flask pandas requests waitress

9. pipenv install --python 3.9

 
# Deploy the model 

* Install Docker on your system
* Install pipenv on your system
* pipenv install requirements 
* docker built -t diabetes-prediction .
* docker run -it -p 9696:9696 LiverCirrhosis:latest


# Deploy the model on a web service (AWS)

* pipenv install awsebcli --dev
* pipenv shell
* eb init -p docker livercirrhosis-serving
* eb local run --port 9696
* python predict.py
* eb create livercirrhosis-serving-env