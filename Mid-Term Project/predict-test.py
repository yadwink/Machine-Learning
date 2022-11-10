import requests

url = 'http://127.0.0.1:5000/predict/'

patient_id = 'patient-abc123'

patient = {
 "n_days": 4427,
 "status": "C",
 "drug": "Placebo",
 "age": 17947,
 "sex": "M",
 "ascites": "Y",
 "hepatomegaly": "Y",
 "spiders": "Y",
 "edema": "Y",
 "bilirubin": 1.9,
 "cholesterol": 500.0,
 "albumin'""copper": 281.0,
 "alk_phos": 10396.8,
 "sgot": 188.34,
 "tryglicerides": 178.0,
 "platelets": 500.0,
 "prothrombin": 11.0
}


response = requests.post(url, json=patient).json() ## post the customer information in json format
print(response)

if response['LiverCirrhosis'] == True:
    print('patient is diagnosed with liver cirrhosis %s' % patient_id)
else:
    print('patient is not diagnosed with liver cirrhosis %s' % patient_id)    
