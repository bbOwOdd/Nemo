import requests
import json

url = 'http://0.0.0.0:8000/v1/completions'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

# Example from the PubMedQA test set
prompt="BACKGROUND: Sublingual varices have earlier been related to ageing, smoking and cardiovascular disease. The aim of this study was to investigate whether sublingual varices are related to presence of hypertension.\nMETHODS: In an observational clinical study among 431 dental patients tongue status and blood pressure were documented. Digital photographs of the lateral borders of the tongue for grading of sublingual varices were taken, and blood pressure was measured. Those patients without previous diagnosis of hypertension and with a noted blood pressure \u2265 140 mmHg and/or \u2265 90 mmHg at the dental clinic performed complementary home blood pressure during one week. Those with an average home blood pressure \u2265 135 mmHg and/or \u2265 85 mmHg were referred to the primary health care centre, where three office blood pressure measurements were taken with one week intervals. Two independent blinded observers studied the photographs of the tongues. Each photograph was graded as none/few (grade 0) or medium/severe (grade 1) presence of sublingual varices. Pearson's Chi-square test, Student's t-test, and multiple regression analysis were applied. Power calculation stipulated a study population of 323 patients.\nRESULTS: An association between sublingual varices and hypertension was found (OR = 2.25, p<0.002). Mean systolic blood pressure was 123 and 132 mmHg in patients with grade 0 and grade 1 sublingual varices, respectively (p<0.0001, CI 95 %). Mean diastolic blood pressure was 80 and 83 mmHg in patients with grade 0 and grade 1 sublingual varices, respectively (p<0.005, CI 95 %). Sublingual varices indicate hypertension with a positive predictive value of 0.5 and a negative predictive value of 0.80.\nQUESTION: Is there a connection between sublingual varices and hypertension?\n ### ANSWER (yes|no|maybe): "

data = {
    "model": "llama3-8b-pubmed-qa",
    "prompt": prompt,
    "max_tokens": 128
}

response = requests.post(url, headers=headers, json=data)
response_data = response.json()

print(json.dumps(response_data, indent=4))