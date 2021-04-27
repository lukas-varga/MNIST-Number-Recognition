import os
import io
import boto3
import json
import csv
import numpy as np

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')
            

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    data = event['body']
    payload = {"instances" : [data]}
    print(payload)
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/json', Body=json.dumps(payload))
    print(response)
    
    output = json.loads(response['Body'].read().decode('utf-8'))
    print(output)
    
    label = np.argmax(output['predictions'])
    final_output = 'The predicted number is {}'.format(label)
    print(final_output)
    
    return final_output