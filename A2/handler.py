# Label  
try:
    import unzip_requirements
except ImportError:
    pass

# Include required libraries
import boto3
import os
import json
import logging
import io
import base64
from requests_toolbelt.multipart import decoder

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from utils import class_labels
from predictor import predict

project_name = 'assignment2'

# Initialize logging
FORMAT = '%(asctime)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'user': project_name}
logger = logging.getLogger(project_name)
logger.setLevel(logging.INFO)

# Log import completed
logger.info('import completed', extra=d)

def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as ex:
        logger.error(repr(ex))
        raise(ex)

def classify_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        logger.info(event)
        body = base64.b64decode(event['body'])
        logger.info(len(body))
        logger.info('Body loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = predict(image_bytes=picture.content)
        logger.info(prediction)

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'file': filename.replace('"', ''), 'predicted': prediction})
        }
    except Exception as ex:
        logging.error(repr(ex))
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'error': repr(ex)})
        }