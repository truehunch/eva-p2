# Label  
try:
    import unzip_requirements
except ImportError:
    pass

# Imagenet label names
class_labels = {0: 'Amar_Miracle', 
                1: 'Dinesh_K', 
                2: 'Sahal_K', 
                3: 'Siva_Sankar', 
                4: 'Suparna_S'}

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

import math

from PIL import Image

project_name = 'assignment4-face-recognition'

# Initialize logging
FORMAT = '%(asctime)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'user': project_name}
logger = logging.getLogger(project_name)
logger.setLevel(logging.INFO)

S3_BUCKET = 'evap2-dataset'
MODEL_PATH = 'facenet_custom.pth'

# Log import completed
logger.info('import completed', extra=d)

# Download model
s3 = boto3.client('s3')

try:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
    logger.info('Creating byte stream', extra=d)

    byte_stream = io.BytesIO(obj['Body'].read())
    logger.info('Loading model', extra=d)

    model = torch.jit.load(byte_stream)
    logger.info('Model loaded', extra=d)

except Exception as ex:
    logging.error(repr(ex))
    raise(ex)


def face_label_predictor(img):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    device = 'cpu'
    x = data_transforms['val'](img).unsqueeze(dim=0)
    x = x.to(device)
    model.to(device)
    y = model(x)
    _, preds = torch.max(y, 1)
    return class_labels[preds[0].item()]


def face_recognition(event, context):
    #try:
    content_type_header = event['headers']['content-type']
    logger.info(event)
    body = base64.b64decode(event['body'])
    logger.info('Body loaded')

    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]

    image_bytes = io.BytesIO(picture.content)
    img = Image.open(image_bytes)

    prediction = face_label_predictor(img=img)

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

    '''
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
    '''