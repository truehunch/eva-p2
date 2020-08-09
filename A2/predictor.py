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
import torch.nn.functional as F
from PIL import Image

from utils import class_labels, classes
from model import Net

project_name = 'predictor-assignment2'

# Initialize logging
FORMAT = '%(asctime)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'user': project_name}
logger = logging.getLogger(project_name)
logger.setLevel(logging.INFO)

S3_BUCKET = 'evap2-dataset'
MODEL_PATH = 'mobilenet_flight.pth'

# Log import completed
logger.info('import completed', extra=d)

# Download model
s3 = boto3.client('s3')

try:
    # Loading mobilenet v2
    obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
    logger.info('Creating byte stream', extra=d)

    byte_stream = io.BytesIO(obj['Body'].read())
    logger.info('Loading mobilenet', extra=d)

    model = torch.jit.load(byte_stream)

    sample_input_cpu = torch.randn(1, 3, 224, 224)
    model(sample_input_cpu)
    
    logger.info('Model loaded', extra=d)

except Exception as ex:
    logging.error(repr(ex))
    raise(ex)

def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5276, 0.5793, 0.6078],
                                 std=[0.1930, 0.1871, 0.2064]),
        ])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as ex:
        logger.error(repr(ex))
        raise(ex)

def predict(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    predictions = F.softmax(model(tensor))
    predictions = predictions.detach().numpy().tolist()
    result = dict(zip(classes, predictions))
    logging.info(result)
    return result
