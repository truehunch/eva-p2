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

import cv2
import dlib
import numpy as np
import math

project_name = 'assignment3-face-align'

# Initialize logging
FORMAT = '%(asctime)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'user': project_name}
logger = logging.getLogger(project_name)
logger.setLevel(logging.INFO)

S3_BUCKET = 'evap2-dataset'
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
PREDICTOR_LOCAL_PATH = '/tmp/shape_predictor_68_face_landmarks.dat'

# Log import completed
logger.info('import completed', extra=d)

# Download model
s3 = boto3.client('s3')

try:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=PREDICTOR_PATH)
    logger.info('Creating byte stream', extra=d)

    byte_stream = io.BytesIO(obj['Body'].read())

    with open(PREDICTOR_LOCAL_PATH, "wb") as dlib_data_file:
        # Copy the BytesIO stream to the output file
        dlib_data_file.write(byte_stream.getbuffer())

    face_detector = dlib.get_frontal_face_detector()
    face_landmark_predictor = dlib.shape_predictor(PREDICTOR_LOCAL_PATH)

    logger.info('Loaded predictor', extra=d)

except Exception as ex:
    logging.error(repr(ex))
    raise(ex)


def get_faces(img, detector, landmark_predictor):
    logging.error(img.shape)
    face_rects = detector(img, 2)
    faces = []
  
    if len(face_rects) <= 0:
        return faces

    face_img = img.copy()
    H, W, _ = face_img.shape
    for i, face_rect in enumerate(face_rects):
        left, top, right, bottom = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()

    rect = dlib.rectangle(left, top, right, bottom)
    landmarks = landmark_predictor(img, rect)

    # top_left = (left, top)
    # bottom_right = (right, bottom)
    
    # face_img = cv2.rectangle(face_img, top_left, bottom_right, color=(255, 255, 0), thickness=2) 
    
    border = int(2 * min(abs(top - bottom), abs(left - right)))
    top_crop = top - border if top - border > 0 else 0
    bottom_crop = bottom + border if bottom + border < H else H 
    left_crop = left - border if left - border > 0 else 0
    right_crop = right + border if right + border < W else W

    # img_rect = face_img[top_crop:bottom_crop, left_crop:right_crop]

    landmarks_actual = []

    for i, landmark in enumerate(landmarks.parts()):
        x, y, = landmark.x, landmark.y
        landmarks_actual.append((x, y))
        # img_rect = cv2.circle(img_rect, (x, y), radius=1, color=(255, 0, 0), thickness=2)

    faces.append({'idx': i, 
                  'rect_xy': (left, top, right, bottom), 
                  'rect_xy_offsetted': (left_crop, top_crop, right_crop, bottom_crop), 
                  # 'landmarks_offsetted': landmarks_offsetted,
                  'landmarks': landmarks_actual,
                  'img': img,
                  'rect': img[top_crop:bottom_crop, left_crop:right_crop]})
    return faces

# Normalizes a facial image to a standard size given by outSize.
# Normalization is done based on Dlib's landmark points passed as pointsIn
# After normalization, left corner of the left eye is at (0.3 * w, h/3 )
# and right corner of the right eye is at ( 0.7 * w, h / 3) where w and h
# are the width and height of outSize.
def normalize_images_and_landmarks(out_size, img_in, points_in):
    h, w = out_size

    # Corners of the eye in input image
    eyecorner_src = [points_in[36], points_in[45]]

    # Corners of the eye in normalized image
    eyecorner_dst = [(np.int(0.3 * w), np.int(h/3)), 
                    (np.int(0.7 * w), np.int(h/3))]

    # Calculate similarity transform
    tform = similarity_transform(eyecorner_src, eyecorner_dst)

    # Apply similarity transform to input image
    img_out = cv2.warpAffine(img_in, tform, (w, h))

    # reshape points_in from numLandmarks x 2 to numLandmarks x 1 x 2
    points2 = np.reshape(points_in, (points_in.shape[0], 1, points_in.shape[1]))

    # Apply similarity transform to landmarks
    points_out = cv2.transform(points2, tform)

    # reshape points_out to numLandmarks x 2
    points_out = np.reshape(points_out, (points_in.shape[0], points_in.shape[1]))

    return img_out, points_out

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarity_transform(in_points, out_points):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    in_pts = np.copy(in_points).tolist()
    out_pts = np.copy(out_points).tolist()

    # The third point is calculated so that the three points make an equilateral triangle
    xin = c60*(in_pts[0][0] - in_pts[1][0]) - s60*(in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
    yin = s60*(in_pts[0][0] - in_pts[1][0]) + c60*(in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]

    in_pts.append([np.int(xin), np.int(yin)])

    xout = c60*(out_pts[0][0] - out_pts[1][0]) - s60*(out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
    yout = s60*(out_pts[0][0] - out_pts[1][0]) + c60*(out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]

    out_pts.append([np.int(xout), np.int(yout)])

    # Now we can use estimateRigidTransform for calculating the similarity transform.
    # tform = cv2.estimateRigidTransform(np.array([in_pts]), np.array([out_pts]), False)
    tform, _ = cv2.estimateAffinePartial2D(np.array([in_pts]), np.array([out_pts]), False)

    # tform = cv2.getAffineTransform(eyecorner_src, eyecorner_dst)
    # tform = cv2.getAffineTransform(np.array([in_pts], dtype=np.float32), np.array([out_pts], dtype=np.float32))
    return tform

def get_aligned_face(img):
    faces = get_faces(img, face_detector, face_landmark_predictor)
    logger.info(f'Got {len(faces)}) faces.')

    if len(faces) == 1: 
        face_extracted = faces[0]
        h, w, _ = face_extracted['img'].shape
        face_in = np.float32(face_extracted['img']) / 255.0
        img_out, points_out = normalize_images_and_landmarks((h, w), face_in, np.array(face_extracted['landmarks']))
        for i, (x, y) in enumerate(points_out):
            img_out = cv2.circle(img_out, (x, y), radius=2, color=(255, 0, 0), thickness=2)
        img_out = np.uint8(img_out * 255)

        return {
            'img_out': img_out,
            'landmarks_out': points_out,
            'status': True,
            'message': 'Successfully aligned'
        }
    elif len(faces) == 0: 
        return {
            'img_out': None,
            'landmarks_out': None,
            'status': False,
            'message': 'Failed as no faces were found'
        }
    else: 
        return {
            'img_out': None,
            'landmarks_out': None,
            'status': False,
            'message': f'Failed as multiple faces(expected 1 got {len(faces)}) were found.'
        }

def face_align(event, context):
    try:
        content_type_header = event['headers']['content-type']
        logger.info(event)
        body = base64.b64decode(event['body'])
        logger.info('Body loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        
        image_stream = io.BytesIO(picture.content)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = get_aligned_face(img=img)

        logger.info(result)

        if result['status']:
            _, img_encoded = cv2.imencode('.png', result['img_out'])
            img_bytes = img_encoded.tobytes()

            return {
                'statusCode': 200,
                "isBase64Encoded": True,
                'headers': {
                    'Content-Type': 'image/png',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Credentials': True
                },
                'body': base64.b64encode(img_bytes).decode("utf-8")
            }

        else:
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
                'body': json.dumps({'file': filename.replace('"', ''), 
                                    'message': result['message']})
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