import json
import logging
import os
import traceback
from sys import exc_info
from modules import FaceEmbeddings
from insightface.app import FaceAnalysis
import cv2


# Logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)

THRESHOLD = os.getenv('THRESHOLD', 0.5)

# Face analysis setup
face_app = FaceAnalysis(
    name="auraface",
    providers=["CPUExecutionProvider"],
    root=".",
)
verification_model = FaceEmbeddings(model=face_app, threshold=float(THRESHOLD))


def handle_verification(event, context):

    try:
        body = event.get('body', json.dumps(event)).strip()
        body = json.loads(body)

        album_path = body['album']
        selfie_path = body['selfie']
        logger.info(f'Received paths: Album: {album_path}, Selfie: {selfie_path}')

        local_album_path = album_path
        local_selfie_path = selfie_path

        logger.info('Images downloaded, starting human face validation.')

        # Load images
        album = cv2.imread(local_album_path)
        selfie = cv2.imread(local_selfie_path)


        # Verification
        verification_results = verification_model.batch_embeddings_and_similarity(
            image1=album, image2=selfie
        )
        return {
            'statusCode': 200,
            'body': {
                'status': verification_results[1],
                'message': verification_results[0]
            }
        }
    except AssertionError:
        logger.error(traceback.format_exc())
        return {
            'statusCode': 400,
            'body': {
                'error': f'{exc_info()[1]}'
            }
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': {
                'error': f"{str(exc_info()[0]).split('class')[1].split('>')[0].strip()}: {exc_info()[1]}"
            }
        }