import json
import logging
import os
import traceback
from sys import exc_info
from modules import FaceEmbeddings, download_file_from_s3
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
import cv2

# Load environment variables
load_dotenv('.env')

# Logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Constants
os.makedirs('/tmp/local', exist_ok=True)
USE_S3 = os.getenv('USE_S3', 'ON').upper() == 'ON'
BUCKET_NAME = os.getenv('BUCKET_NAME')
KEY_ID = os.getenv('KEY_ID')
SECRET = os.getenv('SECRET')

THRESHOLD = os.getenv('THRESHOLD', 0.5)

# Face analysis setup
face_app = FaceAnalysis(
    name="auraface",
    providers=["CPUExecutionProvider"],
    root=".",
)
verification_model = FaceEmbeddings(model=face_app, threshold=float(THRESHOLD))


def handle_verification(event, context):
    local_album_path = None
    local_selfie_path = None

    try:
        body = event.get('body', json.dumps(event)).strip()
        body = json.loads(body)
        if 'aleef' in body:
            return {'payload': {"selfie": "# contains selfie path", "album": "#contains album path"}}

        if 'album' not in body or 'selfie' not in body:
            raise ValueError("Album or selfie picture is missing from payload.")

        album_path = body['album']
        selfie_path = body['selfie']
        logger.info(f'Received paths: Album: {album_path}, Selfie: {selfie_path}')

        # Define local paths
        local_album_path = f'/tmp/local/{os.path.basename(album_path)}'
        local_selfie_path = f'/tmp/local/{os.path.basename(selfie_path)}'

        # Download images from S3 or use local paths
        if USE_S3:
            download_file_from_s3(
                s3_key=album_path,
                local_path=local_album_path,
                bucket_name=BUCKET_NAME,
                key_id=KEY_ID,
                secret=SECRET
            )
            download_file_from_s3(
                s3_key=selfie_path,
                local_path=local_selfie_path,
                bucket_name=BUCKET_NAME,
                key_id=KEY_ID,
                secret=SECRET
            )
        else:
            local_album_path = album_path
            local_selfie_path = selfie_path

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
    finally:
        # Ensure cleanup of temporary files
        if USE_S3:
            if local_album_path and os.path.exists(local_album_path):
                os.remove(local_album_path)
            if local_selfie_path and os.path.exists(local_selfie_path):
                os.remove(local_selfie_path)
            logger.info("Temporary files deleted.")
        else:
            logger.info("Files are not deleted.")
