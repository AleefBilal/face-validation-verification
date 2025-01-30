import numpy as np
import boto3
from PIL import Image
import botocore


class FaceEmbeddings:
    def __init__(self, model, threshold:float=0.5, ctx_id=0, det_size=(640, 640)):
        self.face_app = model
        self.threshold = threshold
        self.face_app.prepare(ctx_id=ctx_id, det_size=det_size)

    def preprocess_and_get_embedding(self, image):
        # Open or convert the input image to RGB
        img = Image.open(image).convert("RGB") if isinstance(image, str) else Image.fromarray(image[:, :, ::-1])

        # Detect faces in the image
        faces = self.face_app.get(np.array(img)[:, :, ::-1])

        # Handle cases where no face or more than one face is detected
        if len(faces) != 1:
            return {
                'statusCode': 200,
                'body': {
                    'status': False,
                    'message': "No face detected" if not faces else "Multiple faces detected"
                }
            }

        # Return the normalized embedding of the first detected face
        return faces[0].normed_embedding

    def batch_embeddings_and_similarity(self, image1, image2):
        # Get embeddings for both images
        embeddings = []
        for img in [image1, image2]:
            result = self.preprocess_and_get_embedding(img)
            # Check if the result is an error dictionary
            if isinstance(result, dict):
                return result['body']['message'], False
            embeddings.append(result)

        # Calculate similarity if embeddings are valid
        similarity = np.dot(*embeddings) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        is_same = bool(similarity > self.threshold)
        message = f"{'Same' if is_same else 'Different'} person with similarity: {similarity:.2f}"
        return message, is_same

def download_file_from_s3(s3_key, local_path, bucket_name, **kwargs):
    try:
        s3_client = boto3.client('s3', aws_access_key_id=kwargs['key_id'], aws_secret_access_key=kwargs['secret'])
        s3_client.download_file(bucket_name, s3_key, local_path)
        print(f"File downloaded successfully: {local_path}")
    except botocore.exceptions.ClientError as e:
        error = e.response['Error']
        if error['Code'] == "404":
            raise FileNotFoundError(f"File {s3_key} not found in bucket {bucket_name}.")
        raise e
