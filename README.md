# Face Verification Lambda Function

This repository contains a **Face Verification Lambda Function** designed to:
1. Detect human faces in images.
2. AI generated image prevention and detection.
3. Verify if two images belong to the same individual.

The function leverages:
- **[AuraFace-v1](https://huggingface.co/fal/AuraFace-v1)** for face embedding and similarity verification.
- **[Trained AI Image Classifier (CatBoost)](https://huggingface.co/Aleef/AI-Image-Classification)** for AI generated Image classification and prevention.

Special thanks to the developers of these models!

---

## Features
- Detects faces in provided images.
- Validates that both the album and selfie images contain a single face.
- Validate that the given image is not AI generated Image
- Verifies whether the faces in the two images belong to the same individual.
- Supports both **AWS S3** integration and local file paths.

---

## Important Note
- Run `python model_download.py` script to download AuraFace and CatBoost.

---

## Requirements
- **Docker** for local testing.
- **Postman** or any API testing tool for testing the function.

---

## Running Locally

### Step 1: Start the Lambda Function in Docker
Run the following command to spin up the Lambda function in a Docker container:

```bash
sudo docker run -it --rm --platform linux/amd64 \
  -v ~/.aws-lambda-rie:/aws-lambda \
  -p 9000:8080 \
  --entrypoint /aws-lambda/aws-lambda-rie <your-image-name> \
  /usr/local/bin/python3.11 -m awslambdaric handler.<your-handler-function-name>
```

### Step 2: Test with Postman
1. Open Postman.
2. Use the following URL:
   ```text
   http://localhost:9000/2015-03-31/functions/function/invocations
   ```
3. Send a **POST** request with a JSON body similar to:

```json
{
    "album": "<path-to-album-image>",
    "selfie": "<path-to-selfie-image>"
}
```

---

## Configuration

### `.env` File
Set the following variables in a `.env` file:

```env
USE_S3=ON                # Set to 'OFF' to use local paths
BUCKET_NAME=<bucket-name>
KEY_ID=<aws-key-id>
SECRET=<aws-secret>
THRESHOLD = 0.5          # WOrks fine on 0.5
```

### Using Local Paths
To test locally:
- Set `USE_S3=OFF` in the `.env` file.
- Provide absolute paths to the `album` and `selfie` images in your request payload.

### Switching to AWS S3
For production or Lambda deployment:
- Set `USE_S3=ON` in the `.env` file.
- Provide S3 paths for the `album` and `selfie` in your request payload.

---

## Workflow

1. **Input**: The function accepts two images:
   - **Album**: A reference image of the person.
   - **Selfie**: A selfie image of the same person.

2. **Process**:
   - Downloads the images from S3 (if `USE_S3=ON`).
   - Validates that each image contains exactly one face.
   - Validate that given image is not AI generated image.
   - Extracts embeddings and calculates cosine similarity.

3. **Output**:
   - **Success**: Confirms if both images are of the same individual.
   - **Error**: Provides clear error messages if validation fails.

---

## Models Used


### 1. **Face Detection & Verification**
- **[AuraFace-v1](https://huggingface.co/fal/AuraFace-v1)**  
  Detect Faces and Generates embeddings and compares cosine similarity to verify identity.
- **[In-House Trained AI Image Classifier (CatBoost)](https://huggingface.co/Aleef/AI-Image-Classification)**  
  AI generated Image classification and prevention.


---

## Troubleshooting

- **Images Not Found**: Ensure the provided file paths or S3 keys are correct.
- **Multiple Faces Detected**: Ensure only one person is visible in each image.
- **AI Image detected**: Do not use AI generated image. AI generated Image is not acceptable.
- **Similarity Check Fails**: Images of the same individual may differ due to lighting or angles; try clearer images.

---

## Future Work

- Develop methods to prevent the face validation pipeline from accepting faces displayed on a screen.

---

## License & Acknowledgments
- **Face Verification Model**: [AuraFace-v1](https://huggingface.co/fal/AuraFace-v1)
- **AI Image Classification (In-house)**: [Trained AI Image Classifier (CatBoost)](https://huggingface.co/Aleef/AI-Image-Classification)

Thank you to the contributors of these models for making them publicly available.