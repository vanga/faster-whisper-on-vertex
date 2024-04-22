import os
from google.cloud.aiplatform.prediction import LocalModel
from google.cloud.aiplatform import Model
from faster_whisper_predictor import FasterWhisperPredictor

GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
REGISTERY_NAMESPACE = os.getenv("REGISTERY_NAMESPACE")

image_uri = f"us-central1-docker.pkg.dev/{GOOGLE_PROJECT_ID}/{REGISTERY_NAMESPACE}/faster-whisper:latest"

local_model = LocalModel.build_cpr_model(
    "./",
    image_uri,
    predictor=FasterWhisperPredictor,
    requirements_path="./requirements.txt",
)

local_model.push_image()

local_model = LocalModel(
    serving_container_image_uri=image_uri,
    serving_container_environment_variables={
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/nccl2/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib",
        "VERTEX_CPR_WEB_CONCURRENCY": 1,
    },
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
)
local_model.push_image()
print(local_model.get_serving_container_spec())

model = Model.upload(
    local_model=local_model,
    display_name="faster-whisper",
)
