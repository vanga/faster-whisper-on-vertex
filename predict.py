import os
from google.cloud.aiplatform.prediction import LocalModel
from faster_whisper_predictor import FasterWhisperPredictor

GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
REGISTERY_NAMESPACE = os.getenv("REGISTERY_NAMESPACE")
image_uri = f"us-central1-docker.pkg.dev/{GOOGLE_PROJECT_ID}/{REGISTERY_NAMESPACE}/faster-whisper:latest"

local_model = LocalModel(
    serving_container_image_uri=image_uri,
    serving_container_environment_variables={
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib",
        "VERTEX_CPR_WEB_CONCURRENCY": 1,
    },
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
)

with local_model.deploy_to_local_endpoint(
    gpu_count=1, host_port=8083
) as local_endpoint:
    health_check_response = local_endpoint.run_health_check()
    print(health_check_response.text)
    print(health_check_response.request.path_url)
    predict_response = local_endpoint.predict(
        request_file="./input.txt", headers={"Content-Type": "application/json"}
    )
    print(predict_response)
