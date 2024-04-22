from google.cloud.aiplatform import Model

model_id = ""

model = Model(model_id)
endpoint = model.deploy(
    machine_type="n1-standard-2",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)
