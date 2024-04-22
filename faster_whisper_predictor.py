import logging
from typing import Any, Dict
import GPUtil
import pynvml
from google.cloud import storage
import os
from faster_whisper import WhisperModel
from pathlib import Path
import pydub


from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FasterWhisperPredictor(Predictor):
    def __init__(self) -> None:
        self.model_size = "large-v3"
        self.model = None

    def _debug(self):
        try:

            gpus = GPUtil.getGPUs()

            if gpus:
                # Print out the available GPUs
                for gpu in gpus:
                    logger.warn(
                        f"ID: {gpu.id}  Name: {gpu.name}  GPU Utilization: {gpu.load * 100}%"
                    )
            else:
                logger.warn("No GPU devices found")
            # Initialize NVML
            pynvml.nvmlInit()

            # Get the driver version
            driver_version = pynvml.nvmlSystemGetDriverVersion()

            logger.warn("CUDA Driver Version: %s", driver_version)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error: %s", e)
        try:
            # run a command
            os.system("nvidia-smi")
        except Exception as e:
            logger.error("Error smi: %s", e)

        try:
            # run a command
            os.system("ls /usr/local/cuda")
        except Exception as e:
            logger.error("Error local cuda list: %s", e)

        try:
            # run a command
            os.system("cat /proc/driver/nvidia/version")
        except Exception as e:
            logger.error("Error proc version: %s", e)

        # print env variables
        logger.warn(os.environ)

    def load(self, artifacts_uri: str) -> None:
        self.model = WhisperModel(
            self.model_size, device="cuda", compute_type="float16"
        )

    def _convert_audio_format(self, src: Path):

        suffix = src.suffix
        if suffix == ".wav":
            return str(src)
        output_file = src.with_suffix(".wav")
        if output_file.exists():
            return str(output_file)
        audio = pydub.AudioSegment.from_file(src)
        converted_audio = (
            audio.set_frame_rate(16100)
            .set_channels(1)
            .export(format="wav", parameters=["-q", "0"])
        )
        output_file.write_bytes(converted_audio.read())
        return str(output_file)

    def _download(self, storage_path: str, local_path: str) -> None:
        client = storage.Client()
        with open(local_path, "wb") as fo:
            client.download_blob_to_file(storage_path, fo)

    def predict(self, body: list[dict]) -> dict:
        instances = body["instances"]
        outputs = []
        for instance in instances:
            storage_path = instance["path"]  # "gs://bucket/path"
            # given storage path of google bucket, download and write to a temporary location
            # download google object
            tmp_path = "/tmp/audio.wav"
            self._download(storage_path, tmp_path)
            # convert to wav format
            model_input_path = self._convert_audio_format(Path(tmp_path))

            segments, _ = self.model.transcribe(model_input_path)
            transcription = "".join([segment.text for segment in segments])
            outputs.append(transcription)
        return {"data": outputs}
