
#### Whisper dependencies/setup
* Faster whisper docs say to install `pip install nvidia-cublas-cu11 nvidia-cudnn-cu11`
* nvidia-cudnn-cu11 needs to be pinned down to 8 version (latest version 9 is not compatible)
* On the other hand, `nvidia-cublas-cu12` needed to be installed instead of `nvidia-cublas-cu11`
* `LD_LIBRARY_PATH` env variable needs to be set up with value `/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib`. This path can be different on different envs. Use `python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'` to get the value. [Ref](https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file#install-with-pip-linux-only)
* Any error like `Could not load library libcudnn_ops_infer.so.8` would mean lack of cublas/cudnn dependencies or an incompatible version being installed than what is being expected.
