from .inference import inference_recognizer, init_recognizer
from .test import multi_gpu_test, single_gpu_test
from .train import train_model,train_model_1

__all__ = [
    'train_model', 'train_model_1', 'init_recognizer', 'inference_recognizer', 'multi_gpu_test',
    'single_gpu_test'
]
