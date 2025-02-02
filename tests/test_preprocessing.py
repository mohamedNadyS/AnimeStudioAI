import pytest
from PIL import Image
import numpy as np
from data.preprocessing import AnimePreprocessor

@pytest.fixture
def sample_image():
    return Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype='uint8'))

def test_preprocessing(sample_image):
    preprocessor = AnimePreprocessor()
    tensor = preprocessor.process(sample_image)
    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.min() >= -1
    assert tensor.max() <= 1

def test_batch_processing(sample_image):
    preprocessor = AnimePreprocessor()
    batch = [sample_image, sample_image]
    tensor = preprocessor.process_batch(batch)
    assert tensor.shape == (2, 3, 512, 512)
