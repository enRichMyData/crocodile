# crocodile/__init__.py
import os
import warnings

import absl.logging
import aiohttp
import nltk
from nltk.corpus import stopwords

MY_TIMEOUT = aiohttp.ClientTimeout(
    total=30,  # Total time for the request
    connect=5,  # Time to connect to the server
    sock_connect=5,  # Time to wait for a free socket
    sock_read=25,  # Time to read response
)

# Suppress certain Keras/TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Do not pass an `input_shape`.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Compiled the loaded model, but the compiled metrics.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Error in loading the saved optimizer state.*"
)

# Set logging levels
# tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)

# NLTK setup
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Define STOP_WORDS
STOP_WORDS = set(stopwords.words("english"))

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import the main class
from .crocodile import Crocodile as Crocodile
from .result_fetcher import CrocodileResultFetcher as CrocodileResultFetcher