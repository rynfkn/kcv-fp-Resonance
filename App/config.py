import os
import torch

SCALER_PATH = 'model/scaler.pkl'
MODEL_PATH = 'model/best_model.pth'

TABULAR_DIM = 17
IMAGE_DIM = 1024
TEXT_DIM = 1024

IMAGE_MODEL_NAME = "facebook/dinov2-large"
TEXT_MODEL_NAME = "intfloat/multilingual-e5-large"

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

FEATURE_NAMES = ['Kamar Tidur', 'Kamar Mandi', 'Luas Tanah', 'Luas Bangunan',
 'Nama Perumahan', 'Sertifikat', 'Carpots', 'Daya Listrik',
 'Interior', 'Jumlah Lantai', 'Orientasi Bangunan',
 'Tahun Dibangun', 'Garasi', 'Latitude', 'Longitude', 'City',
 'District']

# Temp file storage
TEMP_IMAGE_PATH = "temp_image.jpg"

# Flask settings
DEBUG = os.environ.get('DEBUG', 'False') == 'True'
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 5000))