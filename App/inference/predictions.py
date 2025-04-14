from processor.tabular_processor import TabularProcessor
from processor.image_processor import ImageProcessor
from processor.text_processor import TextProcessor

from utils.model_loader import EnhancedFusionModel
from utils.dataset import InferenceDataset

import torch
from torch.utils.data import DataLoader

from config import IMAGE_MODEL_NAME, TEXT_MODEL_NAME, TABULAR_DIM, IMAGE_DIM, TEXT_DIM, DEVICE, MODEL_PATH, SCALER_PATH

class MultimodalPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.tabular_processor = TabularProcessor()
        self.image_processor = ImageProcessor(model_name=IMAGE_MODEL_NAME)
        self.text_processor = TextProcessor(model_name=TEXT_MODEL_NAME)

        self.model = EnhancedFusionModel(
            tab_dim=TABULAR_DIM,
            img_dim=IMAGE_DIM,
            text_dim=TEXT_DIM
        )

        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, tabular_data, image_path, text_data, batch_size=1):
        tabular_processed = self.tabular_processor.process(tabular_data)
        image_processed = self.image_processor.process(image_path)
        text_processed = self.text_processor.process(text_data)

        dataset = InferenceDataset(tabular_data=tabular_processed, image_features=image_processed, text_features=text_processed)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                tab_batch, img_batch, text_batch = [b.to(DEVICE) for b in batch]
                outputs = self.model(tab_batch, img_batch, text_batch)
                predictions.extend(outputs.cpu().numpy().flatten().tolist())

        predictions = [pred * 1000000 for pred in predictions]
        return predictions





