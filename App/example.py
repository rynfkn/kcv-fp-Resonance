from inference.predictions import MultimodalPredictor
import numpy as np

def main():
    # Initialize the predictor
    predictor = MultimodalPredictor()

    # Example property data
    property_data = {
        'Kamar Tidur': 3,
        'Kamar Mandi': 2,
        'Luas Tanah': 50.0,
        'Luas Bangunan': 68.0,
        'Nama Perumahan': 'Pakuwon City',
        'Sertifikat': 'SHM',
        'Carpots': 1,
        'Daya Listrik': np.nan,
        'Interior': 'Semi Furnished',
        'Jumlah Lantai': 1,
        'Orientasi Bangunan': 'Selatan',
        'Tahun Dibangun': 2021,
        'Garasi': 1,
        'Latitude': -7.3132,
        'Longitude': 112.7672,
        'City': 'Surabaya',
        'District': 'Pakuwon City',
    }

    # Path to property image
    image_path = "test_image.png"

    # Property description text
    property_description = "Dijual.cepat Bulan Ini Harus Laku, Harga Spesial.di Bulan Ramadhan.. Siapa Cepat.pasti.dapat Rejeki.jangan.lewatkan Kesempatan.ini."

    # Make prediction
    price_prediction = predictor.predict(property_data, image_path, property_description)

    print(f"Predicted property price: Rp {price_prediction[0]:,.2f}")


if __name__ == "__main__":
    main()