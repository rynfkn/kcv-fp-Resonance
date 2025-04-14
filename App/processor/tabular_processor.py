import pandas as pd
import pickle
from config import SCALER_PATH

# Data processing modules
class TabularProcessor:
    """Processes tabular data with comprehensive manual encoding"""
    def __init__(self, scaler_path=SCALER_PATH):
        # Load the fitted scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Define complete manual encoding mappings
        self.encoding_mappings = {
            'Sertifikat': {
                'HGB': 0, 'Hak Pakai': 1, 'Lain-lain': 2, 
                'SHM': 3, 'Strata Title': 4
            },
            'Interior': {
                'Full Furnished': 0, 'Semi Furnished': 1, 
                'Unfurnished': 2, 'Unknown': 3
            },
            'Orientasi Bangunan': {
                'Barat': 0, 'Barat Daya': 1, 'Barat Laut': 2, 
                'Hook': 3, 'Selatan': 4, 'Timur': 5, 
                'Timur Laut': 6, 'Unknown': 7, 'Utara': 8
            },
            'City': {
                'Malang': 0, 'Sidoarjo': 1, 'Surabaya': 2
            },
            'District': {
                'Blimbing': 0, 'Dau': 1, 'Dinoyo': 2, 'Dukuh Pakis': 3,
                'Gedangan': 4, 'Griya Shanta': 5, 'Gubeng': 6, 'Jatimulyo': 7,
                'Kedungkandang': 8, 'Kenjeran': 9, 'Lowokwaru': 10, 
                'Malang Kota': 11, 'Merjosari': 12, 'Mojolangu': 13, 
                'Mulyorejo': 14, 'Pakis': 15, 'Pakuwon City': 16, 
                'Pakuwon Indah': 17, 'Permata Jingga': 18, 'Rungkut': 19, 
                'Sidoarjo': 20, 'Soekarno Hatta': 21, 'Sukolilo': 22, 
                'Sukun': 23, 'Surabaya Kota': 24, 'Tenggilis Mejoyo': 25, 
                'Tlogomas': 26, 'Wagir': 27, 'Waru': 28
            },
            'Nama Perumahan': {
                '-': 0, 'Alana Regency Cemandi': 1, 'Amesta living': 2,
                'Anggrek Residence': 3, 'Atrani Residence': 4, 'Babatan Pantai': 5,
                'Barata Jaya': 6, 'Baruk Utara': 7, 'Bratang Binangun': 8,
                'Citra Garden': 9, 'Citra Garden Sidoarjo': 10, 'Citra Harmoni': 11,
                'Citra garden': 12, "D'gardenia City": 13, 'DE IMPERIAL ESTATE': 14,
                'Darmo Permai': 15, 'Dekat Taman Pinang Indah Sidoarjo': 16,
                'Delta Mandala': 17, 'Deltasari Indah': 18, 'Dharma Husada Mas': 19,
                'Dharmahusada Mas': 20, 'Dinoyo': 21, 'Dream Park Regency': 22,
                'Dukuh Kupang': 23, 'Dukuh Kupang Timur soho': 24,
                'Galaxy Bumi Permai': 25, 'Graha Sukolilo Regency': 26,
                'Grand Delta Sari': 27, 'Grand Nature Residence': 28,
                'Green Lake Natural Living': 29, 'Green Orchid Residence': 30,
                'Griya Permata Gedangan': 31, 'Gunung Anyar Baru': 32,
                'Gunung Anyar Utara': 33, 'Istana Mentari': 34, 'Jaya Maspion Permata': 35,
                'Jemursari Regency': 36, 'Jl. Ngagel': 37, 'Joyo Grand': 38,
                'KECIPIR REGENCY': 39, 'Kahuripan Nirwana': 40, 'Kalijudan': 41,
                'Kendalsari': 42, 'Klampis Semolo': 43, 'Koala Regency': 44,
                'Kosagrha': 45, 'Kresna Asri': 46, 'Kupang Baru': 47,
                'Kutisari Indah Utara': 48, 'Kutisari Selatan': 49,
                'Kutisari Selatan XIII': 50, 'Kutisri Indah': 51,
                'Lebak Indah Town House': 52, 'Manyar': 53, 'Manyar Jaya': 54,
                'Manyar Kertoadi': 55, 'Manyar Tirtoyoso': 56, 'Medayu': 57,
                'Medayu Utara': 58, 'Medokan Ayu Tambak': 59,
                'Medokan Sawah Timur Rungkut Surabaya': 60, 'Mojoklangru Kidul': 61,
                'Mulyosari': 62, 'Mulyosari Utara': 63, 'Mutiara City': 64,
                'Ngagel Wasana': 65, 'Nginden': 66, 'Nginden Intan Timur': 67,
                'Nirwana Eksekutif': 68, 'Nirwana Eskekutif': 69, 'Omah View': 70,
                'Pakisjajar': 71, 'Pakuwon City': 72, 'Pantai Mentari': 73,
                'Park Regency Keputih': 74, 'Penjaringan Asri': 75,
                'Penjaringan Sari': 76, 'Pepelegi Indah': 77,
                'Perum Puri Surya Jaya Cluster Valencia Residence Gedangan Sidoarjo': 78,
                'Perum perlian kencana sari': 79, 'Perumahan': 80,
                'Perumahan Bluru Permai Blok C no 11': 81,
                'Perumahan Dharma Husada Indah': 82, 'Perumahan Green Lake': 83,
                'Perumahan Griya Amerta': 84, 'Perumahan Griya Galaxy': 85,
                'Perumahan Griya Pesona Asri': 86, 'Perumahan Istana Safira DAU': 87,
                'Perumahan Medokan Ayu Surabaya': 88, 'Perumahan Mutiara Regency': 89,
                'Perumahan Puri Indah Sidoarjo Kota': 90, 'Perumahan Putra Bangsa': 91,
                'Perumahan Royal Park Regency Rungkut': 92, 'Perumahan Rungkut Mapan': 93,
                'Perumahan Rungkut Menanggal Harapan Surabaya': 94,
                'Perumahan Semolowaru Elok': 95, 'Perumahan Semolowaru Indah 2': 96,
                'Perumahan Taman Pinang Indah': 97,
                'Perumahan Taman Puspa Anggaswangi': 98, 'Perumahan YKP': 99,
                'Perumahan river viuw': 100, 'Pondok Candra': 101,
                'Pondok Candra Indah Waru Rungkut Sidoarjo': 102,
                'Pondok Chandra': 103, 'Pondok Jati': 104, 'Pondok Mutiara': 105,
                'Pondok Tjandra': 106, 'Pondok Tjandra Indah': 107,
                'Prapen indah Kec. Tenggilis Mejoyo Surabaya': 108,
                'Puri Asri Regency': 109, 'Puri Gunung Anyar': 110,
                'Puri Indah Lestari': 111, 'Puri Surya Jaya': 112,
                'Putra Bangsa': 113, 'Rifera Townhouse': 114, 'Rungkut Asri': 115,
                'Rungkut Asri Utara': 116, 'Rungkut Harapan': 117,
                'Rungkut Mapan': 118, 'Rungkut Menanggal': 119,
                'Saphire residence': 120, 'Semampir Tengah': 121,
                'Semolowaru Elok': 122, 'Sentra Point': 123,
                'Simpang Darmo Permai Selatan': 124, 'Springville Residence': 125,
                'Sukolilo Dian Regency 2': 126, 'Sunan Kalijaga': 127,
                'Sutorejo Prima': 128, 'Sutorejo Prima Indah': 129,
                'Sutorejo Timur': 130, 'Sutorejo Utara': 131,
                'TENGGILIS MEJOYO SELATAN': 132, 'Taman Pondok Legi 4': 133,
                'Taman Rivera Regency': 134, 'Taman pondok indah': 135,
                'Taman pondok legi': 136, 'Tambak Medokan Ayu': 137,
                'Tambak medokan ayu VI C': 138, 'Tenggilis Mejoyo': 139,
                'Unknown': 140, 'Valencia garden , gedangan': 141, 'Vbt': 142,
                'Vila bukit tidar': 143, 'Villa Kalijudan Indah': 144,
                'Wellington Park Residence': 145, 'Wisata Semanggi Mangrove': 146,
                'Wisma Mukti': 147, 'Wisma Permai': 148, 'Wisma Permai Waru': 149,
                'Wisma Tropodo': 150, 'Wisma permai tengah': 151,
                'YKP Medokan Asri Rungkut Surabaya Timur': 152,
                'klampis semolo': 153, 'kompleks mojoarum': 154,
                'perum jaya maspion permata': 155,
                'perum taman pinang indah sidoarjo': 156,
                'regency one east point': 157, 'sukolilo Dian Regency': 158
            }
        }

        # Default values for unknown categories
        self.default_values = {
            'Sertifikat': 3,  # 'Lain-lain'
            'Interior': 3,     # 'Unknown'
            'Orientasi Bangunan': 7,  # 'Unknown'
            'City': 0,         # 'Malang'
            'District': 0,     # 'Blimbing'
            'Nama Perumahan': 140  # 'Unknown'
        }

        # Median/mode values for imputation
        self.imputation_values = {
            'numerical': {
                'Luas Bangunan': 138.0,  # Will be set during training
                'Luas Tanah': 120.0,
                'Daya Listrik': 2200.0,
                'Tahun Dibangun': 2021,
                'Latitude' : -7.294603,
                'Longitude' : 112.769703

            },
            'categorical': {
                'Kamar Tidur': 3,
                'Kamar Mandi': 2,
                'Carpots': 1,
                'Garasi': 1,
                'Jumlah Lantai': 2
            }
        }

    def process(self, data):
        """Process tabular data with comprehensive preprocessing"""
        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data])
        
        # Ensure all expected columns are present
        expected_cols = [
            'District', 'City', 'Longitude', 'Latitude', 'Garasi', 
            'Tahun Dibangun', 'Orientasi Bangunan', 'Jumlah Lantai', 
            'Interior', 'Daya Listrik', 'Carpots', 'Sertifikat', 
            'Nama Perumahan', 'Luas Bangunan', 'Luas Tanah', 
            'Kamar Mandi', 'Kamar Tidur'
        ]
        
        for col in expected_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Apply preprocessing steps
        self._handle_missing_values(data)
        self._apply_manual_encoding(data)
        self._handle_special_cases(data)
        
        # Apply the scaler
        scaled_data = self.scaler.transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)
    
    def _handle_missing_values(self, data):
        """Handle missing values according to training logic"""
        # Numerical features
        for col in self.imputation_values['numerical']:
            if col in data.columns and data[col].isna().any():
                data[col].fillna(self.imputation_values['numerical'][col], inplace=True)
        
        # Categorical features
        for col in self.imputation_values['categorical']:
            if col in data.columns and data[col].isna().any():
                data[col].fillna(self.imputation_values['categorical'][col], inplace=True)
        
        # Special cases
        for col in ['Interior', 'Orientasi Bangunan', 'Nama Perumahan']:
            if col in data.columns and data[col].isna().any():
                data[col].fillna("Unknown", inplace=True)
    
    def _apply_manual_encoding(self, data):
        """Apply comprehensive manual encoding"""
        for col, mapping in self.encoding_mappings.items():
            if col in data.columns:
                # Convert to string type for consistent mapping
                data[col] = data[col].astype(str)
                # Handle unknown categories
                data[col] = data[col].apply(
                    lambda x: mapping.get(x, self.default_values[col]))
    
    def _handle_special_cases(self, data):
        """Handle special cases in the data"""
        # Orientation special values
        if 'Orientasi Bangunan' in data.columns:
            orientation_replace = {'1': 'Unknown', '2': 'Unknown', 
                                '3': 'Unknown', '4': 'Unknown'}
            data['Orientasi Bangunan'] = data['Orientasi Bangunan'].replace(
                orientation_replace)
            data['Orientasi Bangunan'] = data['Orientasi Bangunan'].map(
                self.encoding_mappings['Orientasi Bangunan'])
        
        # Fill any remaining NA values
        for col in data.columns:
            if data[col].isna().any():
                if col in self.default_values:
                    data[col].fillna(self.default_values[col], inplace=True)
                elif data[col].dtype.kind in 'biufc':  # numeric
                    data[col].fillna(0, inplace=True)
                else:
                    data[col].fillna("Unknown", inplace=True)