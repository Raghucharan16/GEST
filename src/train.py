from src.data_loader import DataLoaderClass
from src.model import BERT4Rec
from recbole.trainer import Trainer
from tqdm import tqdm
import torch
from pathlib import Path
import yaml

with open(Path(__file__).parent.parent / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def train(dataset_name):
    loader = DataLoaderClass(dataset_name)
    train_data, valid_data, test_data, rec_config, dataset = loader.get_recbole_dataset()
    model = BERT4Rec(rec_config, dataset).to(rec_config['device'])
    trainer = Trainer(rec_config, model)
    trainer.fit(train_data, valid_data, saved=True)
    print('Training complete. Model saved.')

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        train(sys.argv[1])