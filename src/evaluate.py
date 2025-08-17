from src.data_loader import DataLoaderClass
from src.model import BERT4Rec
from recbole.trainer import Trainer
from pathlib import Path
import yaml

with open(Path(__file__).parent.parent / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def evaluate(dataset_name):
    loader = DataLoaderClass(dataset_name)
    _, _, test_data, rec_config, dataset = loader.get_recbole_dataset()
    model = BERT4Rec(rec_config, dataset).to(rec_config['device'])
    trainer = Trainer(rec_config, model)
    result = trainer.evaluate(test_data)
    print(result)  # Prints MRR@10, Hit@10, NDCG@10 etc.

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        evaluate(sys.argv[1])