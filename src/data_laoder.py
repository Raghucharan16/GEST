import pandas as pd
import yaml
from pathlib import Path
from recbole.data.utils import create_dataset as recbole_create_dataset
from recbole.data import data_preparation
from sklearn.model_selection import train_test_split

with open(Path(__file__).parent.parent / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class DataLoaderClass:
    def __init__(self, dataset_name):
        self.data_path = Path(f'../data/{dataset_name}')
        self.processed_path = self.data_path / 'processed'
        self.processed_path.mkdir(exist_ok=True)
        self.win_len = config['win_len']
        self.df = self.load_and_process()
        self.convert_to_recbole_format()

    def load_and_process(self):
        df = pd.read_csv(self.data_path / 'raw.csv')  # Assume cols: session_id, item_id, category_id, timestamp, price; rename if needed
        df = df.groupby('session_id').filter(lambda x: len(x) >= 3)
        item_counts = df['item_id'].value_counts()
        df = df[df['item_id'].isin(item_counts[item_counts >= 5].index)]
        df['dwell_time'] = df.groupby('session_id')['timestamp'].diff().fillna(0)
        # For RecBole, use session_id as user_id, sort by timestamp
        df = df.sort_values(['session_id', 'timestamp'])
        # Map items to tokens if needed (RecBole handles)
        return df

    def convert_to_recbole_format(self):
        # Split to train/val/test (80/10/10)
        sessions = self.df.groupby('session_id')
        train_df, val_df, test_df = [], [], []
        for _, sess in sessions:
            sess = sess.iloc[-self.win_len:] if len(sess) > self.win_len else sess
            if len(sess) < 3: continue
            train, temp = train_test_split(sess, test_size=0.2)
            val, test = train_test_split(temp, test_size=0.5)
            train_df.append(train)
            val_df.append(val)
            test_df.append(test)
        train_df = pd.concat(train_df)
        val_df = pd.concat(val_df)
        test_df = pd.concat(test_df)
        # Save as .inter files
        columns = ['session_id:token', 'item_id:token', 'timestamp:float']
        train_df[columns[:3]].to_csv(self.processed_path / f'{self.data_path.name}.train.inter', sep='\t', index=False)
        val_df[columns[:3]].to_csv(self.processed_path / f'{self.data_path.name}.valid.inter', sep='\t', index=False)
        test_df[columns[:3]].to_csv(self.processed_path / f'{self.data_path.name}.test.inter', sep='\t', index=False)
        # For seq, RecBole will handle grouping

    def get_recbole_dataset(self):
        rec_config = config['recbole_config']
        rec_config['dataset'] = self.data_path.name
        rec_config['data_path'] = str(self.processed_path)
        recbole_conf = recbole.config.Config(config_dict=rec_config)
        dataset = recbole_create_dataset(recbole_conf)
        train, val, test = data_preparation(recbole_conf, dataset)
        return train, val, test, recbole_conf, dataset

    def process_session(self, session):
        # For Streamlit: Pad to win_len, return as list of item_ids
        if len(session) < self.win_len:
            session = [0] * (self.win_len - len(session)) + session
        else:
            session = session[-self.win_len:]
        return session