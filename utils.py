import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

possible_cont_cols = ['FINAL_MARGIN', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME',
                     'SHOT_DIST', 'CLOSE_DEF_DIST', 'TIME_CHUCK', 'CRUNCH_TIME', 'TOTAL_ELAPSED_TIME']
possible_cat_cols = ['W', 'PERIOD', 'PTS_TYPE', 'CLOSEST_DEFENDER_PLAYER_ID', 'player_id']

conf = {
    'file_path': '/Users/jackarmand/Documents/hot_hand/shot_logs.csv',
    'cont_cols': possible_cont_cols,
    'cat_cols': possible_cat_cols,
    'embedding_dims': {'W': 5, 'PERIOD':25, 'PTS_TYPE': 25, 'CLOSEST_DEFENDER_PLAYER_ID':50, 'player_id': 300},
    'cat_layers': [500, 250, 50],
    'cont_layers': [500, 250, 250],
    'deep_layers': [750, 500, 500, 50],
    'num_epochs': 30,
    'learning_rate': 1e-2,
    'optim': 'SGD',
    'weight_decay': 0,
    'momentum': 0.9,
    'criterion': nn.BCEWithLogitsLoss(),
    'device': 'cpu'
}

class Utils:
    def __init__(self, config):
        self.file_path = config['file_path']
        self.cat_cols = config['cat_cols']
        self.cont_cols = config['cont_cols']

    def prep_dataset(self):
        shots_df = pd.read_csv(self.file_path)

        # Find players only with over 500 shots taken on the season
        player_shot_counts = pd.DataFrame(shots_df.groupby('player_id').count()['GAME_ID'])
        players_over_500 = list(player_shot_counts.loc[player_shot_counts['GAME_ID']>500].index)
        shots_df = shots_df.loc[shots_df['player_id'].isin(players_over_500)]

        # Perform feature engineering on time options
        def convert_game_clock_to_seconds(clock_string):
            splits = clock_string.split(':')
            splits = [int(x) for x in splits]
            return splits[0]*60 + splits[1]

        shots_df['GAME_CLOCK'] = shots_df['GAME_CLOCK'].apply(convert_game_clock_to_seconds)

        time_chunks = []
        crunch_time = []
        for row_idx in range(len(shots_df)):
            row = shots_df.iloc[row_idx]
            game_clock = row['GAME_CLOCK']
            shot_clock = row['SHOT_CLOCK']
            if game_clock < 3:
                time_chunks.append(1)
            elif shot_clock < 3:
                time_chunks.append(1)
            else:
                time_chunks.append(0)

            period = row['PERIOD']
            if period == 4:
                if game_clock < 240:
                    crunch_time.append(1)
                else:
                    crunch_time.append(0)
            else:
                crunch_time.append(0)

        shots_df['TIME_CHUCK'] = time_chunks
        shots_df['CRUNCH_TIME'] = crunch_time

        shots_df['TOTAL_ELAPSED_TIME'] = (shots_df['PERIOD']-1) * 12 * 60 + shots_df['GAME_CLOCK']

        # Transformations to categorical columns to prepare for modeling

        self.w_map = {'W':1, 'L':0}
        shots_df['W'] = shots_df['W'].map(self.w_map)

        period_options = list(set(list(shots_df['PERIOD'])))
        self.period_map = dict(zip(period_options, list(np.arange(len(period_options)))))
        shots_df['PERIOD'] = shots_df['PERIOD'].map(self.period_map)
        self.id = shots_df.loc[~shots_df['PERIOD'].isin(list(self.period_map.values()))]

        self.pts_type_map = {2:0, 3:1}
        shots_df['PTS_TYPE'] = shots_df['PTS_TYPE'].map(self.pts_type_map)

        defender_list = list(set(list(shots_df['CLOSEST_DEFENDER_PLAYER_ID'])))
        self.defender_map = dict(zip(defender_list, list(np.arange(len(defender_list)))))

        self.player_list = players_over_500.copy()
        self.player_map = dict(zip(self.player_list, list(np.arange(len(self.player_list)))))

        shots_df['CLOSEST_DEFENDER_PLAYER_ID'] = shots_df['CLOSEST_DEFENDER_PLAYER_ID'].map(self.defender_map)
        shots_df['player_id'] = shots_df['player_id'].map(self.player_map)

        # Creating y values with 1s and 0s for made or missed
        self.y_col = ['SHOT_RESULT']
        result_map = {'made':1, 'missed':0}
        shots_df['SHOT_RESULT'] = shots_df['SHOT_RESULT'].map(result_map)

        self.possible_cont_cols = ['FINAL_MARGIN', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME',
                     'SHOT_DIST', 'CLOSE_DEF_DIST']

        self.possible_cat_cols = ['W', 'PERIOD', 'PTS_TYPE', 'CLOSEST_DEFENDER_PLAYER_ID', 'player_id']

        shots_df.reset_index(inplace=True, drop=True)
        for idx in range(len(shots_df)):
            if pd.isnull(shots_df['SHOT_CLOCK'].iat[idx]):
                shots_df['SHOT_CLOCK'].iat[idx] = shots_df['GAME_CLOCK'].iat[idx]
        shots_df.dropna(axis='rows', how='any', inplace=True)
        for col in self.cont_cols:
            if col not in ['TIME_CHUCK', 'CRUNCH_TIME']:
                shots_df[col] = (shots_df[col]-shots_df[col].mean())/shots_df[col].std()

        self.shots_df = shots_df

    def create_torch_loaders(self):

        # Want training and test sets to be seperated by player, so that for each player's shots specifically 2/3 are
        # in training and 1/3 is in testing
        np.random.seed(0)

        self.total_train_set = []
        self.total_test_set = []

        for player in np.arange(len(self.player_list)):
            player_df = self.shots_df.loc[self.shots_df['player_id'] == player]
            player_idx = list(player_df.index)

            test_set = np.random.choice(player_idx, replace=False, size=int(len(player_idx)/3))
            train_set = set(player_idx) - set(test_set)
            train_set = list(train_set)
            test_set = list(test_set)

            self.total_train_set += train_set
            self.total_test_set += test_set

        train_data = self.shots_df.loc[self.total_train_set]
        test_data = self.shots_df.loc[self.total_test_set]

        x_train = train_data[self.cat_cols + self.cont_cols].values
        y_train = train_data[self.y_col].values

        x_test = test_data[self.cat_cols + self.cont_cols].values
        y_test = test_data[self.y_col].values

        train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(x_test), shuffle=False)

        self.train_loader = train_loader
        self.test_loader = test_loader

        all_x = self.shots_df[self.cat_cols + self.cont_cols].values
        all_y = self.shots_df[self.y_col].values
        all_dataset = TensorDataset(torch.Tensor(all_x), torch.Tensor(all_y))
        all_loader = DataLoader(all_dataset, batch_size = len(all_x), shuffle=False)
        self.all_loader = all_loader
