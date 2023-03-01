import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


def get_X_y_smiles(df, task, smiles_column='canonical_smiles', get_y=True, get_smiles=True):
    '''
    Extract X, y (+ smiles) from a dataframe to use as input for the models
    :param df: dataframe containing the input features (CDDDs) and the class labels and smiles (if get_y and get_smiles are True)
    :param task: name of the column containing the class labels
    :param smiles_column: name of the column (in input_df) containing the input SMILES
    :param get_y: get an array with the class labels for task
    :param get_smiles: get the smiles corresponding to X and y
    :return: X, y, smiles arrays
    '''
    subset = df.dropna(subset=[task], how='any')
    feat_col = [c for c in subset.columns if 'cddd_' in c]
    
    l1 = len(subset)
    subset = subset.dropna(subset=feat_col, axis=0)
    if l1-len(subset) > 0:
        print(f'Removed {l1-len(subset)} results from table due to missing features.')

    X = subset[feat_col].values
    if get_y:
        y = np.array(subset[task].tolist())
        assert(np.isnan(y).any()==False)
    else:
        y = []
    
    smiles = np.array(subset[smiles_column].tolist()) if get_smiles else []
    return X, y, smiles

def predict(Xtest, model_path, task):
    ''' 
    Make predictions on a test set with a NN or RF model 
    :param Xtest: array with the input descriptors for each test sample
    :param model_path: path to the location of the model with which to predict Xtest
    :param task: name of the task on which the model was trained on
    :return: array with predicted probabilities
    '''
    if '_nn_' in model_path or '.pt' in model_path:
        # load model
        hparams = get_hparams(task)
        model = SingleTaskNN(input_size=512, params=hparams)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        # predict probabilities
        predictions = model(torch.from_numpy(Xtest).to('cpu').float()).squeeze(1)
        pred_probability = predictions.cpu().detach().numpy()
        
    else:
        # load model
        model = pickle.load(open(model_path, 'rb'))
        # predict probabilities
        predictions = model.predict_proba(Xtest)
        temp = pd.DataFrame(predictions, columns=['0', '1'])
        pred_probability = temp['1'].values #column with the probability of class 1 
        
    return pred_probability

def get_hparams(task):
    ''' 
    Get optimized hyperparameters for the NN model
    :param task: name of the task on which the model was trained on
    :return: dictionary with the optimized hyperparameters
    '''
    hparams = {'overall': {'num_layers': 2, 'learning_rate': 0.001, 'weight_decay': 1e-05, 'epochs': 50, 'n_units': [500, 100], 'dropout': [0, 0.2]},
               'membrane_potential': {'num_layers': 3, 'learning_rate': 0.0001, 'weight_decay': 0.01, 'epochs': 150, 'n_units': [2000, 2000, 1000], 'dropout': [0, 0.2, 0.3]}
              }

    return hparams[task]

class SingleTaskNN(nn.Module):
    ''' 
    Model architecture for the NN models
    '''
    def __init__(self, input_size, params):
        super().__init__()
        # define layers
        self.fc = nn.ModuleList([nn.Linear(input_size, params['n_units'][0])])
        input_units = params['n_units'][0]
        
        for n_units in params['n_units'][1:]:
            self.fc.append(nn.Linear(input_units, n_units))
            input_units = n_units                            
        self.fc.append(nn.Linear(input_units, 1))
        
        # define dropouts
        self.dropouts = nn.ModuleList([nn.Dropout(p=drop) for drop in params['dropout']])

        # weight initialization
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(mod.bias, 0)

        
    def forward(self, x):
        for i in range(len(self.fc)-1):
            x = F.relu(self.fc[i](x))
            x = self.dropouts[i](x)
                                    
        x = torch.sigmoid(self.fc[-1](x))
        return x


