import numpy as np
import pandas as pd

import os
import glob
import subprocess

import argparse
import joblib

from tensorflow.keras.models import model_from_json 
from rdkit import Chem
from rdkit.Chem import AllChem

import xgboost as xgb

def argument_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-o', '--output_dir', required=True, help="Output directory")
    parser.add_argument('-i', '--input_file', required=True, help="Input file")
    return parser

def read_smiles(input_file):
    cid_list = []
    smiles_list = []
    with open(input_file, 'r') as fp:
        fp.readline()
        for line in fp:
            sptlist = line.strip().split(',')
            smiles = sptlist[0].strip()
            cid = sptlist[1].strip()            
            cid_list.append(cid)
            smiles_list.append(smiles)
    
    return cid_list, smiles_list

def calculate_feature(smiles_list):
    features = []
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        bits = fp.ToBitString()
        feature = []
        for f in bits:
            feature.append(int(f))
        features.append(feature)
    return np.asarray(features)

def predict_toxicity_using_other_models(model_file, features):
    
    if 'XGBoost' in model_file:
        features = xgb.DMatrix(data=features)
        model = joblib.load(model_file)
        
    else:
        model = joblib.load(model_file)
    
    y_predicted = model.predict(features)

    return y_predicted

def predict_toxicity_using_dnn_model(target_dir, features):
    trained_model = target_dir+ 'DNN.json'
    trained_weight = target_dir+ 'DNN.h5'
    
    json_file = open(trained_model, "r")
    loaded_model_json = json_file.read() 
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(trained_weight)
    
    y_predicted = model.predict(features)
    y_predicted = y_predicted.reshape((-1,))
    return y_predicted

def predict_toxicity(target_species='mouse'):
    target_dir = './models/%s/'%(target_species)
    
    DNN_result = predict_toxicity_using_dnn_model(target_dir, features)
    
    target_model_dir = target_dir + 'LightGBM.pkl'    
    LightGBM_result = predict_toxicity_using_other_models(target_model_dir, features)
    
    target_model_dir = target_dir + 'XGBoost.pkl'    
    XGBoost_result = predict_toxicity_using_other_models(target_model_dir, features)
    
    target_model_dir = target_dir + 'RF.pkl'    
    RF_result = predict_toxicity_using_other_models(target_model_dir, features)
    
    target_model_dir = target_dir + 'SVR.pkl'    
    SVR_result = predict_toxicity_using_other_models(target_model_dir, features)

    return DNN_result, LightGBM_result, XGBoost_result, RF_result, SVR_result

def run_chemprop(input_file, output_dir):
    subprocess.call('chemprop_predict --test_path %s --checkpoint_dir models/chemprop --preds_path %s/chemprop_result.csv --no_cuda'%(input_file, output_dir), shell=True, stderr=subprocess.STDOUT)
    return

def merge_predictions(DNN_result, LightGBM_result, XGBoost_result, RF_result, SVR_result, chemprop_mouse_result):
    pred_values = []
    for i in range(len(DNN_result)):
        ensemble_value = np.mean([DNN_result[i], LightGBM_result[i], XGBoost_result[i], RF_result[i], SVR_result[i], chemprop_mouse_result[i]])
        pred_values.append(ensemble_value)
    return np.asarray(pred_values)

if __name__ == '__main__':
    parser = argument_parser()    
    options = parser.parse_args()
    output_dir = options.output_dir
    input_file = options.input_file
    
    try:
        os.mkdir(output_dir)
    except:
        pass
    
    
    cid_list, smiles_list = read_smiles(input_file)
    features = calculate_feature(smiles_list)
    
    run_chemprop(input_file, output_dir)
    chemprop_result_df = pd.read_csv(output_dir+'/chemprop_result.csv')
    chemprop_mouse_result = chemprop_result_df['MOUSE'].values
    chemprop_rat_result = chemprop_result_df['RAT'].values
    
    DNN_result, LightGBM_result, XGBoost_result, RF_result, SVR_result = predict_toxicity('mouse', features)
    mouse_ensemble_values = merge_predictions(DNN_result, LightGBM_result, XGBoost_result, RF_result, SVR_result, chemprop_mouse_result)
    
    result_df = pd.DataFrame()
    result_df['Mouse DNN'] = DNN_result
    result_df['Mouse LightGBM'] = LightGBM_result
    result_df['Mouse XGBoost'] = XGBoost_result
    result_df['Mouse RF'] = RF_result
    result_df['Mouse SVR'] = SVR_result
    result_df['Mouse chemprop'] = chemprop_mouse_result
    result_df['Mouse ENSEMBLE'] = mouse_ensemble_values
    
    DNN_result, LightGBM_result, XGBoost_result, RF_result, SVR_result = predict_toxicity('rat', features)
    rat_ensemble_values = merge_predictions(DNN_result, LightGBM_result, XGBoost_result, RF_result, SVR_result, chemprop_rat_result)
    
    result_df['Rat DNN'] = DNN_result
    result_df['Rat LightGBM'] = LightGBM_result
    result_df['Rat XGBoost'] = XGBoost_result
    result_df['Rat RF'] = RF_result
    result_df['Rat SVR'] = SVR_result
    result_df['Rat chemprop'] = chemprop_rat_result
    result_df['Rat ENSEMBLE'] = rat_ensemble_values
    
    result_df.index = cid_list
    result_df.to_csv(output_dir + '/Toxicity_prediction_result.csv')
    