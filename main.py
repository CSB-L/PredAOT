import numpy as np
import pandas as pd

import os
import glob
import subprocess
import tqdm
import argparse
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem


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
            sptlist = line.strip().split('\t')
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


def predict_aot(model_file, features):
    model = joblib.load(model_file)
    y_predicted = model.predict(features)

    return y_predicted


def predict_toxicity_reg(target_model_dir, features):
    RF_result = predict_aot(target_model_dir, features)
    return RF_result[0]

def predict_toxicity_cls(target_species, features):
    target_dir = './models/'
    
    target_model_dir = target_dir+'rf_cls_%s_v2.pkl'%(target_species)
    RF_result = predict_aot(target_model_dir, features)
    
    return RF_result[0]

def convert_ld50(value):
    result = np.power(10, value)
    return result-1

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
    fp = open(output_dir+'/Toxicity_prediction_result.txt', 'w')
    fp.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%('Smiles', 'Mouse(cls)', 'Mouse LD50(mg/kg)', 'Rat(cls)', 'Rat LD50(mg/kg)', 'Mouse all', 'Rat all'))
    for i in tqdm.tqdm(range(len(smiles_list))):
        smi_list = [smiles_list[i]]
        features = calculate_feature(smi_list)
        
        mouse_cls_result = predict_toxicity_cls('mouse', features)
        if mouse_cls_result == 0:
            target_model_dir = './models/rf_reg_mouse_nt.pkl'
        else:
            target_model_dir = './models/rf_reg_mouse_t.pkl'

        mouse_reg_result = predict_toxicity_reg(target_model_dir, features)        
        mouse_ld50_mpk = convert_ld50(mouse_reg_result)
        
        rat_cls_result = predict_toxicity_cls('rat', features)
        if rat_cls_result == 0:
            target_model_dir = './models/rf_reg_rat_nt.pkl'
        else:
            target_model_dir = './models/rf_reg_rat_t.pkl'
        
        rat_reg_result = predict_toxicity_reg(target_model_dir, features)        
        rat_ld50_mpk = convert_ld50(rat_reg_result)
        
        # conventional mouse
        target_model_dir = './models/rf_conventional_mouse_all.pkl'
        mouse_conventional_result = predict_toxicity_reg(target_model_dir, features)     
        mouse_conventional_ld50_mpk = convert_ld50(mouse_conventional_result)
        
        # conventional rat
        target_model_dir = './models/rf_conventional_rat_all.pkl'
        rat_conventional_result = predict_toxicity_reg(target_model_dir, features)     
        rat_conventional_ld50_mpk = convert_ld50(rat_conventional_result)
        
        mouse_cls = None
        if mouse_cls_result == 0:
            mouse_cls = 'Non-toxic'
        else:
            mouse_cls = 'Toxic'
            
        rat_cls = None
        if rat_cls_result == 0:
            rat_cls = 'Non-toxic'
        else:
            rat_cls = 'Toxic'
            
        fp.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(smiles_list[i], mouse_cls, mouse_ld50_mpk, rat_cls, rat_ld50_mpk, mouse_conventional_ld50_mpk, rat_conventional_ld50_mpk))
    fp.close()
    