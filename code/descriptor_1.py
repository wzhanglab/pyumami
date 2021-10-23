# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 15:58:08 2019

@author: Renhb
"""
if __name__ == "__main__":
    from rdkit import Chem
    from mordred import Calculator, descriptors
    import pandas as pd
    #from sklearn.preprocessing import StandardScaler
    #from sklearn.svm import SVR
    import numpy as np
    bitter = pd.read_csv("./data/model_data/nonbitter_and_bitter_new+MW.csv")
    sweet = pd.read_csv("./data/model_data/nonsweet_and_sweet_new+MW.csv")
    
    ##########################################
    #########################################
    ##################################
    #bitter descriptor
    #calculate descriptor
    mols_bitter = [Chem.MolFromSmiles(mol) for mol in bitter.SMILES]
    
    calc_bitter = Calculator(descriptors, ignore_3D=True)
    
    
    df = {'mols_bitter':np.array(mols_bitter), 'bitter_smiles':np.array(bitter.SMILES)}  
    t=pd.DataFrame(df)
    dt = {'mols_bitter':np.array(mols_bitter), 'T_F':np.array(t.mols_bitter.isnull()), 'bitter_smiles':np.array(bitter.SMILES), 'bitter':np.array(bitter.Bitter), 'Reference':np.array(bitter.Reference)}
    dt_dt = pd.DataFrame(dt)
    bitter = dt_dt[dt_dt.T_F == False]
    
    bitter_descriptor = calc_bitter.pandas(list(bitter.mols_bitter))
    # Convert to dataframe
    data_bitter = pd.DataFrame(bitter_descriptor)
    
    # Find null rows
    null_values = data_bitter.isnull().sum(axis=1)
    null_rows = data_bitter.loc[null_values > 0]
    
    # Drop null rows
    data_bitter = data_bitter.loc[null_values == 0]
    
    #Drop rawdata including null_rows
    bitter = bitter.drop(index=null_rows._stat_axis.values.tolist())
    
    data_bitter.to_csv('./data/model_data/bitter_descriptors.csv')
    bitter.to_csv('./data/model_data/bitter_all.csv')
    ##########################################
    ##########################################
    ##########################################
    #########################
    
    
    
    
    
    ###############
    ###################################
    ########################################
    ##########################################
    #sweet_descriptor
    #calculate descriptor
    mols_sweet = [Chem.MolFromSmiles(mol) for mol in sweet.SMILES]
    
    calc_sweet = Calculator(descriptors, ignore_3D=True)
    
    df = {'mols_sweet':np.array(mols_sweet), 'sweet_smiles':np.array(sweet.SMILES)}  
    t=pd.DataFrame(df)
    dt = {'mols_sweet':np.array(mols_sweet), 'T_F':np.array(t.mols_sweet.isnull()), 'sweet_smiles':np.array(sweet.SMILES), 'sweet':np.array(sweet.Sweet), 'Reference':np.array(sweet.Reference)}
    dt_dt = pd.DataFrame(dt)
    sweet = dt_dt[dt_dt.T_F == False]
    
    sweet_descriptor = calc_sweet.pandas(list(sweet.mols_sweet))
    # Convert to dataframe
    data_sweet = pd.DataFrame(sweet_descriptor)
    
    # Find null rows
    null_values = data_sweet.isnull().sum(axis=1)
    null_rows = data_sweet.loc[null_values > 0]
    
    # Drop null rows
    data_sweet = data_sweet.loc[null_values == 0]
    sweet = sweet.drop(index=null_rows._stat_axis.values.tolist())
    data_sweet.to_csv('./data/model_data/sweet_descriptors.csv')
    sweet.to_csv('./data/model_data/sweet_all.csv')
    ###############################################
    ###############################################
    ###############################################
    
    
    
    
    
    
    
    ###############################################
    ###############################################
    ###############################################
    #bitswet_descriptor
    #calculate descriptor
    bitswet = pd.read_csv('./data/data_calibration/calibration/last/bitter_sweet_DB_all.csv')
    mols_bitswet = [Chem.MolFromSmiles(mol) for mol in bitswet.SMILES]
    
    calc_bitswet = Calculator(descriptors, ignore_3D=True)
    
    
    df = {'mols_bitswet':np.array(mols_bitswet), 'bitswet_smiles':np.array(bitswet.SMILES)}  
    t=pd.DataFrame(df)
    dt = {'mols_bitswet':np.array(mols_bitswet), 'T_F':np.array(t.mols_bitswet.isnull()), 
          'bitswet_smiles':np.array(bitswet.SMILES), 'id':np.array(bitswet.id), 'cid':np.array(bitswet.cid)}
    dt_dt = pd.DataFrame(dt)
    bitswet = dt_dt[dt_dt.T_F == False]
    
    bitswet_descriptor = calc_bitswet.pandas(list(bitswet.mols_bitswet))
    # Convert to dataframe
    data_bitswet = pd.DataFrame(bitswet_descriptor)
    
    # Find null rows
    null_values = data_bitswet.isnull().sum(axis=1)
    
    null_rows = data_bitswet.loc[null_values > 0]
    
    # Drop null rows
    data_bitswet = data_bitswet.loc[null_values == 0]
    #Drop bitswet including null_rows
    bitswet_1 = bitswet.drop(index=null_rows._stat_axis.values.tolist())
    
    data_bitswet.to_csv('./data/data_calibration/calibration/last/bitswet_descriptors.csv')
    bitswet_1.to_csv('./data/data_calibration/calibration/last/bitswet_all.csv')
    ###############################################
    ###############################################
    ###############################################
