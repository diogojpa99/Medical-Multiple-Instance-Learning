import breast_scripts.data_setup as breast_data_setup
import skin_scripts.data_setup as skin_data_setup

datasets=['ISIC2019-Clean', 'PH2', 'Derm7pt','DDSM+CBIS+MIAS_CLAHE-Binary-Mass_vs_Normal', 
          'DDSM+CBIS+MIAS_CLAHE-Binary-Benign_vs_Malignant', 'DDSM+CBIS+MIAS_CLAHE', 'DDSM+CBIS+MIAS_CLAHE-v2', 'INbreast',
          'MIAS_CLAHE', 'MIAS_CLAHE-Mass_vs_Normal', 'MIAS_CLAHE-Benign_vs_Malignant',
          'DDSM', 'DDSM-Mass_vs_Normal', 'DDSM-Benign_vs_Malignant', 
          'DDSM+CBIS-Mass_vs_Normal', 'DDSM+CBIS-Benign_vs_Malignant', 'DDSM+CBIS-Benign_vs_Malignant-Processed', 
          'CBIS', 'CBIS-Processed_CLAHE', 'CBIS-DDSM-only_mass', 'CBIS-DDSM',
          'CMMD-only_mass-processed_crop_CLAHE', 'CMMD-only_mass',
          'CMMD-only_mass-processed',
          'CBIS-DDSM-train_val-pad_clahe']

def Build_Dataset(data_path, input_size, args):
    
    if args.dataset_type == 'Skin':
        return skin_data_setup.Build_Dataset(True, data_path, args), skin_data_setup.Build_Dataset(False, data_path, args)
    elif args.dataset_type == 'Breast':
        if args.breast_clahe:
            setup_clahe(args.testset, args)
        if args.finetune or args.train:
            return breast_data_setup.Build_Datasets(data_path, input_size, args)
        else: 
            return breast_data_setup.Get_Testset(data_path, input_size, args), breast_data_setup.Get_Testset(data_path, input_size, args) # We will use the test set as the validation set
    else:
        ValueError('Invalid dataset type. Please choose from the following dataset types: {}'.format(['Skin', 'Breast']))


# def Build_Dataset(data_path, input_size, args):
    
#     if args.dataset in datasets:
#         if args.dataset_type == 'Skin':
#             return skin_data_setup.Build_Dataset(True, data_path, args), skin_data_setup.Build_Dataset(False, data_path, args)
#         elif args.dataset_type == 'Breast':
#             if args.finetune or args.train:
#                 return breast_data_setup.Build_Datasets(data_path, input_size, args)
#             else: 
#                 return breast_data_setup.Get_Testset(data_path, input_size, args), breast_data_setup.Get_Testset(data_path, input_size, args) # We will use the test set as the validation set
#     else:
#         ValueError('Invalid dataset. Please choose from the following datasets: {}'.format(datasets))

def setup_clahe(dataset, args):
    """Sets up the CLAHE parameters for the dataset

    Args:
        dataset (_type_): _description_
        args (_type_): _description_
    """
    if 'CMMD' in dataset:
        args.clahe_clip_limit = 5.0
        
    print('[Info] - CLAHE clip limit: {}'.format(args.clahe_clip_limit))
        