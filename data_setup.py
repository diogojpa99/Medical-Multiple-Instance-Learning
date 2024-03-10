import breast_scripts.data_setup as breast_data_setup
import skin_scripts.data_setup as skin_data_setup

datasets=['ISIC2019-Clean', 'PH2', 'Derm7pt','DDSM+CBIS+MIAS_CLAHE-Binary-Mass_vs_Normal', 
          'DDSM+CBIS+MIAS_CLAHE-Binary-Benign_vs_Malignant', 'DDSM+CBIS+MIAS_CLAHE', 'DDSM+CBIS+MIAS_CLAHE-v2', 'INbreast',
          'MIAS_CLAHE', 'MIAS_CLAHE-Mass_vs_Normal', 'MIAS_CLAHE-Benign_vs_Malignant',
          'DDSM', 'DDSM-Mass_vs_Normal', 'DDSM-Benign_vs_Malignant', 
          'DDSM+CBIS-Mass_vs_Normal',
          'CBIS', 'CBIS-Processed_CLAHE']

def Build_Dataset(data_path, input_size, args):
    
    if args.dataset in datasets:
        if args.dataset_type == 'Skin':
            return skin_data_setup.Build_Dataset(True, data_path, args), skin_data_setup.Build_Dataset(False, data_path, args)
        elif args.dataset_type == 'Breast':
            return breast_data_setup.Build_Datasets(data_path, input_size, args)
    else:
        ValueError('Invalid dataset. Please choose from the following datasets: ISIC2019-Clean, PH2, Derm7pt, DDSM+CBIS+MIAS_CLAHE-Binary, DDSM+CBIS+MIAS_CLAHE, INbreast, \
                   MIAS_CLAHE, MIAS_CLAHE-Mass_vs_Normal, MIAS_CLAHE-Benign_vs_Malignant, DDSM, DDSM-Mass_vs_Normal, DDSM-Benign_vs_Malignant, DDSM+CBIS-Mass_vs_Normal')