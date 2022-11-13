import pandas as pd
import shutil
import os


def sort_images_cropped(image_path, csv_path, train_image_path, test_image_path,):
# if folder does not exist, create it
    IMAGE_PATH =  image_path #'data/img'
    IMAGE_TYPE = 'cropped image file path'
    TRAIN_IMAGE_PATH = train_image_path #'data/train'
    TEST_IMAGE_PATH = test_image_path #'data/test'
    CSV_PATH = csv_path #'data/train.csv'

    if not os.path.exists(TRAIN_IMAGE_PATH):
        os.makedirs(TRAIN_IMAGE_PATH)
        # if BENIGN and MALIGNANT folders do not exist, create them
        if not os.path.exists(TRAIN_IMAGE_PATH + '/BENIGN'):
            os.makedirs(TRAIN_IMAGE_PATH + '/BENIGN')
        if not os.path.exists(TRAIN_IMAGE_PATH + '/MALIGNANT'):
            os.makedirs(TRAIN_IMAGE_PATH + '/MALIGNANT')
    if not os.path.exists(TEST_IMAGE_PATH):
        os.makedirs(TEST_IMAGE_PATH)
        # if BENIGN and MALIGNANT folders do not exist, create them
        if not os.path.exists(TEST_IMAGE_PATH + '/BENIGN'):
            os.makedirs(TEST_IMAGE_PATH + '/BENIGN')
        if not os.path.exists(TEST_IMAGE_PATH + '/MALIGNANT'):
            os.makedirs(TEST_IMAGE_PATH + '/MALIGNANT')
            

    # read the csv files in /archive/csv to dataframe
    mass_train_df = pd.read_csv(f'{CSV_PATH}/mass_case_description_train_set.csv')
    mass_test_df = pd.read_csv(f'{CSV_PATH}/mass_case_description_test_set.csv')
    dicom_df = pd.read_csv(f'{CSV_PATH}/dicom_info.csv')

    # get pathology and image file path in new dataframe
    mass_images_train = mass_train_df[['pathology', IMAGE_TYPE]]
    mass_images_test = mass_test_df[['pathology', IMAGE_TYPE]]

    full_mammogram_images=dicom_df[dicom_df.SeriesDescription == 'cropped images']

    for index, row in mass_images_train.iterrows():
        row[IMAGE_TYPE] = row[IMAGE_TYPE].split('/')[2]
        # add path to start of image file path
        row[IMAGE_TYPE] = IMAGE_PATH + '/' + row[IMAGE_TYPE]

    for index, row in mass_images_test.iterrows():
        row[IMAGE_TYPE] = row[IMAGE_TYPE].split('/')[2]
        # add path to start of image file path
        row[IMAGE_TYPE] = IMAGE_PATH + '/' + row[IMAGE_TYPE]

    mass_full_train = full_mammogram_images[full_mammogram_images.PatientID.isin(mass_train_df[IMAGE_TYPE].apply(lambda x: x.split('/')[0]))]
    mass_full_test = full_mammogram_images[full_mammogram_images.PatientID.isin(mass_test_df[IMAGE_TYPE].apply(lambda x: x.split('/')[0]))]


    full_training_images_path = mass_full_train.image_path.apply(lambda x: x.replace('CBIS-DDSM/jpeg', IMAGE_PATH))
    full_testing_images_path = mass_full_test.image_path.apply(lambda x: x.replace('CBIS-DDSM/jpeg', IMAGE_PATH))

    # for each row in full_images_path, if the image file path is in the mass_images_train dataframe, add the pathology to the new dataframe
    full_training_images_pathology = pd.DataFrame(columns=['pathology', IMAGE_TYPE])
    for index, row in full_training_images_path.iteritems():
        check = row.split('/')[:-1]
        check = '/'.join(check)
        if check in mass_images_train[IMAGE_TYPE].values:
            pathology = mass_images_train[mass_images_train[IMAGE_TYPE] == check]['pathology'].values[0]
            full_training_images_pathology = pd.concat([full_training_images_pathology, pd.DataFrame([[pathology, row]], columns=['pathology', IMAGE_TYPE])], ignore_index=True)

    full_testing_images_pathology = pd.DataFrame(columns=['pathology', IMAGE_TYPE])
    for index, row in full_testing_images_path.iteritems():
        check = row.split('/')[:-1]
        check = '/'.join(check)
        if check in mass_images_test[IMAGE_TYPE].values:
            pathology = mass_images_test[mass_images_test[IMAGE_TYPE] == check]['pathology'].values[0]
            full_testing_images_pathology = pd.concat([full_testing_images_pathology, pd.DataFrame([[pathology, row]], columns=['pathology', IMAGE_TYPE])], ignore_index=True)



    for index, row in full_training_images_pathology.iterrows():

        if row['pathology'] == 'BENIGN':
            shutil.copy(row[IMAGE_TYPE], f'{TRAIN_IMAGE_PATH}/BENIGN/{index}.jpg')
        elif row['pathology'] == 'MALIGNANT':
            shutil.copy(row[IMAGE_TYPE], f'{TRAIN_IMAGE_PATH}/MALIGNANT/{index}.jpg')
        elif row['pathology'] == 'BENIGN_WITHOUT_CALLBACK':
            shutil.copy(row[IMAGE_TYPE], f'{TRAIN_IMAGE_PATH}/BENIGN/{index}.jpg')
        elif row['pathology'] == 'MALIGNANT_WITHOUT_CALLBACK':
            shutil.copy(row[IMAGE_TYPE], f'{TRAIN_IMAGE_PATH}/MALIGNANT/{index}.jpg')
        elif row['pathology'] == 'BENIGN_WITH_CALLBACK':
            shutil.copy(row[IMAGE_TYPE], f'{TRAIN_IMAGE_PATH}/BENIGN/{index}.jpg')
        elif row['pathology'] == 'MALIGNANT_WITH_CALLBACK':
            shutil.copy(row[IMAGE_TYPE], f'{TRAIN_IMAGE_PATH}/MALIGNANT/{index}.jpg')

    for index, row in full_testing_images_pathology.iterrows():

        if row['pathology'] == 'BENIGN':
            shutil.copy(row[IMAGE_TYPE], f'{TEST_IMAGE_PATH}/BENIGN/{index}.jpg')
        elif row['pathology'] == 'MALIGNANT':
            shutil.copy(row[IMAGE_TYPE], f'{TEST_IMAGE_PATH}/MALIGNANT/{index}.jpg')
        elif row['pathology'] == 'BENIGN_WITHOUT_CALLBACK':
            shutil.copy(row[IMAGE_TYPE], f'{TEST_IMAGE_PATH}/BENIGN/{index}.jpg')
        elif row['pathology'] == 'MALIGNANT_WITHOUT_CALLBACK':
            shutil.copy(row[IMAGE_TYPE], f'{TEST_IMAGE_PATH}/MALIGNANT/{index}.jpg')
        elif row['pathology'] == 'BENIGN_WITH_CALLBACK':
            shutil.copy(row[IMAGE_TYPE], f'{TEST_IMAGE_PATH}/BENIGN/{index}.jpg')
        elif row['pathology'] == 'MALIGNANT_WITH_CALLBACK':
            shutil.copy(row[IMAGE_TYPE], f'{TEST_IMAGE_PATH}/MALIGNANT/{index}.jpg')