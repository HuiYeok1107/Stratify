import tensorflow_datasets as tfds
from datasets import load_dataset
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import numpy as np
import pandas as pd
import gdown
import zipfile
import pickle
import io
import os
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def get_mnist_df():
    data = tfds.load('mnist', split='train')
    traindf = tfds.as_dataframe(data)
    
    data = tfds.load('mnist', split='test')
    testdf = tfds.as_dataframe(data)
    
    return traindf, testdf, 10

def transform_mnist(img, train=False, augment=False):
    transform_list = [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
    if augment:
        transform_list.insert(1, transforms.RandomHorizontalFlip())
    
    transform = transforms.Compose(transform_list)    
    transformed_img = transform(img)
    return transformed_img
    
    
def get_cifar10_df():
    data = tfds.load('cifar10', split='train')
    traindf = tfds.as_dataframe(data)
    traindf = traindf.sample(frac=0.1)
    data = tfds.load('cifar10', split='test')
    testdf = tfds.as_dataframe(data)
    
    return traindf, testdf, 10

def transform_cifar10(img, train=False, augment=False):
    transform_list = [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    if augment:
        transform_list.insert(1, transforms.RandomCrop(32, padding=4, padding_mode='reflect'))
        transform_list.insert(1, transforms.RandomHorizontalFlip())
    
    transform = transforms.Compose(transform_list)    
    transformed_img = transform(img)
    return transformed_img
    

def get_cifar100_df():
    data = tfds.load('cifar100', split='train')
    traindf = tfds.as_dataframe(data)
    
    data = tfds.load('cifar100', split='test')
    testdf = tfds.as_dataframe(data)
    
    return traindf, testdf, 100

def transform_cifar100(img, train=False, augment=False):
    transform_list = [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    if augment:
        transform_list.insert(1, transforms.RandomCrop(32, padding=4, padding_mode='reflect'))
        transform_list.insert(1, transforms.RandomHorizontalFlip())
    
    transform = transforms.Compose(transform_list)    
    transformed_img = transform(img)
    return transformed_img
    
    
def get_tinyImageNet_df():
    ds = load_dataset("zh-plus/tiny-imagenet", trust_remote_code=True) #cache_dir="/scr/user/huiyeok/datasets"
    traindf = ds['train'].to_pandas()
    testdf = ds['valid'].to_pandas()
    
    return traindf, testdf, 200

def transform_tinyImageNet(img, train=False, augment=False):
    img = Image.open(io.BytesIO(img['bytes']))
    if img.mode != 'RGB':
        img = img.convert('RGB') 
        
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    if augment:
        transform_list.insert(0, transforms.RandomCrop(64, padding=4, padding_mode='reflect'))
        transform_list.insert(0, transforms.RandomHorizontalFlip())
        transform_list.insert(0, transforms.AugMix())
    else:
        transform_list.insert(0, transforms.Resize(64))
        transform_list.insert(0, transforms.CenterCrop(64))
        
    transform = transforms.Compose(transform_list)    
    transformed_img = transform(img)
    return transformed_img
    
    
def get_covtype_df():
    testDataFrac = 0.2
    traindf = pd.read_csv("./dataset/covtype.csv")
    traindf = traindf.loc[traindf['Cover_Type'].isin([1, 2]), :]
    label_mapping = {1: 0, 2: 1}
    traindf['Cover_Type'] = traindf['Cover_Type'].map(label_mapping)
    traindf = traindf.rename(columns={'Cover_Type': "labels"})
    
    testdf = traindf.sample(frac=testDataFrac, replace=False, random_state=1)
    traindf = traindf.drop(testdf.index)
    
    return traindf, testdf, 2

    
def get_pacs_df():
    import deeplake
    if os.path.exists(f"Fed-GT/utils/dataset/pacsTrain.pkl") and os.path.exists("Fed-GT/utils/dataset/pacsTest.pkl") :
        traindf = pd.read_pickle("Fed-GT/utils/dataset/pacsTrain.pkl")
        testdf = pd.read_pickle("Fed-GT/utils/dataset/pacsTest.pkl")
    else:
        dataset = deeplake.load("hub://activeloop/pacs-train")
        dataloader = dataset.tensorflow()
        traindf = pd.DataFrame(list(dataloader.as_numpy_iterator()))
        traindf.to_pickle("Fed-GT/utils/dataset/pacsTrain.pkl")

        dataset = deeplake.load("hub://activeloop/pacs-test")
        dataloader = dataset.tensorflow()
        testdf = pd.DataFrame(list(dataloader.as_numpy_iterator()))
        testdf.to_pickle("Fed-GT/utils/dataset/pacsTest.pkl")
        del dataset, dataloader
    
    traindf = traindf.rename(columns={"images": "image", "labels": "label", "domains": "domain"})
    testdf = testdf.rename(columns={"images": "image", "labels": "label", "domains": "domain"})
    
    traindf['label'] = traindf['label'].apply(lambda x: x[0])
    traindf['domain'] = traindf['domain'].apply(lambda x: x[0])
    testdf['label'] = testdf['label'].apply(lambda x: x[0])
    testdf['domain'] = testdf['domain'].apply(lambda x: x[0])
    
    return traindf, testdf, 7
    
def transform_pacs(img, train=False, augment=False):
    transform_list = [
        transforms.ToPILImage(),
        transforms.ToTensor()
    ]
    if augment:
        transform_list.insert(1, transforms.RandomHorizontalFlip())
    
    transform = transforms.Compose(transform_list)    
    transformed_img = transform(img)
    return transformed_img
    
    
def get_digitDG_df():
    if not os.path.exists(f"Fed-GT/utils/dataset/digits_dg"):
        gdown.download("https://drive.google.com/uc?id=15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7", "digits_dg.zip", quiet=False)
        with zipfile.ZipFile('digits_dg.zip', "r") as zip_ref:
            zip_ref.extractall("Fed-GT/utils/dataset")
    
    data_dir = 'Fed-GT/utils/dataset/digits_dg'
    # Load the dataset
    mnist_dataset = datasets.ImageFolder(root=f'{data_dir}/mnist/train')
    svhn_dataset = datasets.ImageFolder(root=f'{data_dir}/svhn/train')
    mnistM_dataset = datasets.ImageFolder(root=f'{data_dir}/mnist_m/train')
    syn_dataset = datasets.ImageFolder(root=f'{data_dir}/syn/train')
    
    
    mnist_testdataset = datasets.ImageFolder(root=f'{data_dir}/mnist/val')
    svhn_testdataset = datasets.ImageFolder(root=f'{data_dir}/svhn/val')
    mnistM_testdataset = datasets.ImageFolder(root=f'{data_dir}/mnist_m/val')
    syn_testdataset = datasets.ImageFolder(root=f'{data_dir}/syn/val')
    
    trainDatasets_list = [
        (0, mnist_dataset),
        (1, svhn_dataset),
        (2, mnistM_dataset),
        (3, syn_dataset),
    ]
    
    testDatasets_list = [
        (0, mnist_testdataset),
        (1, svhn_testdataset),
        (2, mnistM_testdataset),
        (3, syn_testdataset),
    ]
    
    def to_dataframe(datasetList):
        records = []
        for dataset_name, dataset in datasetList:
            for idx, (image, label) in enumerate(dataset):
                image_data = np.array(image)  # Convert the PIL image to a NumPy array
                records.append({
                    'image': image_data,  # You can store the raw pixel values here
                    'label': label,       # Store the corresponding label
                    'domain': dataset_name  # The class name corresponding to the label
                })
        return records
    
    # Create a DataFrame
    traindf = pd.DataFrame(to_dataframe(trainDatasets_list))
    testdf = pd.DataFrame(to_dataframe(testDatasets_list))
    
    return traindf, testdf, 10
    
def transform_digitDG(img, train=False, augment=False):
    img = Image.fromarray(img)
    transform_list = [
        transforms.Resize((28, 28)),  
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor()
    ]
    if augment:
        transform_list.insert(1, transforms.RandomHorizontalFlip())
    
    transform = transforms.Compose(transform_list)    
    transformed_img = transform(img)
    return transformed_img
        


datasets_labels_count = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "tinyimagenet": 200,
    "covtype": 2,
    "pacs": 7,
    "digitdg": 10
}

dataset_train_test = {
    "mnist": get_mnist_df,
    "cifar10": get_cifar10_df,
    "cifar100": get_cifar100_df,
    "tinyimagenet": get_tinyImageNet_df,
    "covtype": get_covtype_df,
    "pacs": get_pacs_df,
    "digitdg": get_digitDG_df

}

dataset_transform = {
    "mnist": transform_mnist,
    "cifar10": transform_cifar10,
    "cifar100": transform_cifar100,
    "tinyimagenet": transform_tinyImageNet,
    "pacs": transform_pacs,
    "digitdg": transform_digitDG
}




















    
    
    