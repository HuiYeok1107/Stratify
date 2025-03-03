from args import args_parser
global args
args = args_parser()

import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import uvicorn
from fastapi import FastAPI, Response,  File, UploadFile
import psutil

import tenseal as ts
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import pickle
from collections import Counter
from itertools import islice
import signal
import logging 
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data import dataset_train_test, dataset_transform, datasets_labels_count
from utils.nonIIDPartition import *
import utils.model
from utils.model import batch_learning_model, setBatchSize
from utils.model_metrics import model_performance

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TBB_NUM_THREADS"] = "1" 
torch.set_num_threads(1) 

local_model = batch_learning_model[args.dataset]# initialise local model
criterion = nn.CrossEntropyLoss(reduction='sum')
device = torch.device('cuda')

context = None
PlaceholderMaptoRealTarget = None 
batchSize = 0
train_losses = 0
train_labels = []
train_preds = []


def create_fastapi_app(base_port, rank, port, clientTrainData, clientTestData, clientTrainLabelDataCount):
    app = FastAPI() # Create a FastAPI app for each client process
    trainDataByLabels = {} # store client data generator by label
    testData = clientTestData # verify if assigning this is needed

    @app.post('/generateEncryptContext') 
    async def generate_EncryptContext(clients: UploadFile = File(...)):
        global context 

        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=2)
        context.generate_galois_keys()
        context.global_scale = 2**40
        
        clientsSerialzContext = context.serialize(save_secret_key=True) # all clients need to use the same context for the server to perform operations on compatible encrypted contents
        serverSerialzContext = context.serialize(save_secret_key=True) # context without secret key so that the server will not be able to decrypt the clients encrypted contents but with the context to perform operations on the encrypted contents

        clientsAddrs = pickle.loads(await clients.read())
        with ThreadPoolExecutor() as executor: 
            futures = [executor.submit(lambda client: requests.post(f'http://127.0.0.1:{base_port - 1 + int(client)}/receiveEncryptContext', files={"serialized_client_context": clientsSerialzContext}), client) for client in clientsAddrs] ##

            for future in as_completed(futures):
                response = future.result()

        return Response(serverSerialzContext, media_type='application/octet-stream')
    

    @app.post('/receiveEncryptContext') 
    async def receive_EncryptionContext(serialized_client_context: UploadFile = File(...)):
        global context
        context = ts.context_from(await serialized_client_context.read(), n_threads=2)
        return {"message": "received context"}
    

    @app.post('/decryptIntermediateComparisonResult') 
    async def decryptComparisonResult(enc_comparison_val: UploadFile = File(...), mapping_stage: UploadFile = File(...)):
        global context
        placeh_mapping_stage = await mapping_stage.read()
        # set comparison value as 0 or 1 during placeholder mapping stage to avoid malicious server from using the minus result to infer the real label a client holds
        if placeh_mapping_stage.decode() == 'True':
            intermediateRes = ts.ckks_vector_from(context, pickle.loads(await enc_comparison_val.read())).decrypt() 
            if abs(intermediateRes[0]) < 1e-5: 
                intermediateRes = 0
            else:
                intermediateRes = 1
        else:
            intermediateRes = pickle.loads(await enc_comparison_val.read())
            for p, noisyEncValue in intermediateRes.items():
                intermediateRes[p] = ts.ckks_vector_from(context, noisyEncValue).decrypt()
        return Response(pickle.dumps(intermediateRes), media_type='application/octet-stream')


    @app.post('/encryptLabels') 
    async def encryptLabels():
        global context
        enc_info = []
        for label, amount in clientTrainLabelDataCount.items():
            enc_info.append([ts.ckks_vector(context, [label]).serialize(), ts.ckks_vector(context, [amount]).serialize()])
        
        return Response(pickle.dumps(enc_info), media_type='application/octet-stream')


    @app.post('/placeholderToRealLabelMapping') 
    async def placeholderToRealLabelMapping(mapping: UploadFile = File(...)):
        global context, PlaceholderMaptoRealTarget
        serialized_mapping = pickle.loads(await mapping.read())
        PlaceholderMaptoRealTarget = {placeholder: round(ts.ckks_vector_from(context, encRealLabel).decrypt()[0]) for placeholder, encRealLabel in serialized_mapping.items()}
        # print(f'port {port} placeholderMapReal: {PlaceholderMaptoRealTarget}')
        return {'message': "received placeholder to real label maps"}
    

    @app.post('/prepareTrainData')
    async def prepare_trainData():
        clientData_copy = clientTrainData.copy()
        clientData_copy['image'] = clientData_copy['image'].apply(lambda img: dataset_transform[args.dataset](img, train=True, augment=True if args.augmentation == 1 else False))
        if args.dataset in ['mnist', 'cifar10', 'cifar100', 'tinyimagenet', 'pacs', 'digitdg']:
            for label in clientData_copy['label'].unique():
                trainDataByLabels[label] = iter(clientData_copy.loc[clientData_copy['label'] == label, ['image', 'label']].sample(frac=1, replace=False).itertuples(index=False, name=None))
        else:
            labels = clientData_copy['label'].unique()
            for label in labels:
                rows = clientData_copy.loc[clientData_copy['label'] == label].drop(columns=['label']).values
                trainDataByLabels[label] = iter([(torch.tensor(row), torch.tensor(label)) for row in rows])
        
        return {"message": f"Train data is ready by process {rank}"}


    def prepare_testData():
        testData['image'] = testData['image'].apply(lambda img: dataset_transform[args.dataset](img, train=False, augment=False))


    @app.post('/test')
    async def test():
        global local_model, train_losses, train_labels, train_preds
        test_losses = 0
        test_labels, test_preds = [], []

        glbModelParam = pickle.loads(requests.get(f'http://127.0.0.1:{base_port - 1}/get_glb_params').content)
        local_model.load_state_dict(glbModelParam)
        local_model.eval()

        if args.dataset in ['mnist', 'cifar10', 'cifar100', 'tinyimagenet', 'pacs', 'digitdg']:
            inputs = torch.stack(testData['image'].tolist()).float()
            labels = torch.tensor(testData['label'].values)
        else:
            inputs = torch.from_numpy(testData.drop(columns=['labels']).values).float()
            labels = torch.from_numpy(testData['labels'].values).long()

        test_loader = DataLoader(TensorDataset(inputs, labels), batch_size=128, shuffle=False)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = local_model(inputs)

                _, predicted = preds.max(1)

                test_losses += criterion(preds, labels).item()
                test_labels.extend(labels.tolist())
                test_preds.extend(predicted.tolist())
        
        conf_matrix, normal_accuracy, macro_avg_f1, weighted_f1, macro_avg_ba_allclasses, weighted_ba, total_train_size = model_performance(train_labels, train_preds, datasets_labels_count[args.dataset])    
        test_conf_matrix, test_normal_accuracy, test_macro_avg_f1, test_weighted_f1, test_macro_avg_ba_allclasses, test_weighted_ba, total_test_size = model_performance(test_labels, test_preds, datasets_labels_count[args.dataset])  
        
        response_json = pickle.dumps({f"client {rank}": {
                                    "train_conf_matrix": conf_matrix,
                                    "train_normal_accuracy": normal_accuracy, 
                                    "train_macro_avg_f1": macro_avg_f1, "train_weighted_f1": weighted_f1, 
                                    "train_macro_avg_ba_allclasses": macro_avg_ba_allclasses, "train_weighted_ba": weighted_ba,
                                    "Train summed loss": train_losses,
                                    "train_size": total_train_size,
                                    "test_conf_matrix": test_conf_matrix,
                                    "test_normal_accuracy": test_normal_accuracy, 
                                    "test_macro_avg_f1": test_macro_avg_f1, "test_weighted_f1": test_weighted_f1, 
                                    "test_macro_avg_ba_allclasses": test_macro_avg_ba_allclasses, "test_weighted_ba": test_weighted_ba,
                                    "Test summed loss": test_losses,
                                    "test_size": total_test_size
                                }})
        
        train_losses = 0
        train_labels, train_preds = [], []
        
        return Response(response_json, media_type='application/octet-stream')
                                   
                                  
    @app.post('/train')
    async def train(to_train: UploadFile = File(...), glb_model_params: UploadFile = File(...), batch_size: UploadFile = File(...)):
        global local_model, criterion, batchSize, PlaceholderMaptoRealTarget, train_losses, train_labels, train_preds

        # Extract placeholders to train and global model parameter from server
        serializ_to_train = await to_train.read()
        to_train = pickle.loads(serializ_to_train)
        
        serializ_glb_model_params = await glb_model_params.read()
        glb_model_params = pickle.loads(serializ_glb_model_params)

        batchSize = pickle.loads(await batch_size.read())
        setBatchSize(batchSize) # set the batch size in client's model architecture file for custom batch normalisation usage

        # forward and backward pass to accumulate local gradients summed
        local_model.load_state_dict(glb_model_params)
        local_model = local_model.to(device)
        local_model.train()
        
        # map placeholder to real label and get the label data to train
        samples = []
        for p, count in Counter(to_train).items():
            realLabel = PlaceholderMaptoRealTarget[p] 
            try:
                samples.extend(list(islice(trainDataByLabels[realLabel], count))) 
            except Exception as e:
                print('exception')
                print(p, count)
                print(port)
        inputs = torch.stack(list(zip(*samples))[0]).to(device).float()
        labels = torch.tensor(list(zip(*samples))[1], dtype=torch.long).to(device)
        
        outputs = local_model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)

        train_labels.extend(list(zip(*samples))[1])
        train_preds.extend(predicted.tolist())
        train_losses += loss.item()

        # get the grad of summed loss
        grad = torch.autograd.grad(loss, local_model.parameters(), retain_graph=False)
        serialz_grad = pickle.dumps(grad)

        # return summed gradients to server
        return Response(serialz_grad, media_type='application/octet-stream')
    
    
    @app.post('/federatedLearningCompleted')
    def federatedLearningCompleted():
        os.kill(os.getpid(), signal.SIGTERM)
        return 'client connection closed.'   
    
    if args.dataset in ['mnist', 'cifar10', 'cifar100', 'tinyimagenet', 'pacs', 'digitdg']:
        prepare_testData()

    uvicorn.run(app, port=port)  # Start the app on the assigned port


def start_training_process(base_port, rank, port, clientTrainData, clientTestData, clientTrainLabelDataCount):
    print(f"port: {port}, unique labels: {clientTrainData['label'].unique()}, {clientTrainLabelDataCount}")
    print(f"Process {rank} started. Running Flask app on port {port}")
    print(f'process id: {os.getpid()}, {os.getppid()}')
    # set_cpu_affinity(cpu_aff_c)
    pid = os.getpid()
    process = psutil.Process(pid)
    current_affinity = process.cpu_affinity()
    process.cpu_affinity(current_affinity[rank:rank+1]) # [rank:rank+1] in hpc

    print(os.system('taskset -cp %s' %os.getpid()))
    create_fastapi_app(base_port, rank, port, clientTrainData, clientTestData, clientTrainLabelDataCount)


if __name__ == "__main__":
    total_clients = args.client_num # total clients
    base_port = args.start_port + 1 # client starting port (server port + 1)
    labelOrDomain_per_client = args.labelOrDomainPerClientHold
    dataset = args.dataset
    
    train_df, test_df, total_labels = dataset_train_test[dataset]() 
    
    if args.dataset in ['pacs', 'digitdg']:
        total_domains = 4
        if labelOrDomain_per_client != 0:
            clientsNonIIDDomains =  assignClientDomain(total_domains, labelOrDomain_per_client, total_clients, total_labels)
            clients_traindf, clientsTrainLabelDataCount = domainHold_nonIID_partition(train_df, clientsNonIIDDomains)
            clients_testdf, clientsTestDataCount = domainHold_nonIID_partition(test_df, clientsNonIIDDomains)
        else:
            dirProp = []
            for i in range(total_domains):    
                dirProp.append(np.random.dirichlet([0.5] * total_clients))

            clients_traindf, clientsTrainLabelDataCount = dirichlet_nonIID_domain_partition(train_df, dirProp, num_clients=total_clients)
            clients_testdf, clientsTestDataCount = dirichlet_nonIID_domain_partition(test_df, dirProp, num_clients=total_clients)
                
    else:
        if labelOrDomain_per_client != 0:
            clientsNonIIDLabels = assignClientLabel(total_labels, labelOrDomain_per_client, total_clients)
            clients_traindf, clientsTrainLabelDataCount = classHold_nonIID_partition(train_df, clientsNonIIDLabels)
            clients_testdf, clientsTestDataCount = classHold_nonIID_partition(test_df, clientsNonIIDLabels)
        else:
            dirProp = []
            for i in range(total_labels):    
                dirProp.append(np.random.dirichlet([0.5] * total_clients))
            
            clients_traindf, clientsTrainLabelDataCount = dirichlet_nonIID_label_partition(train_df, dirProp, num_clients=total_clients)
            clients_testdf, clientsTestDataCount = dirichlet_nonIID_label_partition(test_df, dirProp, num_clients=total_clients)
        
    print(clientsTrainLabelDataCount)
    print(clientsTestDataCount)
    mp.set_start_method('spawn', force=True)    

    # Spawn multiple processes, each with its own Flask app
    processes = []
    for rank in range(total_clients):
        port = base_port + rank
        p = mp.Process(target=start_training_process, args=(base_port, rank, port, clients_traindf[rank], clients_testdf[rank], clientsTrainLabelDataCount[rank]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()




