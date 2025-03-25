from args import args_parser
global args
args = args_parser()

from fastapi import FastAPI, Response, File, UploadFile, Request
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import uvicorn
import httpx
import psutil 
import io

import os 
from itertools import islice
import pickle
import signal
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data import dataset_train_test, dataset_transform, datasets_labels_count
from utils.nonIID_partition import *
import utils.model
from utils.model import single_sample_learning_model
from utils.model_metrics import model_performance

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tenseal as ts 
context = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = single_sample_learning_model[args.dataset]
lr = args.lr
weight_decay = args.weight_decay
momentum = args.momentum
eps = args.eps

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)  
else:
    # log error ask user to add in other optimizer in code if required
    pass
criterion = nn.CrossEntropyLoss()     

train_losses, train_labels, train_preds = [], [], []
PlaceholderMaptoRealTarget = None


def create_fastapi_app(base_port, rank, port, clientTrainData, clientTestData, clientTrainLabelDataCount):
    app = FastAPI() 
    trainDataByLabels = {}
    testData = clientTestData

    @app.post('/federatedLearningCompleted')
    def federatedLearningCompleted():
        os.kill(os.getpid(), signal.SIGINT)


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

    if args.dataset in ['mnist', 'cifar10', 'cifar100', 'tinyimagenet', 'pacs', 'digitdg']:
        prepare_testData()


    @app.post('/generateEncryptContext') 
    async def generate_EncryptContext(clients: UploadFile = File(...)):
        global context 

        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=2)
        context.generate_galois_keys()
        context.global_scale = 2**40
        
        clientsSerialzContext = context.serialize(save_secret_key=True) # all clients need to use the same context for the server to perform operations on compatible encrypted contents
        serverSerialzContext = context.serialize(save_secret_key=False) # context without secret key so that the server will not be able to decrypt the clients encrypted contents but with the context to perform operations on the encrypted contents

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


    @app.post('/encryptLabels') 
    async def encryptLabels():
        global context
        enc_info = []
        for label, amount in clientTrainLabelDataCount.items():
            enc_info.append([ts.ckks_vector(context, [label]).serialize(), ts.ckks_vector(context, [amount]).serialize()])
        
        return Response(pickle.dumps(enc_info), media_type='application/octet-stream')
    

    @app.post('/decryptIntermediateComparisonResult') 
    async def decryptComparisonResult(enc_comparison_val: UploadFile = File(...), mapping_stage: UploadFile = File(...)):
        global context
        placeh_mapping_stage = await mapping_stage.read()
        # set comparison value as 0 or 1 during placeholder mapping stage to avoid malicious server from using the minus result to infer the real label a client holds
        if placeh_mapping_stage.decode() == 'True':
            intermediateCompVals = pickle.loads(await enc_comparison_val.read())
            if isinstance(intermediateCompVals, list):
                intermediateRes = []
                intermediateRes = [0 if abs(ts.ckks_vector_from(context, intermediateCompVal).decrypt()[0]) < 1e-5 else 1 for intermediateCompVal in intermediateCompVals]
            else:
                intermediateRes = {}
                for p, compVal in intermediateCompVals.items():
                    intermediateRes[p] = 0 if abs(ts.ckks_vector_from(context, compVal).decrypt()[0]) < 1e-5 else 1
        else:
            intermediateRes = pickle.loads(await enc_comparison_val.read())
            for p, noisyEncValue in intermediateRes.items():
                intermediateRes[p] = ts.ckks_vector_from(context, noisyEncValue).decrypt()
                
        return Response(pickle.dumps(intermediateRes), media_type='application/octet-stream')


    @app.post('/placeholderToRealLabelMapping') 
    async def placeholderToRealLabelMapping(mapping: UploadFile = File(...)):
        global context, PlaceholderMaptoRealTarget
        serialized_mapping = pickle.loads(await mapping.read())
        PlaceholderMaptoRealTarget = {placeholder: round(ts.ckks_vector_from(context, encRealLabel).decrypt()[0]) for placeholder, encRealLabel in serialized_mapping.items()}
        print(f'port {port} placeholderMapReal: {PlaceholderMaptoRealTarget}')
        return {'message': "received placeholder to real label maps"}


    @app.post('/currentGlobalModelParams')
    async def currentGlobalModelParams(request: Request):
        global model
        serializ_glb_model_params = await request.body()
        buffer = io.BytesIO(serializ_glb_model_params)
        model.load_state_dict(torch.load(buffer))
        return {"message": "Model updated with latest global parameters."}


    @app.post('/test')
    async def test():
        # global model
        global model, train_losses, train_labels, train_preds

        glbModelParam = pickle.loads(requests.get(f'http://127.0.0.1:{base_port - 1}/get_glb_params').content)
        model.load_state_dict(glbModelParam)
        model.eval()
        
        test_losses, test_labels, test_preds = [], [], []
        correct = 0

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
                preds = model(inputs)

                _, predicted = preds.max(1)
                correct += predicted.eq(labels).sum().item()
                test_losses.append(criterion(preds, labels).item())
                test_labels.append(labels.tolist())
                test_preds.append(predicted.tolist())
                
        
        conf_matrix, normal_accuracy, macro_avg_f1, weighted_f1, macro_avg_ba_allclasses, weighted_ba, total_train_size = model_performance(train_labels, train_preds, datasets_labels_count[args.dataset])    
        test_conf_matrix, test_normal_accuracy, test_macro_avg_f1, test_weighted_f1, test_macro_avg_ba_allclasses, test_weighted_ba, total_test_size = model_performance(test_labels, test_preds, datasets_labels_count[args.dataset])  
        
        response_json = pickle.dumps({f"client {rank}": {
                                    "train_conf_matrix": conf_matrix,
                                    "train_normal_accuracy": normal_accuracy, 
                                    "train_macro_avg_f1": macro_avg_f1, "train_weighted_f1": weighted_f1, 
                                    "train_macro_avg_ba_allclasses": macro_avg_ba_allclasses, "train_weighted_ba": weighted_ba,
                                    "Train avg loss": np.mean(train_losses),
                                    "train_size": total_train_size,
                                    "test_conf_matrix": test_conf_matrix,
                                    "test_normal_accuracy": test_normal_accuracy, 
                                    "test_macro_avg_f1": test_macro_avg_f1, "test_weighted_f1": test_weighted_f1, 
                                    "test_macro_avg_ba_allclasses": test_macro_avg_ba_allclasses, "test_weighted_ba": test_weighted_ba,
                                    "Test avg loss": np.mean(test_losses),
                                    "test_size": total_test_size
                                }})
        
        train_losses, train_labels, train_preds = [], [], []
        
        return Response(response_json, media_type='application/octet-stream')


    @app.post('/train')
    async def train(to_train: UploadFile = File(...), next_client: UploadFile = File(...)):
        global model, criterion, epoch_total_correct, epoch_total_samples, PlaceholderMaptoRealTarget, train_losses, train_labels, train_preds
        model.train().to(device)
        target_exhausted = []
        target_noTrain = []

        # receive target to train and next assigned client address from server
        to_train = pickle.loads(await to_train.read())
        nextClient = pickle.loads(await next_client.read())
        # train model on the target data required by the server in sequence 
        for targetPlaceholder in to_train:
            targetRealLabel = PlaceholderMaptoRealTarget[targetPlaceholder]
            sample = list(islice(trainDataByLabels[targetRealLabel], 1))
            if sample:
                x, y = sample[0]
                optimizer.zero_grad()
                output = model(x.unsqueeze(0).float().to(device))
                loss = criterion(output, torch.tensor([y]).to(device))
                loss.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args.grad_clip)

                optimizer.step()

                _, predicted = torch.max(output, 1)
                train_labels.append(y)
                train_preds.append(predicted.item())
                train_losses.append(loss.item())


            else:
                target_exhausted.append(targetPlaceholder)
                target_noTrain.append(targetPlaceholder) 
        
        # send updated model param to next assigned client
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer, _use_new_zipfile_serialization=False)
        buffer.seek(0)
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f'http://127.0.0.1:{base_port - 1 + int(nextClient[1:])}/currentGlobalModelParams', 
                headers={"Content-Type": "application/octet-stream"},
                content=buffer.getvalue()
            )
        buffer.close() 
        
        # return target exhausted and target no train list back to server
        return {'target exhausted': list(set(target_exhausted)), 'target no train': target_noTrain}

    # Start the Flask app on the assigned port
    uvicorn.run(app, port=port)  


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
