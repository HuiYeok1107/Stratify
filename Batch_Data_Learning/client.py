import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import uvicorn
from fastapi import FastAPI, Response,  File, UploadFile

import tenseal as ts

import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
# from PIL import Image

import numpy as np
import pandas as pd

import pickle
import time
from collections import Counter
from itertools import islice
import signal
# import io
import logging 
import os
import psutil
# from ResNet9_BN2D import *
# import ResNet9_BN2D 
# from ResNet18_BN2D import *
# import ResNet18_BN2D
# from model import *

# # # # # # # # # #
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data import dataset_train_test, dataset_transform
from utils.nonIIDPartition import *
from args import args_parser
global args
args = args_parser()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TBB_NUM_THREADS"] = "1" 
torch.set_num_threads(1) 



context = None
PlaceholderMaptoRealTarget = None ##
# create local model
# model = ResNet9(3, 100)
# model = CovtypeNN()
# model = resnet18
# print(f'model: {model}')
criterion = nn.CrossEntropyLoss(reduction='sum')
device = torch.device('cuda')

batchSize = 0
epoch_total_correct = 0 # to remove
epoch_total_samples = 0 # to remove
train_losses = 0
train_labels = []
train_preds = []



def create_fastapi_app(base_port, rank, port, clientTrainData, clientTestData, clientTrainLabelDataCount):
    # Create a FastAPI app for each process
    app = FastAPI()
    trainDataByLabels = {}
    testData = clientTestData # verify if assigning this is needed
    
    @app.post('/federatedLearningCompleted')
    def federatedLearningCompleted():
        os.kill(os.getpid(), signal.SIGTERM)
        return 'done'


    @app.post('/generateEncryptContext') ##
    async def generate_EncryptContext(clients: UploadFile = File(...)):
        global context 

        clientsAddrs = pickle.loads(await clients.read())

        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=2)
        context.generate_galois_keys()
        context.global_scale = 2**40
        
        clientsSerialzContext = context.serialize(save_secret_key=True) # all clients need to use the same context for the server to perform operations on compatible encrypted contents
        serverSerialzContext = context.serialize(save_secret_key=True) # context without secret key so that the server will not be able to decrypt the clients encrypted contents but with the context to perform operations on the encrypted contents

        with ThreadPoolExecutor() as executor: 
            futures = [executor.submit(lambda client: requests.post(f'http://127.0.0.1:{base_port - 1 + int(client)}/receiveEncryptContext', files={"serialized_client_context": clientsSerialzContext}), client) for client in clientsAddrs] ##

            for future in as_completed(futures):
                response = future.result()
                # print(response.json())

        return Response(serverSerialzContext, media_type='application/octet-stream')
    

    @app.post('/receiveEncryptContext') ##
    async def receive_EncryptionContext(serialized_client_context: UploadFile = File(...)):
        global context
        context = ts.context_from(await serialized_client_context.read(), n_threads=2)
        # print(context)
        return {"message": "received context"}
    
    
    @app.post('/decryptIntermediateComparisonResult') ##
    async def decryptComparisonResult(enc_comparison_val: UploadFile = File(...)):
        global context
        encComparisonValue = ts.ckks_vector_from(context, pickle.loads(await enc_comparison_val.read()))
        comparisonValue = encComparisonValue.decrypt()
        return Response(pickle.dumps(comparisonValue), media_type='application/octet-stream')


    @app.post('/encryptLabels') ##
    async def encryptLabels():
        global context
        enc_info = []
        for label, amount in clientTrainLabelDataCount.items():
            enc_info.append([ts.ckks_vector(context, [label]).serialize(), ts.ckks_vector(context, [amount]).serialize()])
        
        return Response(pickle.dumps(enc_info), media_type='application/octet-stream')


    @app.post('/placeholderToRealLabelMapping') ##
    async def placeholderToRealLabelMapping(mapping: UploadFile = File(...)):
        global context, PlaceholderMaptoRealTarget
        serialized_mapping = pickle.loads(await mapping.read())
        PlaceholderMaptoRealTarget = {placeholder: round(ts.ckks_vector_from(context, encRealLabel).decrypt()[0]) for placeholder, encRealLabel in serialized_mapping.items()}
        print(f'port {port} placeholderMapReal: {PlaceholderMaptoRealTarget}')
        return {'message': "received placeholder to real label maps"}
    
    
    # def preprocess_trainImage(img):
    #     transform_train = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    #     ])
    #     transformed_img = transform_train(img)
    #     return transformed_img

    # def preprocess_testImage(img):
    #     transform_test = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ])
    #     transformed_img = transform_test(img)
    #     return transformed_img
    
    
    # change name to start of new epoch & get batch size from server
    @app.post('/prepareTrainData')
    async def prepare_trainData():
        # global trainDataByLabels
        clientData_copy = clientTrainData.copy()
        clientData_copy['image'] = clientData_copy['image'].apply(lambda img: dataset_transform[args.dataset](img, train=True, augment=True))
        # FOR IMAGE: convert client data by label to each data-label generator
        for label in clientData_copy['label'].unique():
            trainDataByLabels[label] = iter(clientData_copy.loc[clientData_copy['label'] == label, ['image', 'label']].sample(frac=1, replace=False).itertuples(index=False, name=None))
        
        # FOR TABULAR
        # labels = clientData_copy['labels'].unique()
        # for label in labels:
        #     rows = clientData_copy.loc[clientData_copy['labels'] == label].drop(columns=['labels']).values
        #     trainDataByLabels[label] = iter([(torch.tensor(row), torch.tensor(label)) for row in rows])
        
        print(trainDataByLabels)
        return {"message": f"Train data is ready by process {rank}"}

    def prepare_testData():
        # global testData
        testData['image'] = testData['image'].apply(lambda img: dataset_transform[args.dataset](img, train=False, augment=False))
        
    
    
    def model_performance(labels, preds, totalTargetClass):
        conf_matrix = np.zeros((totalTargetClass, totalTargetClass), dtype=int)
        for pred, label in zip(preds, labels):
            conf_matrix[label, pred] += 1
    
        # Initialize lists to store precision, recall, and F1 scores for each class
        # precision_per_class = []
        # recall_per_class = []
        f1_per_class = []
        ba_per_class = []
        correct_per_class = []
        total_per_class = []
    
        # Calculate metrics for each class
        for i in range(conf_matrix.shape[0]):
            if conf_matrix[i, :].sum() != 0: # excluded metrics calculation for class that a client does not hold
                TP = conf_matrix[i, i]  # True Positives
                FP = conf_matrix[:, i].sum() - TP  # False Positives
                FN = conf_matrix[i, :].sum() - TP  # False Negatives
                TN = conf_matrix.sum() - (TP + FP + FN)  # True Negatives
    
                # Precision: TP / (TP + FP)
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                # Recall: TP / (TP + FN)
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                # Specificity = TN / (TN + FP)
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
                
                # Balanced Accuracy: (Recall + Specificity) / 2
                balanced_acc = (recall + specificity) / 2
    
                # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
                # # Append results
                # precision_per_class.append(precision)
                # recall_per_class.append(recall)
                f1_per_class.append(f1)
                ba_per_class.append(balanced_acc)
    
                correct_per_class.append(TP)
                total_per_class.append(conf_matrix[i, :].sum())
            
    
        total_correct = conf_matrix.diagonal().sum()  # Sum of all true positives
        total_samples = conf_matrix.sum()  # Total number of predictions
        normal_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        # print(f'total  correct:{total_correct}')
        # print(f'total sample: {total_samples}')
    
        # Calculate Macro-Averaged Precision, Recall, and F1
        # macro_precision = np.mean([p for p in precision_per_class if p != 0])
        # macro_balanceAcc = np.mean([r for r in recall_per_class if r != 0])
        macro_avg_f1 = np.mean([f for f in f1_per_class])
        macro_avg_ba_allclasses = np.mean([ba for ba in ba_per_class])
    
        # weighted overall ba per class (total i class sample * ba for i class + ...) / total all classes samples
        weighted_ba = np.sum([ba * total_samples for ba, total_samples in zip(ba_per_class, total_per_class)]) / np.sum(total_per_class)
        weighted_f1 = np.sum([f * total_samples for f, total_samples in zip(f1_per_class, total_per_class)]) / np.sum(total_per_class)

        return conf_matrix, correct_per_class, total_per_class, ba_per_class, f1_per_class, normal_accuracy, macro_avg_f1, weighted_f1, macro_avg_ba_allclasses, weighted_ba


    @app.post('/test')
    async def test():
        global model, epoch_total_correct, epoch_total_samples,  train_losses, train_labels, train_preds

        glbModelParam = pickle.loads(requests.get(f'http://127.0.0.1:{base_port - 1}/get_glb_params').content)
        model.load_state_dict(glbModelParam)
        model.eval()
        
        test_losses = 0
        test_labels = []
        test_preds = []

        with torch.no_grad():
            inputs = torch.stack(testData['image'].tolist()).float().to(device)
            labels = torch.tensor(testData['label'].values).to(device)
            
            # inputs = torch.from_numpy(testData.drop(columns=['labels']).values).to(device).float()
            # labels = torch.from_numpy(testData['labels'].values).long().to(device)
            preds = model(inputs)

            _, predicted = preds.max(1)
            total = labels.size(0)
            correct = predicted.eq(labels).sum().item()
            # print(total)
            # print(correct)
            
            test_losses += criterion(preds, labels).item()
            test_labels.extend(labels.tolist())
            test_preds.extend(predicted.tolist())
            
        conf_matrix, correct_per_class, total_per_class,  ba_per_class, f1_per_class, normal_accuracy, macro_avg_f1, weighted_f1, macro_avg_ba_allclasses, weighted_ba = model_performance(train_labels, train_preds, config['NonIIDSetup']['totalLabels']) ###  TOTAL CLASS    
        test_conf_matrix, test_correct_per_class, test_total_per_class,  test_ba_per_class, test_f1_per_class, test_normal_accuracy, test_macro_avg_f1, test_weighted_f1, test_macro_avg_ba_allclasses, test_weighted_ba = model_performance(test_labels, test_preds, config['NonIIDSetup']['totalLabels']) ###  TOTAL CLASS    

       
        
        conf_matrix = list([[int(value) for value in row] for row in conf_matrix.tolist()])
        test_conf_matrix = list([[int(value) for value in row] for row in test_conf_matrix.tolist()])
        
        response_json = pickle.dumps({f"client {rank}":  {"train_conf_matrix": conf_matrix,
                                    "train_correct_per_class": correct_per_class, 
                                    "train_total_per_class": total_per_class, 
                                    "train_ba_per_class": ba_per_class, 
                                    "train_f1_per_class": f1_per_class, 
                                    "train_normal_accuracy": normal_accuracy, 
                                    "train_macro_avg_f1": macro_avg_f1, "train_weighted_f1": weighted_f1, 
                                    "train_macro_avg_ba_allclasses": macro_avg_ba_allclasses, "train_weighted_ba": weighted_ba,
                                    "Train summed loss": train_losses,
                                    "test_conf_matrix": test_conf_matrix,
                                    "test_correct_per_class": test_correct_per_class, 
                                    "test_total_per_class": test_total_per_class, 
                                    "test_ba_per_class": test_ba_per_class, 
                                    "test_f1_per_class": test_f1_per_class, 
                                    "test_normal_accuracy": test_normal_accuracy, 
                                    "test_macro_avg_f1": test_macro_avg_f1, "test_weighted_f1": test_weighted_f1, 
                                    "test_macro_avg_ba_allclasses": test_macro_avg_ba_allclasses, "test_weighted_ba": test_weighted_ba,
                                    "Test summed loss": test_losses
                                   }})
        
        train_losses = 0
        train_labels = []
        train_preds = []
        
        return Response(response_json, media_type='application/octet-stream')
                                   
                                  
    @app.post('/train')
    async def train(to_train: UploadFile = File(...), glb_model_params: UploadFile = File(...), batch_size: UploadFile = File(...)):
        global model, criterion, batchSize, epoch_total_correct, epoch_total_samples, PlaceholderMaptoRealTarget, train_losses, train_labels, train_preds
        # print('in clinet train func')
        # Extract placeholders to train and global model parameters from server
        serializ_to_train = await to_train.read()
        to_train = pickle.loads(serializ_to_train)
        
        serializ_glb_model_params = await glb_model_params.read()
        glb_model_params = pickle.loads(serializ_glb_model_params)
        # print(f'to_train: {to_train}')
        # print(f'client port: {port}')
        # print(f'placeMapL: {PlaceholderMaptoRealTarget}')


        batchSize = pickle.loads(await batch_size.read())
        ResNet9_BN2D.batchSize = batchSize
        # ResNet18_BN2D.batchSize = batchSize
        
        placeMap2RealLabel = PlaceholderMaptoRealTarget ##
        # placeMap2RealLabel = pickle.loads(request.files['PlaceholderMaptoRealTarget'].read()) #####################new mapping code

        # forward and backward pass to accumulate local gradients summed
        model.load_state_dict(glb_model_params)
        model = model.to(device)
        model.train()
        
        samples = []
        for p, count in Counter(to_train).items():
            # print(p, count)
            # print(placeMap2RealLabel[p])
            realLabel = placeMap2RealLabel[p] ############################## new mapping code
            try:
                samples.extend(list(islice(trainDataByLabels[realLabel], count))) #################### new mapping code
            except Exception as e:
                print('exception')
                print(p, count)
                print(port)
        inputs = torch.stack(list(zip(*samples))[0]).to(device).float()
        labels = torch.tensor(list(zip(*samples))[1], dtype=torch.long).to(device)
        
        # stream = torch.cuda.Stream()
        # with torch.cuda.stream(stream):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs, 1)
        # epoch_total_correct = epoch_total_correct + (predicted == labels).sum().item()
        # epoch_total_samples = epoch_total_samples + labels.size(0)
        train_labels.extend(list(zip(*samples))[1])
        train_preds.extend(predicted.tolist())
        train_losses += loss.item()
    

        grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        
        # stream.synchronize()
        serialz_grad = pickle.dumps(grad)

        # return summed gradients to server
        return Response(serialz_grad, media_type='application/octet-stream')
        
    
    prepare_testData()
    # Start the app on the assigned port
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
    base_port = args.start_port + 1 # Client starting port (server port + 1)
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
    
    # Optionally join processes (this will block the main process)
    for p in processes:
        p.join()




