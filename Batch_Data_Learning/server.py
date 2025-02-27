from args import args_parser
global args
args = args_parser()

import asyncio
import uvicorn
from fastapi import FastAPI, Request, Response, File, UploadFile
import httpx
import requests

import tenseal as ts 
import torch
import torch.optim as optim
import numpy as np

import os
import sys
import pickle
import random
import signal
import itertools 
import string
from datetime import datetime
import psutil
import logging 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.model
from utils.model import batch_learning_model
from Batch_Data_Learning.custom_batch_norm import *
from utils.model_metrics import get_metrics_average

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TBB_NUM_THREADS"] = "1" 
torch.set_num_threads(1) 

# log = logging.getLogger('werkzeug')
# log.disabled = True # disable restapi logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tenseal context
context = None
# Shared variable to accumulate tensors
overallBatchSum, overallBatchStdv = [], []
batchMean, batchVar = None, None
overallTerm2, overallPartTerm3 = [], []
batchTerm2, batchPartTerm3 = None, None

# use to store all the batch norm layers of glb_model for updating running batch mean & var 
batchNormLayersBM, batchNormLayersVAR = [], [] 

# a threading event to signal when the local batch statistic received from all participating client
totalAssignedClientsInBatch = 0
batchMean_ready_event, batchVar_ready_event, batchTerm2_ready_event, batchPartTerm3_ready_event = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
clientsCountTracker = 0

basePort = args.start_port
glb_model = batch_learning_model[args.dataset].to(device)
max_lr = 0.001
optimizer = optim.Adam(glb_model.parameters(), max_lr, weight_decay=0.001)  
lr_schedl = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=30, steps_per_epoch=int(50000/400))

app = FastAPI()

def set_cpu_affinity(core_ids):
    # Set the current process to run on specific CPU cores.
    pid = os.getpid()  # Get the current process ID
    process = psutil.Process(pid) # Get the process object
    process.cpu_affinity(core_ids) # Set CPU affinity
    print(f"Process {pid} is now bound to cores: {core_ids}")


def init(sums, clientsAvailPlaceholderTargets):
    # Generate N length Placeholder list based on the global count for each placeholder obtained through homomorphic encryption process  
    targets = []
    for placeholder, count in sums.items():
        targets.extend([placeholder] * round(count))
    random.shuffle(targets)    

    # Generate N number of client ID for each client based on its normalized proportion for each placeholder
    clientAssignedForPlaceholder = {} 
    for client, placeholderProp in clientsAvailPlaceholderTargets.items():
        for placeholder, prop in placeholderProp:
            if placeholder not in clientAssignedForPlaceholder:
                clientAssignedForPlaceholder[placeholder] = []
            clientAssignedForPlaceholder[placeholder].extend([client] * (int(round(sums[placeholder]) * prop)))
        
    for placeholder in clientAssignedForPlaceholder:
        random.shuffle(clientAssignedForPlaceholder[placeholder])
        
    # Map the assigned client for each placeholder in Placeholder list
    mappedClientAssignedForPlaceholder = []
    targetsI_toremove = []
    for i, target in enumerate(targets):
        if clientAssignedForPlaceholder[target]:
            mappedClientAssignedForPlaceholder.append(clientAssignedForPlaceholder[target].pop(0))
        else:
            targetsI_toremove.append(i)
    
    for index in sorted(targetsI_toremove, reverse=True):
        del targets[index]
    
    return targets, mappedClientAssignedForPlaceholder


@app.get('/get_glb_params')
async def get_glb_params():
    model_params = pickle.dumps(glb_model.state_dict())
    return Response(model_params, media_type='application/octet-stream')


@app.post('/computeBatchMean')
async def compute_batch_mean(n: UploadFile = File(...), localSum: UploadFile = File(...)):
    global overallBatchSum, batchMean, clientsCountTracker, totalAssignedClientsInBatch, batchNormLayersBM

    n = float(pickle.loads(await n.read()))
    localSum = pickle.loads(await localSum.read())
    overallBatchSum.append(localSum) 
    
    if len(overallBatchSum) == totalAssignedClientsInBatch:
        batchMean = torch.sum(torch.stack(overallBatchSum, dim=0), dim=0) / n 
        batchNormLayersBM[0].running_mean.data = batchNormLayersBM[0].momentum * batchMean + (1 - batchNormLayersBM[0].momentum) * batchNormLayersBM[0].running_mean # update glb model running mean
        del batchNormLayersBM[0] # delete the batch norm layer that has been updated with new running mean from list
        batchMean_ready_event.set()
    
    # Wait until the batch mean computation event is completed
    await batchMean_ready_event.wait()
    serialzBatchMean = pickle.dumps(batchMean)

    clientsCountTracker = clientsCountTracker + 1
    if clientsCountTracker == totalAssignedClientsInBatch:
        overallBatchSum = []
        batchMean = None
        batchMean_ready_event.clear() # reset event
        clientsCountTracker = 0

    # return serialized batch mean back to client
    return Response(serialzBatchMean, media_type='application/octet-stream')


@app.post('/computeBatchVar')
async def compute_batch_var(n: UploadFile = File(...), localStdv: UploadFile = File(...)):
    global overallBatchStdv, batchVar, clientsCountTracker, totalAssignedClientsInBatch, batchNormLayersVAR

    n = float(pickle.loads(await n.read()))
    localStdv = pickle.loads(await localStdv.read())

    overallBatchStdv.append(localStdv) 
    if len(overallBatchStdv) == totalAssignedClientsInBatch:
        batchVar = torch.sum(torch.stack(overallBatchStdv, dim=0), dim=0) / n 
        batchNormLayersVAR[0].running_var.data = batchNormLayersVAR[0].momentum * batchVar * n / (n - 1) + (1 - batchNormLayersVAR[0].momentum) * batchNormLayersVAR[0].running_var # update glb model running var
        del batchNormLayersVAR[0] # delete the batch norm layer that has been updated with new running var from list
        batchVar_ready_event.set()
    
    # Wait until the batch variance computation event is completed
    await batchVar_ready_event.wait()
    serialzBatchVar = pickle.dumps(batchVar)

    clientsCountTracker = clientsCountTracker + 1
    if clientsCountTracker == totalAssignedClientsInBatch:
        overallBatchStdv = []
        batchVar = None
        batchVar_ready_event.clear()  # reset event
        clientsCountTracker = 0

    # return serialized batch var back to client
    return Response(serialzBatchVar, media_type='application/octet-stream')


@app.post('/computeBatchTerm2')
async def compute_batch_term2(request: Request):
    global overallTerm2, batchTerm2, clientsCountTracker, totalAssignedClientsInBatch

    localTerm2 = pickle.loads(await request.body())
    overallTerm2.append(localTerm2) 
    if len(overallTerm2) == totalAssignedClientsInBatch:
        batchTerm2 = torch.sum(torch.stack(overallTerm2, dim=0), dim=0)
        batchTerm2_ready_event.set()
    
    # Wait until the batch term 2 computation event is completed
    await batchTerm2_ready_event.wait()
    serialzBatchTerm2 = pickle.dumps(batchTerm2)

    clientsCountTracker = clientsCountTracker + 1
    if clientsCountTracker == totalAssignedClientsInBatch:
        overallTerm2 = []
        batchTerm2 = None
        batchTerm2_ready_event.clear()  # reset event
        clientsCountTracker = 0

    # return serialized batch var back to client
    return Response(serialzBatchTerm2, media_type='application/octet-stream')


@app.post('/computeBatchPartTerm3')
async def compute_batch_partTerm3(request: Request):
    global overallPartTerm3, batchPartTerm3, clientsCountTracker, totalAssignedClientsInBatch
    
    localPartTerm3 = pickle.loads(await request.body())

    overallPartTerm3.append(localPartTerm3) 
    if len(overallPartTerm3) == totalAssignedClientsInBatch:
        batchPartTerm3 = torch.sum(torch.stack(overallPartTerm3, dim=0), dim=0)
        batchPartTerm3_ready_event.set()
    
    # Wait until the batch term 2 computation event is completed
    await batchPartTerm3_ready_event.wait()
    serialzBatchPartTerm3 = pickle.dumps(batchPartTerm3)

    # reset global variables
    clientsCountTracker = clientsCountTracker + 1
    if clientsCountTracker == totalAssignedClientsInBatch:
        overallPartTerm3 = []
        batchPartTerm3 = None
        batchPartTerm3_ready_event.clear()  # reset event
        clientsCountTracker = 0

    # return serialized batch var back to client
    return Response(serialzBatchPartTerm3, media_type='application/octet-stream')


def get_all_batchNormLayers():
    global batchNormLayersBM, batchNormLayersVAR
    batchNormLayersBM = []
    batchNormLayersVAR = []
    for layer in glb_model.modules():  
        if isinstance(layer, CustomBatchNormManualModule):
            batchNormLayersBM.append(layer.to(device))
            batchNormLayersVAR.append(layer.to(device))
    return 'done'


def send_generateEncryptContext_request(clients): 
    global context 

    clonedClients = clients.copy()
    randIdx = random.randint(0, len(clonedClients)-1)
    selectedClient = clonedClients[randIdx]
    clonedClients.pop(randIdx)

    response = requests.post(f'http://127.0.0.1:{basePort + selectedClient}/generateEncryptContext', files={"clients": pickle.dumps(clonedClients)})
    context = ts.context_from(response.content, n_threads=2)


def computePlaceholders(serialz_clients_enc_info):
    global context 

    clients_enc_info = []
    for serialz_client_enc_info in serialz_clients_enc_info:
        c_enc_info = [[ts.ckks_vector_from(context, encSerialTargetVector), ts.ckks_vector_from(context, encSerialTargetAmtVector)] for encSerialTargetVector, encSerialTargetAmtVector in serialz_client_enc_info]
        clients_enc_info.append(c_enc_info)
    
    uniqueList = []
    for encTarget, encTargetAmt in list(itertools.chain(*clients_enc_info)):
        add_elem = True  
        if uniqueList:
            for unique_elem in uniqueList:
                enc_res = encTarget - unique_elem
                # send the enc_res back to the client for decryption and get back the result
                # serialized_enc_res = enc_res.serialize()
                # client_decrypted_res = ts.ckks_vector_from(context, serialized_enc_res).decrypt()
                res = requests.post(f'http://127.0.0.1:{basePort + 1}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(enc_res.serialize()), 'placeholder_to_target_mapping_stage': 'True'})
                client_decrypted_res = pickle.loads(res.content)

                if client_decrypted_res == 0:  
                    add_elem = False 
                    break
        if add_elem:
            # add a small encrypted value into the encrypted target to prevent transparent ciphertext issue in later comparison stage 
            encTarget = encTarget + ts.ckks_vector(context, [0.0000001]) 
            uniqueList.append(encTarget)
    
    
    def generate_placeholders(limit):
        alphabet = string.ascii_uppercase  # 'A' to 'Z'
        placeholders = []
        # start by adding single-letter placeholders
        for i in range(1, limit + 1):
            n = i
            result = ''
            # convert the number to "base-26" where A=1, B=2, ..., Z=26
            while n > 0:
                n -= 1  # decrement by 1 to handle base-26 correctly
                result = alphabet[n % 26] + result  # get the corresponding letter
                n //= 26  # integer division to reduce n
            placeholders.append(result)
        return placeholders
    
    placeholders = generate_placeholders(200)
    
    
    placeholderTargetMapEncRealTarget = {placeholderKey: encTarget for placeholderKey, encTarget in zip(placeholders, uniqueList)}
    clientsAvailPlaceholderTargets = {}
    # Server sends the client its respective list on placeholder target which map to its real target so that when the clients receive the list of placeholders to train from server, the client can know the target is correspond to which real target sample to train
    clientsPlaceholderTargetMapEncRealTarget = {}

    for clientID, clientEncTargetTargetAmtVectors in list(zip(['c' + str(num) for num in range(1, len(clients_enc_info)+1)], clients_enc_info)):   
        clientsAvailPlaceholderTargets[clientID] = []
        clientsPlaceholderTargetMapEncRealTarget[clientID] = {}
        for clientEncTargetTargetAmtVector in clientEncTargetTargetAmtVectors:
            for placeholder, encRealTargetInUniqueList in placeholderTargetMapEncRealTarget.items():
                enc_res = clientEncTargetTargetAmtVector[0] - encRealTargetInUniqueList
                # Server sends the enc_res back to the client for decryption and get back the comparison result
                # client_decrypted_res = enc_res.decrypt()

                res = requests.post(f'http://127.0.0.1:{basePort + int(clientID[1:])}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(enc_res.serialize()), 'placeholder_to_target_mapping_stage': 'True'})
                client_decrypted_res = pickle.loads(res.content)

                if client_decrypted_res == 0:
                    clientsAvailPlaceholderTargets[clientID].append((placeholder, clientEncTargetTargetAmtVector[1]))   
                    clientsPlaceholderTargetMapEncRealTarget[clientID][placeholder] = clientEncTargetTargetAmtVector[0]
                    break   
    
    # global counts
    sums = {}
    for client, items in clientsAvailPlaceholderTargets.items():
        for placeholder, placeholderAmt in items:
            if placeholder not in sums:
                sums[placeholder] = 0
            sums[placeholder] = sums[placeholder] + placeholderAmt

    for placeholder, encPlaceholderSum in sums.items():
        # server adds random value to encrypted sum, serialized sum vector and send to client for decryption 
        noise = random.randint(100, 1000)
        encPlaceholderSum = encPlaceholderSum + noise

        res = requests.post(f'http://127.0.0.1:{basePort + int(clientID[1:])}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(encPlaceholderSum.serialize()), 'placeholder_to_target_mapping_stage': 'False'})
        clientDecrypted_placeholderSum = pickle.loads(res.content)[0] - noise
        sums[placeholder] = clientDecrypted_placeholderSum

        # serializedEncPlaceholderSum = encPlaceholderSum.serialize()
        # clientDecrypted_placeholderSumWithNoise = ts.ckks_vector_from(context, serializedEncPlaceholderSum).decrypt()[0] 
        # clientDecrypted_placeholderSum = clientDecrypted_placeholderSumWithNoise - noise
        # sums[placeholder] = clientDecrypted_placeholderSum


    # calculate the normalized label proportion for each client label
    for client, items in clientsAvailPlaceholderTargets.items():
        for i, (placeholder, placeholderAmt) in enumerate(items):
            inverseTotal = 1 / sums[placeholder]
            encNormalizedLabelProp = placeholderAmt * inverseTotal  # Normalize by dividing by the sum

            res = requests.post(f'http://127.0.0.1:{basePort + int(client[1:])}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(encNormalizedLabelProp.serialize()), 'placeholder_to_target_mapping_stage': 'False'})
            clientDecrypted_normalizedLabelProp = pickle.loads(res.content)[0]

            # serialEncNormalizedLabelProp = encNormalizedLabelProp.serialize() # server serialized encrypted normalized value result and send to the respective client for decryption
            # clientDecrypted_normalizedLabelProp = ts.ckks_vector_from(context, serialEncNormalizedLabelProp).decrypt()[0] # client decrypt encrypted result and send back to server

            clientsAvailPlaceholderTargets[client][i] = (placeholder, clientDecrypted_normalizedLabelProp)  # Assign decrypted normalized value back

    return sums, clientsAvailPlaceholderTargets, clientsPlaceholderTargetMapEncRealTarget
    

def send_PlaceholderMapToRealLabel(client, mapping):
    serialized_data = {placeholder: encRealLabel.serialize() for placeholder, encRealLabel in mapping.items()}
    status = requests.post(f'http://127.0.0.1:{basePort + int(client[1:])}/placeholderToRealLabelMapping', files={'mapping':pickle.dumps(serialized_data)})
    return status 


async def send_prepareTrainData_request(clients):
    async with httpx.AsyncClient(timeout=None) as client_session:
        # Create a list of asynchronous POST request tasks
        tasks = [client_session.post(f"http://127.0.0.1:{basePort + client}/prepareTrainData") for client in clients]
        # Wait for all tasks to complete concurrently
        responses = await asyncio.gather(*tasks)

    return {"status": "Prepare training data signals sent"}


async def send_localTrain_request(serialz_glb_model_params, currentBatchSize, clientsPlaceholdersBatch):
    async with httpx.AsyncClient(timeout=None) as client_session:
        tasks = [
            client_session.post(
                f'http://127.0.0.1:{basePort + int(client[1:])}/train', 
                files={'to_train': pickle.dumps(placeholdersToTrain), 'glb_model_params': serialz_glb_model_params, 'batch_size': pickle.dumps(currentBatchSize)} 
            )                                                                               
            for client, placeholdersToTrain in clientsPlaceholdersBatch.items()
        ]
        responses = await asyncio.gather(*tasks)
        clients_summed_gradients = [pickle.loads(response.content) for response in responses]
    
    return clients_summed_gradients


async def send_computeTestDataAcc_request(clients):
    async with httpx.AsyncClient(timeout=None) as client_session:
        tasks = [client_session.post(f'http://127.0.0.1:{basePort + client}/test') for client in clients]
        results = await asyncio.gather(*tasks)

        epochResults = [pickle.loads(result.content) for result in results]

    return epochResults


async def send_trainingCompleted_signal(clients):
    async with httpx.AsyncClient(timeout=None) as client_session:
        tasks = [client_session.post(f"http://127.0.0.1:{basePort + client}/federatedLearningCompleted") for client in clients]
        responses = await asyncio.gather(*tasks)
    return {"status": "Training completed signals sent"}


async def start_federated_learning(basePort):
    global totalAssignedClientsInBatch
    
    epochs = 2
    batchSize = 400
    clients = list(range(1, args.client_num + 1)) # replace range with actual clients address in production environment
    
    send_generateEncryptContext_request(clients) 

    serialz_clients_enc_info = []
    for client in clients: # send the request using loop instead of threading because the server needs to know the encrypted label is sent from which client so that the assigned placeholder can be sent to the correct client in later stage
        response = requests.post(f'http://127.0.0.1:{basePort + int(client)}/encryptLabels')
        serialz_client_enc_info = pickle.loads(response.content)
        serialz_clients_enc_info.append(serialz_client_enc_info)
        
    sums, clientsAvailPlaceholderTargets, clientsPlaceholderTargetMapEncRealTarget = computePlaceholders(serialz_clients_enc_info) ##
    
    for client, mapping in clientsPlaceholderTargetMapEncRealTarget.items():    
        send_PlaceholderMapToRealLabel(client, mapping) 
    
    resultFilePath = args.resultFilePath
    start_time = datetime.now()

    for epoch in range(1, epochs+1):
        torch.cuda.empty_cache()
        targets, mappedClientAssignedForPlaceholder = init(sums, clientsAvailPlaceholderTargets) ##

        await send_prepareTrainData_request(clients)
        
        start_time = datetime.now()

        while len(targets) != 0:
            placeholderBatch = targets[0: batchSize] # extract placeholders from list based on the defined batch size
            targets = targets[batchSize:]

            clientsBatch = mappedClientAssignedForPlaceholder[0:batchSize] # extract the assigned clients for the current batch of placeholders
            totalAssignedClientsInBatch = len(set(clientsBatch))
            mappedClientAssignedForPlaceholder = mappedClientAssignedForPlaceholder[batchSize:]

            clientsPlaceholdersBatch = {client: [] for client in set(clientsBatch)}
            for client, placeholderToTrain in zip(clientsBatch, placeholderBatch):
                clientsPlaceholdersBatch[client].append(placeholderToTrain)

            get_all_batchNormLayers()
            serialz_glb_model_params = pickle.dumps(glb_model.state_dict())

            # get the average grads by summing up all the summed gradients of clients and then divide by batch size
            accumulated_grads = await send_localTrain_request(serialz_glb_model_params, len(placeholderBatch), clientsPlaceholdersBatch)
            avg_grads = [torch.sum(torch.stack(grads), dim=0)/len(placeholderBatch) for grads in list(zip(*accumulated_grads))]
            # update global model with the average grads
            optimizer.zero_grad()
            for param, avg_grad in zip(glb_model.parameters(), avg_grads):
                clipped_grad = torch.clamp(avg_grad, min=-0.01, max=0.01)
                param.grad = clipped_grad
                # param.grad = avg_grad
            optimizer.step()
            lr_schedl.step()
        
        epoch_acc = {"epoch": epoch, "result": []}
        epoch_time = datetime.now()
        
        clientsTestResult = await send_computeTestDataAcc_request(clients)
        epoch_acc['result'] = clientsTestResult

        epoch_avg_result = get_metrics_average(epoch_acc['result'])
        with open(resultFilePath, 'a') as file:
            # file.write(str(epoch_acc) + '\n')
            file.write(f"Epoch {epoch}:")
            file.write(epoch_avg_result)
            file.write(
                f"Per Epoch Training Start Time: {start_time}\n"
                f"Per Epoch Training End Time: {epoch_time}\n"
                f"\n"
            )

        # train_accs = []
        # train_weight_accs = []
        # train_macro_BA = []
        # train_weight_BA = []
        # train_macro_f1 = []
        # train_weight_f1 = []
        # total_train_samples = 0
        
        # test_accs = []
        # test_weight_accs = []
        # test_macro_BA = []
        # test_weight_BA = []
        # test_macro_f1 = []
        # test_weight_f1 = []
        # total_test_samples = 0
        
        # for client in epoch_acc['result']:
        #     client_name = list(client.keys())[0]
        #     train_accs.append(client[client_name]['train_normal_accuracy'])
        #     train_weight_accs.append(client[client_name]['train_normal_accuracy'] * client[client_name]['train_size'])
        #     train_macro_BA.append(client[client_name]['train_macro_avg_ba_allclasses'])
        #     train_weight_BA.append(client[client_name]['train_weighted_ba'] * client[client_name]['train_size'])
        #     train_macro_f1.append(client[client_name]['train_macro_avg_f1'])
        #     train_weight_f1.append(client[client_name]['train_weighted_f1'] * client[client_name]['train_size'])

        #     total_train_samples += client[client_name]['train_size']
            
        #     test_accs.append(client[client_name]['test_normal_accuracy'])
        #     test_weight_accs.append(client[client_name]['test_normal_accuracy'] * client[client_name]['test_size'])
        #     test_macro_BA.append(client[client_name]['test_macro_avg_ba_allclasses'])
        #     test_weight_BA.append(client[client_name]['test_weighted_ba'] * client[client_name]['test_size'])
        #     test_macro_f1.append(client[client_name]['test_macro_avg_f1'])
        #     test_weight_f1.append(client[client_name]['test_weighted_f1'] * client[client_name]['test_size'])
            
        #     total_test_samples += client[client_name]['test_size']

        
        # with open(resultFilePath, 'a') as file:
        #     file.write(str(epoch_acc) + '\n')
        #     file.write('\n')
        #     file.write(
        #         f"Average Test metrics: Macro Test Acc: {np.mean(test_accs)} "
        #         f"Weighted Test Acc: {np.sum(test_weight_accs) / total_test_samples} "
        #         f"Macro Balanced Test Acc: {np.mean(test_macro_BA)} "
        #         f"Weighted Balanced Test Acc: {np.sum(test_weight_BA) / total_test_samples} "
        #         f"Macro Test F1: {np.mean(test_macro_f1)} "
        #         f"Weighted Test F1: {np.sum(test_weight_f1) / total_test_samples}\n"
                

        #         f"Average Train metrics: Macro Train Acc: {np.mean(train_accs)} "
        #         f"Weighted Train Acc: {np.sum(train_weight_accs) / total_train_samples} "
        #         f"Macro Balanced Train Acc: {np.mean(train_macro_BA)} "
        #         f"Weighted Balanced Train Acc: {np.sum(train_weight_BA) / total_train_samples} "
        #         f"Macro Train F1: {np.mean(train_macro_f1)} "
        #         f"Weighted Train F1: {np.sum(train_weight_f1) / total_train_samples}\n"
        #         f"\n"
        #         f"Per Epoch Training Start Time: {start_time}\n"
        #         f"Per Epoch Training End Time: {epoch_time}\n"
        #         f"\n"
        #     )
        

    end_time = datetime.now()
    with open(resultFilePath, 'a') as file:
        file.write(f"Training End Time: {end_time}\n")
        file.write('\n')
    
    model_params = pickle.dumps(glb_model.state_dict())
    # Save the serialized model parameters to a file in the current directory
    with open(f"ModelParams-ClassHold{args.labelOrDomainPerClientHold}-DirAlpha{args.dirichlet}-ClientNum{args.client_num}.pkl", 'wb') as f:
        f.write(model_params)
            
    await send_trainingCompleted_signal(clients)
    os.kill(os.getpid(), signal.SIGTERM)


async def begin():
    basePort = args.start_port
    
    pid = os.getpid()
    process = psutil.Process(pid)
    current_affinity = process.cpu_affinity()
    process.cpu_affinity(current_affinity[args.client_num:]) 

    print(os.system('taskset -cp %s' %os.getpid())) 
    
    configs = uvicorn.Config(app, port=basePort)
    server = uvicorn.Server(configs)

    loop = asyncio.get_event_loop()
    server_task = loop.create_task(server.serve())

    await asyncio.sleep(20)
    await start_federated_learning(basePort)

    await server_task
        
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(begin())

    
    
    
    
    