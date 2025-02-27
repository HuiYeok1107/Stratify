from args import args_parser
global args
args = args_parser()

import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
# from ResNet9_BN2D import *
# from model import *
# from ResNet18_BN2D import *
# from v2HomomorphicEncryption import * 
import random
device = torch.device('cuda')
from collections import Counter
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TBB_NUM_THREADS"] = "1" 
torch.set_num_threads(1) 

import signal
import gc
import tenseal as ts ##
import itertools ##
import string
from datetime import datetime

import logging 
import asyncio
import uvicorn
from fastapi import FastAPI, Request, Response, File, UploadFile
from fastapi.responses import JSONResponse
import httpx
# log = logging.getLogger('werkzeug')
# log.disabled = True # disable restapi logging
import psutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.model
from utils.model import batch_learning_model
from Batch_Data_Learning.custom_batch_norm import *

# from args import args_parser
# global args
# args = args_parser()

context2 = None
basePort = args.start_port
glb_model = batch_learning_model[args.dataset].to(device)

max_lr = 0.001
optimizer = optim.Adam(glb_model.parameters(), max_lr, weight_decay=0.001)  
lr_schedl = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=30, steps_per_epoch=int(50000/400))

# Shared variable to accumulate tensors
overallBatchSum = []
overallBatchStdv = []
batchMean = None
batchVar = None
overallTerm2 = []
overallPartTerm3 = []
batchTerm2 = None
batchPartTerm3 = None
totalAssignedClientsInBatch = 0

batchNormLayersBM = [] # use to store all the batch norm layers of glb_model for updating running batch mean & var usage
batchNormLayersVAR = []

# a threading event to signal when the local batch statistic received from all participating client
batchMean_ready_event = asyncio.Event()
batchVar_ready_event = asyncio.Event()
batchTerm2_ready_event = asyncio.Event()
batchPartTerm3_ready_event = asyncio.Event()

clientsCountTracker = 0

def set_cpu_affinity(core_ids):
    """
    Set the current process to run on specific CPU cores.

    Args:
        core_ids (list): List of core indices to bind the process to.
    """
    # Get the current process ID
    pid = os.getpid()

    # Get the process object
    process = psutil.Process(pid)

    # Set CPU affinity
    process.cpu_affinity(core_ids)

    print(f"Process {pid} is now bound to cores: {core_ids}")


# app = Flask(__name__)
# asgi_app = WsgiToAsgi(app)
app = FastAPI()
def init(sums, clientsAvailPlaceholderTargets):
    # Generate N length Placeholder list based on the total count for each placeholder obtained through homomorphic encryption process  
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
    
    # print('target counter:')
    # print(Counter(targets))
    
    # print('clinet avail placece')
    # print(clientsAvailPlaceholderTargets)
        
    for placeholder in clientAssignedForPlaceholder:
        # if len(clientAssignedForPlaceholder[placeholder]) < round(sums[placeholder]):  # randomly add client until length of assigned clients for a placeholder reach the desired length for placeholder list
        #     clients = list(set(clientAssignedForPlaceholder[placeholder]))
        #     while len(clientAssignedForPlaceholder[placeholder]) != round(sums[placeholder]):
        #         for client in clients:
        #             clientAssignedForPlaceholder[placeholder].append(client)
        #             if len(clientAssignedForPlaceholder[placeholder]) == round(sums[placeholder]):
        #                 break
        # elif len(clientAssignedForPlaceholder[placeholder]) > round(sums[placeholder]): # remove client until length of assigned clients for a placeholder matches the length for the placeholder in placeholder list
        #     while len(clientAssignedForPlaceholder[placeholder]) != round(sums[placeholder]):
        #         clientAssignedForPlaceholder[placeholder].pop()

        random.shuffle(clientAssignedForPlaceholder[placeholder])
        # print('in init')
        # print(Counter(clientAssignedForPlaceholder[placeholder]))
        
    # Map the assigned client for each placeholder in Placeholder list
    mappedClientAssignedForPlaceholder = []
    # for target in targets:
    #     mappedClientAssignedForPlaceholder.append(clientAssignedForPlaceholder[target].pop(0))
 
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
    print('in compute batch mean')
    global overallBatchSum, batchMean, clientsCountTracker, totalAssignedClientsInBatch, batchNormLayersBM

    # n = float(request.form['n'])
    # n = float(pickle.loads(request.files['n'].read()))
    # print(f'n: {n}')
    # localSum = pickle.loads(request.files['localSum'].read())

    n = float(pickle.loads(await n.read()))
    # print(f'n: {n}')
    localSum = pickle.loads(await localSum.read())
    print(f'local batch mean in gpu: {localSum.is_cuda, localSum[0].is_cuda}')
    # print(localSum)
    overallBatchSum.append(localSum) 
    
    if len(overallBatchSum) == totalAssignedClientsInBatch:
        batchMean = torch.sum(torch.stack(overallBatchSum, dim=0), dim=0) / n 
        
        print(f"batch mean: {batchMean.is_cuda}")
        # print(f"batch mean: {batchMean.is_contiguous()}")
        
        batchNormLayersBM[0].running_mean.data = batchNormLayersBM[0].momentum * batchMean + (1 - batchNormLayersBM[0].momentum) * batchNormLayersBM[0].running_mean # update glb model running mean
        
        # print(f"batch mean running: {batchNormLayersBM[0].running_mean.is_cuda}")
        # print(f"batch mean running: {batchNormLayersBM[0].running_mean.is_contiguous()}")
        
        
        del batchNormLayersBM[0] # delete the batch norm layer that has been updated with new running mean from list
        batchMean_ready_event.set()
    
    print('batch mean here')
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
    # print('in compute batch var')
    global overallBatchStdv, batchVar, clientsCountTracker, totalAssignedClientsInBatch, batchNormLayersVAR
    
    # n = float(request.form['n'])
    # n = float(pickle.loads(request.files['n'].read()))
    # print(f'n: {n}')
    # localStdv = pickle.loads(request.files['localStdv'].read())

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
    # print('compute batch term 2')
    global overallTerm2, batchTerm2, clientsCountTracker, totalAssignedClientsInBatch
    
    # localTerm2 = pickle.loads(request.data)
    localTerm2 = pickle.loads(await request.body())

    overallTerm2.append(localTerm2) 
    
    if len(overallTerm2) == totalAssignedClientsInBatch:
        batchTerm2 = torch.sum(torch.stack(overallTerm2, dim=0), dim=0)
        
        
        # print(f"batch term2: {batchTerm2.is_cuda}")
        # print(f"batch term2: {batchTerm2.is_contiguous()}")
        
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
    # print('in compute batch term part 3')
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

def send_generateEncryptContext_request(clients): ##
    global context2 
    selectedClient = random.choice(clients)
    clients.remove(selectedClient)
    response = requests.post(f'http://127.0.0.1:{basePort + selectedClient}/generateEncryptContext', files={"clients": pickle.dumps(clients)})
    clients.append(selectedClient)
    clients.sort()
    # print(f'in send context clients: {clients}')
    # data = response.json()
    data = response.content
    context2 = ts.context_from(data, n_threads=2)

    # return response.status_code 

def computePlaceholders(serialz_clients_enc_info):
    print('in compute placeholders')
    global context2 

    clients_enc_info = []
    for serialz_client_enc_info in serialz_clients_enc_info:
        print('running..')
        c_enc_info2 = [[ts.ckks_vector_from(context2, encSerialTargetVector), ts.ckks_vector_from(context2, encSerialTargetAmtVector)] for encSerialTargetVector, encSerialTargetAmtVector in serialz_client_enc_info]
        
        clients_enc_info.append(c_enc_info2)
    
    uniqueList = []
    for encTarget, encTargetAmt in list(itertools.chain(*clients_enc_info)):
        add_elem = True  
        if uniqueList:
            for unique_elem in uniqueList:
                enc_res = encTarget - unique_elem
                # * Server needs to send the enc_res back to the client for decryption and get back the result
                serialized_enc_res = enc_res.serialize()
                client_decrypted_res = ts.ckks_vector_from(context2, serialized_enc_res).decrypt()
                # res = requests.post(f'http://127.0.0.1:{basePort + 1}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(enc_res.serialize())})
                # client_decrypted_res = pickle.loads(res.content)

                # print(f"Res: {client_decrypted_res}")
                if abs(client_decrypted_res[0]) < 1e-5:  # Check if client_decrypted_res is close to zero
                    add_elem = False 
                    break
        if add_elem:
            # add a small encrypted value into the encrypted target to prevent transparent ciphertext issue in later comparison stage 
            encTarget = encTarget + ts.ckks_vector(context2, [0.0000001]) 
            uniqueList.append(encTarget)
    
    
    # placeholders = range(1, 200) #list(string.ascii_uppercase)
    # placeholders = list(string.ascii_uppercase)
    
    def generate_placeholders(limit):
        alphabet = string.ascii_uppercase  # 'A' to 'Z'
        placeholders = []
    
        # We start by adding single-letter placeholders
        for i in range(1, limit + 1):
            n = i
            result = ''
            
            # Convert the number to "base-26" where A=1, B=2, ..., Z=26
            while n > 0:
                n -= 1  # Decrement by 1 to handle base-26 correctly
                result = alphabet[n % 26] + result  # Get the corresponding letter
                n //= 26  # Integer division to reduce n
    
            placeholders.append(result)
        
        return placeholders
    
    placeholders = generate_placeholders(200)
    
    
    placeholderTargetMapEncRealTarget = {placeholderKey: encTarget for placeholderKey, encTarget in zip(placeholders, uniqueList)}
    clientsAvailPlaceholderTargets = {}
    # Server sends the client its respective list on placeholder target which map to its real target so that when the clients receive the list of targets to train from server, the client can know the target is correspond to which real target sample to train
    clientsPlaceholderTargetMapEncRealTarget = {}

    for clientID, clientEncTargetTargetAmtVectors in list(zip(['c' + str(num) for num in range(1, len(clients_enc_info)+1)], clients_enc_info)):   
        clientsAvailPlaceholderTargets[clientID] = []
        clientsPlaceholderTargetMapEncRealTarget[clientID] = {}
        for clientEncTargetTargetAmtVector in clientEncTargetTargetAmtVectors:
            for placeholder, encRealTargetInUniqueList in placeholderTargetMapEncRealTarget.items():
                enc_res = clientEncTargetTargetAmtVector[0] - encRealTargetInUniqueList
                # Server sends the enc_res back to the client for decryption and get back the comparison result
                client_decrypted_res = enc_res.decrypt()

                # res = requests.post(f'http://127.0.0.1:{basePort + int(clientID[1:])}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(enc_res.serialize())})
                # client_decrypted_res = pickle.loads(res.content)

                if abs(client_decrypted_res[0]) < 1e-5:
                    # print('in')
                    clientsAvailPlaceholderTargets[clientID].append((placeholder, clientEncTargetTargetAmtVector[1]))   
                    clientsPlaceholderTargetMapEncRealTarget[clientID][placeholder] = clientEncTargetTargetAmtVector[0]
                    break   

    # print(clientsPlaceholderTargetMapEncRealTarget)

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

        # res = requests.post(f'http://127.0.0.1:{basePort + int(clientID[1:])}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(encPlaceholderSum.serialize())})
        # clientDecrypted_placeholderSum = pickle.loads(res.content)[0] - noise
        # sums[placeholder] = clientDecrypted_placeholderSum

        serializedEncPlaceholderSum = encPlaceholderSum.serialize()
        clientDecrypted_placeholderSumWithNoise = ts.ckks_vector_from(context2, serializedEncPlaceholderSum).decrypt()[0] 
        clientDecrypted_placeholderSum = clientDecrypted_placeholderSumWithNoise - noise
        sums[placeholder] = clientDecrypted_placeholderSum
    # End.

    # calculate the normalized label proportion for each client label
    for client, items in clientsAvailPlaceholderTargets.items():
        for i, (placeholder, placeholderAmt) in enumerate(items):
            inverseTotal = 1 / sums[placeholder]
            encNormalizedLabelProp = placeholderAmt * inverseTotal  # Normalize by dividing by the sum

            # res = requests.post(f'http://127.0.0.1:{basePort + int(client[1:])}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(encNormalizedLabelProp.serialize())})
            # clientDecrypted_normalizedLabelProp = pickle.loads(res.content)[0]

            serialEncNormalizedLabelProp = encNormalizedLabelProp.serialize() # server serialized encrypted normalized value result and send to the respective client for decryption
            clientDecrypted_normalizedLabelProp = ts.ckks_vector_from(context2, serialEncNormalizedLabelProp).decrypt()[0] # client decrypt encrypted result and send back to server
            clientsAvailPlaceholderTargets[client][i] = (placeholder, clientDecrypted_normalizedLabelProp)  # Assign decrypted normalized value back

    # print(clientsAvailPlaceholderTargets)
    # print(sums)
    return sums, clientsAvailPlaceholderTargets, clientsPlaceholderTargetMapEncRealTarget
    

def send_PlaceholderMapToRealLabel(client, mapping):
    serialized_data = {placeholder: encRealLabel.serialize() for placeholder, encRealLabel in mapping.items()}
    status = requests.post(f'http://127.0.0.1:{basePort + int(client[1:])}/placeholderToRealLabelMapping', files={'mapping':pickle.dumps(serialized_data)})
    return status 

# def send_prepareTrainData_request(client):
#     print(f'clinet: {client}')
#     status = requests.post(f'http://127.0.0.1:{basePort + client}/prepareTrainData')
#     return status 

async def send_prepareTrainData_request(clients):
    print('in send prepare train data request funct')
    async with httpx.AsyncClient(timeout=None) as client_session:
        # Create a list of asynchronous POST request tasks
        tasks = [client_session.post(f"http://127.0.0.1:{basePort + client}/prepareTrainData") for client in clients]
        # Wait for all tasks to complete concurrently
        responses = await asyncio.gather(*tasks)

    return {"status": "Prepare training data signals sent"}

# def send_localTrain_request(serialz_glb_model_params, client, placeholdersToTrain, currentBatchSize, clientPlaceholderMapToRealTarget):
#     print(f'client to send train request to: {client}, {client[1:]}')
#     trainResult = requests.post(f'http://127.0.0.1:{basePort + int(client[1:])}/train', files={'to_train': pickle.dumps(placeholdersToTrain), 'glb_model_params': serialz_glb_model_params, 'batch_size': pickle.dumps(currentBatchSize)}) #, 'PlaceholderMaptoRealTarget': pickle.dumps(clientPlaceholderMapToRealTarget)
#     return trainResult

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


# def send_computeTestDataAcc_request(client):
#     result = requests.post(f'http://127.0.0.1:{basePort + client}/test')
#     return result

async def send_computeTestDataAcc_request(clients):
    async with httpx.AsyncClient(timeout=None) as client_session:
        # Create a list of asynchronous POST request tasks
        tasks = [client_session.post(f'http://127.0.0.1:{basePort + client}/test') for client in clients]
        # Wait for all tasks to complete concurrently
        results = await asyncio.gather(*tasks)

        epochResults = [pickle.loads(result.content) for result in results]

    return epochResults


# def send_trainingCompleted_signal(client):
#     result = requests.post(f'http://127.0.0.1:{basePort + client}/federatedLearningCompleted')
#     return result

async def send_trainingCompleted_signal(clients):
    async with httpx.AsyncClient(timeout=None) as client_session:
        # Create a list of asynchronous POST request tasks
        tasks = [client_session.post(f"http://127.0.0.1:{basePort + client}/federatedLearningCompleted") for client in clients]
        # Wait for all tasks to complete concurrently
        responses = await asyncio.gather(*tasks)

    return {"status": "Training completed signals sent"}

def get_all_batchNormLayers():
    global batchNormLayersBM, batchNormLayersVAR
    batchNormLayersBM = []
    batchNormLayersVAR = []
    for layer in glb_model.modules():  
        if isinstance(layer, CustomBatchNormManualModule):
            batchNormLayersBM.append(layer.to(device))
            batchNormLayersVAR.append(layer.to(device))
    return 'done'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

async def start_federated_learning(basePort):
    print('in start federated learning function')
    global totalAssignedClientsInBatch
    
    epochs = 30
    batchSize = 400
    clients = list(range(1, args.client_num + 1)) ## # replace range with actual clients address in production environment
    
    send_generateEncryptContext_request(clients) ##

    serialz_clients_enc_info = []
    for client in clients: # send the request using loop instead of threading because the server needs to know the encrypted label is sent from which client so that the assigned placeholder can be sent to the correct client in later stage
        response = requests.post(f'http://127.0.0.1:{basePort + int(client)}/encryptLabels')
        serialz_client_enc_info = pickle.loads(response.content)
        serialz_clients_enc_info.append(serialz_client_enc_info)

    
    # with ThreadPoolExecutor() as executor: 
    #     futures = [executor.submit(lambda client: requests.post(f'http://127.0.0.1:{5000 + int(client)}/encryptLabels'), client) for client in clients] ##

    #     for future in as_completed(futures):
    #         response = future.result()
    #         serialz_client_enc_info = pickle.loads(response.content)
    #         serialz_clients_enc_info.append(serialz_client_enc_info)
        
    sums, clientsAvailPlaceholderTargets, clientsPlaceholderTargetMapEncRealTarget = computePlaceholders(serialz_clients_enc_info) ##
    # print(f'clientsPlaceholderTargetMapEncRealTarget: {clientsPlaceholderTargetMapEncRealTarget}')
    for client, mapping in clientsPlaceholderTargetMapEncRealTarget.items():    
        send_PlaceholderMapToRealLabel(client, mapping) ##

    # os.kill(os.getpid(), signal.SIGTERM)
    
    resultFilePath = args.resultFilePath
    start_time = datetime.now()
    # with open(resultFilePath, 'a') as file:
    #     file.write(f"Training Start Time: {start_time}\n")
    #     file.write('\n')

    for epoch in range(1, epochs+1):
        lrs = []
        torch.cuda.empty_cache()
        # call init()
        targets, mappedClientAssignedForPlaceholder = init(sums, clientsAvailPlaceholderTargets) ##

        # with ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(send_prepareTrainData_request, client) for client in clients] ##

        #     for future in as_completed(futures):
        #         response = future.result()
        #         print(response.json())
        print('done init')
        await send_prepareTrainData_request(clients)
        
        start_time = datetime.now()
        
        # targets = targets[0:1500] # To Remove
        while len(targets) != 0:
            placeholderBatch = targets[0: batchSize] # extract placeholders from list based on the defined batch size
            targets = targets[batchSize:]
            # print(f'target length: {len(targets)}')
            

            clientsBatch = mappedClientAssignedForPlaceholder[0:batchSize] # extract the assigned clients for the current batch of placeholders
            totalAssignedClientsInBatch = len(set(clientsBatch))
            mappedClientAssignedForPlaceholder = mappedClientAssignedForPlaceholder[batchSize:]
            # print(placeholderBatch[0:batchSize])
            # print(clientsBatch)

            clientsPlaceholdersBatch = {client: [] for client in set(clientsBatch)}
            for client, placeholderToTrain in zip(clientsBatch, placeholderBatch):
                clientsPlaceholdersBatch[client].append(placeholderToTrain)
            
            # print(f'clientsPlaceholderBatch: {clientsPlaceholdersBatch}')
            get_all_batchNormLayers()
            serialz_glb_model_params = pickle.dumps(glb_model.state_dict())

            # # concurrent local training requests
            # accumulated_grads = []
            # with ThreadPoolExecutor() as executor:
            #     futures = [executor.submit(send_localTrain_request, serialz_glb_model_params, client, placeholdersToTrain, len(placeholderBatch), clientsPlaceholderTargetMapEncRealTarget[client]) for client, placeholdersToTrain in clientsPlaceholdersBatch.items()]

            #     for future in as_completed(futures):
            #         response = future.result()
            #         grads = pickle.loads(response.content)
          
            #         accumulated_grads.append(grads)

            accumulated_grads = await send_localTrain_request(serialz_glb_model_params, len(placeholderBatch), clientsPlaceholdersBatch)
            
            
            avg_grads = [torch.sum(torch.stack(grads), dim=0)/len(placeholderBatch) for grads in list(zip(*accumulated_grads))]

            optimizer.zero_grad()
            for param, avg_grad in zip(glb_model.parameters(), avg_grads):
                clipped_grad = torch.clamp(avg_grad, min=-0.01, max=0.01)
                param.grad = clipped_grad
                # param.grad = avg_grad
            
            optimizer.step()
            lrs.append(get_lr(optimizer))
            lr_schedl.step()
        
        epoch_acc = {"epoch": epoch, "result": [], "lr": lrs[-1]}
        epoch_time = datetime.now()

        # with ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(send_computeTestDataAcc_request, client) for client in clients]
    
        #     for future in as_completed(futures):
        #         resul = pickle.loads(future.result().content)
        #         epoch_acc['result'].append(resul)
        
        clientsTestResult = await send_computeTestDataAcc_request(clients)
        epoch_acc['result'] = clientsTestResult
        

        train_accs = []
        train_weight_accs = []
        train_macro_BA = []
        train_weight_BA = []
        train_macro_f1 = []
        train_weight_f1 = []
        total_train_samples = 0
        
        test_accs = []
        test_weight_accs = []
        test_macro_BA = []
        test_weight_BA = []
        test_macro_f1 = []
        test_weight_f1 = []
        total_test_samples = 0
        
        for client in epoch_acc['result']:
            client_name = list(client.keys())[0]
            train_accs.append(client[client_name]['train_normal_accuracy'])
            train_weight_accs.append(client[client_name]['train_normal_accuracy'] * np.sum(client[client_name]['train_total_per_class']))
            train_macro_BA.append(client[client_name]['train_macro_avg_ba_allclasses'])
            train_weight_BA.append(client[client_name]['train_weighted_ba'] * np.sum(client[client_name]['train_total_per_class']))
            train_macro_f1.append(client[client_name]['train_macro_avg_f1'])
            train_weight_f1.append(client[client_name]['train_weighted_f1'] * np.sum(client[client_name]['train_total_per_class']))

            total_train_samples += np.sum(client[client_name]['train_total_per_class'])
            
            test_accs.append(client[client_name]['test_normal_accuracy'])
            test_weight_accs.append(client[client_name]['test_normal_accuracy'] * np.sum(client[client_name]['test_total_per_class']))
            test_macro_BA.append(client[client_name]['test_macro_avg_ba_allclasses'])
            test_weight_BA.append(client[client_name]['test_weighted_ba'] * np.sum(client[client_name]['test_total_per_class']))
            test_macro_f1.append(client[client_name]['test_macro_avg_f1'])
            test_weight_f1.append(client[client_name]['test_weighted_f1'] * np.sum(client[client_name]['test_total_per_class']))
            
            total_test_samples += np.sum(client[client_name]['test_total_per_class'])

        
        with open(resultFilePath, 'a') as file:
            file.write(str(epoch_acc) + '\n')
            file.write('\n')
            file.write(
                f"Average Test metrics: Macro Test Acc: {np.mean(test_accs)} "
                f"Weighted Test Acc: {np.sum(test_weight_accs) / total_test_samples} "
                f"Macro Balanced Test Acc: {np.mean(test_macro_BA)} "
                f"Weighted Balanced Test Acc: {np.sum(test_weight_BA) / total_test_samples} "
                f"Macro Test F1: {np.mean(test_macro_f1)} "
                f"Weighted Test F1: {np.sum(test_weight_f1) / total_test_samples}\n"
                

                f"Average Train metrics: Macro Train Acc: {np.mean(train_accs)} "
                f"Weighted Train Acc: {np.sum(train_weight_accs) / total_train_samples} "
                f"Macro Balanced Train Acc: {np.mean(train_macro_BA)} "
                f"Weighted Balanced Train Acc: {np.sum(train_weight_BA) / total_train_samples} "
                f"Macro Train F1: {np.mean(train_macro_f1)} "
                f"Weighted Train F1: {np.sum(train_weight_f1) / total_train_samples}\n"
                f"\n"
                f"Per Epoch Training Start Time: {start_time}\n"
                f"Per Epoch Training End Time: {epoch_time}\n"
                f"\n"
            )
        

    end_time = datetime.now()
    with open(resultFilePath, 'a') as file:
        file.write(f"Training End Time: {end_time}\n")
        file.write('\n')
    
        # if epoch % 5 == 0:
    model_params = pickle.dumps(glb_model.state_dict())
    # Save the serialized model parameters to a file in the current directory
    with open(f"/home/user/huiyeok/embraceNonIID_S2/CIFAR-100/Solution2/ModelExp/modelParams-classHold{args.labelOrDomainPerClientHold}-alpha{args.dirichlet}-client{args.client_num}.pkl", 'wb') as f:
        f.write(model_params)
            
        
    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(send_trainingCompleted_signal, client) for client in clients]

    await send_trainingCompleted_signal(clients)
    
    os.kill(os.getpid(), signal.SIGTERM)


async def begin():
    basePort = args.start_port
    
    pid = os.getpid()
    process = psutil.Process(pid)
    current_affinity = process.cpu_affinity()
    process.cpu_affinity(current_affinity[args.client_num:]) 

    print(os.system('taskset -cp %s' %os.getpid())) 
    
    # # threading.Thread(target=start_federated_learning, kwargs={"basePort": basePort}, daemon=True).start()
    # asyncio.create_task(start_federated_learning(basePort))
    # print(f'process id: {os.getpid()}, {os.getppid()}')
    # # set_cpu_affinity([13, 14, 15])

    # # pid = os.getpid()  # Get the Flask app's process ID
    # # p = psutil.Process(pid)
    # # p.cpu_affinity([0, 1, 2])

    # # app.run(port=basePort, debug=False) #5000
    # # uvicorn.run(app, port=basePort)
    # configs = uvicorn.Config(app, port=basePort)
    # server = uvicorn.Server(configs)
    # await server.serve()
    configs = uvicorn.Config(app, port=basePort)
    server = uvicorn.Server(configs)

    loop = asyncio.get_event_loop()
    server_task = loop.create_task(server.serve())

    await asyncio.sleep(20)
    await start_federated_learning(basePort)

    await server_task
        
if __name__ == '__main__':
    # asyncio.run(begin())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(begin())

    
    
    
    
    