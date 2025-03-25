from args import args_parser
global args
args = args_parser()

from fastapi import FastAPI, Response,  File, UploadFile, Request
import asyncio
import uvicorn
import httpx
import requests
import threading
import concurrent.futures
import io 

import psutil
import pickle
import random
import os 
import signal
import itertools
from collections import Counter 
import string
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.model
from utils.model import single_sample_learning_model
from utils.model_metrics import get_metrics_average

import torch
import numpy as np
import tenseal as ts
context = None 
import time

basePort = args.start_port
glb_model = single_sample_learning_model[args.dataset]

app = FastAPI()
@app.get('/get_glb_params')
async def get_glb_params():
    model_params = pickle.dumps(glb_model.state_dict())
    return Response(model_params, media_type='application/octet-stream')


# init func targets, and client available placeholders dict
def initVars(sums, clientsAvailPlaceholderTargets):
    # Generate N length Placeholder list based on the total count for each placeholder obtained through homomorphic encryption process  
    targets = []
    for placeholder, count in sums.items():
        targets.extend([placeholder] * round(count))
    random.shuffle(targets)

    clientsAvailPlaceholderTarget = clientsAvailPlaceholderTargets

    # queue clients assigned to train labels in sequence
    consecutiveClientsInQueue = [] 
    return targets, clientsAvailPlaceholderTarget, consecutiveClientsInQueue


def getCurrentClientsAvailByPlaceholder(clientsAvailPlaceholderTargets):
    clientsAvailForEachPlaceh = {}
    for client, label_propo_tuple in clientsAvailPlaceholderTargets.items():
        for label, propo in label_propo_tuple:
            if label not in clientsAvailForEachPlaceh:
                clientsAvailForEachPlaceh[label] = []
            clientsAvailForEachPlaceh[label].append((client, propo)) 
    return clientsAvailForEachPlaceh


async def send_prepareTrainData_request(clients):
    async with httpx.AsyncClient(timeout=None) as client_session:
        # Create a list of asynchronous POST request tasks
        tasks = [client_session.post(f"http://127.0.0.1:{basePort + client}/prepareTrainData") for client in clients]
        # Wait for all tasks to complete concurrently
        responses = await asyncio.gather(*tasks)
    return {"status": "Prepare training data signals sent"}


# Dynamic sub-window.  Initial sub-window consider only the clients list for current i. To determine whether to increase the upper and lower bound of a sub-window
    # Increase upper bound: if any clients of current i exist in the list before i (i-1)
    # Increase lower bound: if the top freq client(s) of current i for the current upper and lower bound sub-window exist in the next list after i (i+1) and 
                            # continue increase when the updated current upper and lower bound sub-window top freq client(s) exist in subsequent consecutive lists (i+1+x) 
def sortClients(windowTargets, clientsLabels, probSelectedClientsAvailForSort_forEachTarget): 
    # for each target (label/region) in window, identify which clients have data on that target
    listOfClientsWithAvailTarget_atWindowI = {target_position: [] for target_position in range(0, len(windowTargets))}
    for i, target in enumerate(windowTargets):
        for ckey, cTargetAmountAvailTuple in clientsLabels.items():
            if cTargetAmountAvailTuple: # avoid indexing empty list
                if (target in list(zip(*cTargetAmountAvailTuple))[0]) and (ckey in list(zip(*probSelectedClientsAvailForSort_forEachTarget[target]))[0]):  
                    listOfClientsWithAvailTarget_atWindowI[i].append(ckey)
    
    windowI_subWindow_clientsCoverage = {}
    sortedClientsForEachWindowTargetI = {}
    for i in range(0, len(listOfClientsWithAvailTarget_atWindowI)):
        expandSubWindowLowerBound = True
        lowerSubWindowBound = 0
        start_idx = i 
        end_idx = i

        upperSubWindowBound = max(0, i-1) # prevent out-of-bound upper sub-window
        if set(listOfClientsWithAvailTarget_atWindowI[upperSubWindowBound]) & set(listOfClientsWithAvailTarget_atWindowI[i]): # increase upper bound by i-1 if the clients in i exist in i-1 list
            start_idx = upperSubWindowBound

        while expandSubWindowLowerBound:
            windowI_subWindow_clientsCoverage[i] = Counter(list(itertools.chain(*list(listOfClientsWithAvailTarget_atWindowI.values())[start_idx: end_idx+1]))) # calculate clients freq for current upper and lower bounds sub-window
            lowerSubWindowBound += 1
            lowerSubWindowBound = min(len(listOfClientsWithAvailTarget_atWindowI)-1-i, lowerSubWindowBound) # prevent out-of-bound lower sub-window
 
            maxFreqInSubWindow = max(windowI_subWindow_clientsCoverage[i].values())
            clientsWithMaxFreq = [client for client, freq in windowI_subWindow_clientsCoverage[i].items() if freq == maxFreqInSubWindow] # get the top freq client(s) for the current sub-window size
            clientsWithMaxFreq = set(clientsWithMaxFreq) & set(listOfClientsWithAvailTarget_atWindowI[i]) # consider only freq for clients at i 
            
            if not(set(clientsWithMaxFreq) & set(listOfClientsWithAvailTarget_atWindowI[i+lowerSubWindowBound])): # if the highest freq client(s) of i not exist in next i+lower sub-window bound 
                lowerSubWindowBound -= 1
                end_idx = i + lowerSubWindowBound
                expandSubWindowLowerBound = False 
            else:
                end_idx = i + lowerSubWindowBound

            if i+lowerSubWindowBound == len(listOfClientsWithAvailTarget_atWindowI)-1: # if the sub-window lower bound hits the last element of window
                expandSubWindowLowerBound = False

        sortedClientsForEachWindowTargetI[i] = sorted(listOfClientsWithAvailTarget_atWindowI[i], key=lambda client: windowI_subWindow_clientsCoverage[i][client], reverse=True)
    return sortedClientsForEachWindowTargetI


def groupConsecutiveClients(windowTargets_selectedClients, windowTargets):
    # group consecutive identical client ID
    grouped_by_consecutive_clients = [list(group) for key, group in itertools.groupby(windowTargets_selectedClients)]
    consecutiveClientsAndTargetsList = []
  
    # records the consecutive client ID and the targets assigned to the client 
    for group in grouped_by_consecutive_clients:
        # extract targets to train from list for the consecutive client 
        consecutiveClientsAndTargetsList.append({'assigned_client': group[0], 'target_to_train': windowTargets[0:len(group)]})
        windowTargets = windowTargets[len(group):]
    
    return consecutiveClientsAndTargetsList


def updateProbabilisticSelectedClientsLists(windowTargets_selectedClients, windowTargets, clientsAvailPlaceholderTargets):
    clientsAvailForSort_forEachTarget = getCurrentClientsAvailByPlaceholder(clientsAvailPlaceholderTargets) # get all the current clients available for each placeholder
    for clientAssigned, windowTarget in list(zip(windowTargets_selectedClients, windowTargets)):  # if a client already got assigned to train on a target label in the current window, exclude this client from client assignment process for the same placeholder in the upcoming window
        if (clientAssigned in list(zip(*clientsAvailForSort_forEachTarget[windowTarget]))[0]) and (len(clientsAvailForSort_forEachTarget[windowTarget]) != 1):
            clientsAvailForSort_forEachTarget[windowTarget] = [client_amt_tuple for client_amt_tuple in clientsAvailForSort_forEachTarget[windowTarget] if client_amt_tuple[0] != clientAssigned]
    
    for placeholder, clientPropTuples in clientsAvailForSort_forEachTarget.items(): # for each placeholder's clients available list, include or exclude this client from the list based on the client's normalized placeholder proportion among the clients. This is to avoid having only the client with higher amount of label data remaining near the end of the placeholders list, and in which leads to model overfitting to this client domains data.
        selectedClients = list(set(random.choices(list(list(zip(*clientPropTuples))[0]), weights=list(list(zip(*clientPropTuples))[1]), k=len(clientPropTuples))))  # NEW CODE LOGIC TO TEST
        clientsAvailForSort_forEachTarget[placeholder] = [(client, prop) for client, prop in clientPropTuples if client in selectedClients]

    return clientsAvailForSort_forEachTarget # return each placeholder clients available list prepared for clients sorting 


async def send_localTrain_request(assignedClient, placeholdersToTrain, nextClient): 
    async with httpx.AsyncClient(timeout=None) as client_session:
        trainResult = await client_session.post(
                f'http://127.0.0.1:{basePort + int(assignedClient[1:])}/train',
                files={
                    'to_train': pickle.dumps(placeholdersToTrain),
                    'next_client': pickle.dumps(nextClient)
                }
            )
    return trainResult.json()


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


@app.post('/currentGlobalModelParams')
async def currentGlobalModelParams(request: Request):
    global glb_model
    serializ_glb_model_params = await request.body()
    buffer = io.BytesIO(serializ_glb_model_params)
    glb_model.load_state_dict(torch.load(buffer))
    return {"message": "Model updated with latest global parameters."}


def send_generateEncryptContext_request(clients): 
    global context 

    clonedClients = clients.copy()
    randIdx = random.randint(0, len(clonedClients)-1)
    selectedClient = clonedClients[randIdx]
    clonedClients.pop(randIdx)

    response = requests.post(f'http://127.0.0.1:{basePort + selectedClient}/generateEncryptContext', files={"clients": pickle.dumps(clonedClients)})
    context = ts.context_from(response.content, n_threads=2)


def computePlaceholders(clients, serialz_clients_enc_info):
    global context 

    clients_enc_info = []
    for serialz_client_enc_info in serialz_clients_enc_info:
        c_enc_info = [[ts.ckks_vector_from(context, encSerialTargetVector), ts.ckks_vector_from(context, encSerialTargetAmtVector)] for encSerialTargetVector, encSerialTargetAmtVector in serialz_client_enc_info]
        clients_enc_info.append(c_enc_info)
    
    uniqueList = []
    for encTarget, encTargetAmt in list(itertools.chain(*clients_enc_info)):
        add_elem = True  
        if uniqueList:
            enc_res_list = [encTarget - unique_elem for unique_elem in uniqueList]
            
            serialized_data = pickle.dumps([vec.serialize() for vec in enc_res_list])
    
            res = requests.post(f'http://127.0.0.1:{basePort + random.choice(clients)}/decryptIntermediateComparisonResult', 
                                files={'enc_comparison_val': serialized_data, 'mapping_stage': 'True'})
    
            decrypted_results = pickle.loads(res.content) 
            if 0 in decrypted_results:  
                add_elem = False  

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
    random.shuffle(placeholders)

    placeholderTargetMapEncRealTarget = {placeholderKey: encTarget for placeholderKey, encTarget in zip(placeholders, uniqueList)}
    clientsAvailPlaceholderTargets = {}
    # placeholder to encrypted real label mapping for each client to convert the placeholder to train from server to the corresponding real label to train
    clientsPlaceholderTargetMapEncRealTarget = {}

    def process_client(clientID, clientEncTargetTargetAmtVectors):
        clientsAvailPlaceholderTargets[clientID] = []
        clientsPlaceholderTargetMapEncRealTarget[clientID] = {}
        placeholderTargetMapEncRealTargetCopy = placeholderTargetMapEncRealTarget.copy()

        for clientEncTargetTargetAmtVector in clientEncTargetTargetAmtVectors:
            # Prepare encrypted comparisons
            enc_res = {placeholder: (clientEncTargetTargetAmtVector[0] - encRealTargetInUniqueList).serialize() 
                    for placeholder, encRealTargetInUniqueList in placeholderTargetMapEncRealTargetCopy.items()}

            # Send request concurrently
            res = requests.post(
                f'http://127.0.0.1:{basePort + int(clientID[1:])}/decryptIntermediateComparisonResult', 
                files={'enc_comparison_val': pickle.dumps(enc_res), 'mapping_stage': 'True'}
            )
            client_decrypted_res = pickle.loads(res.content)

            for placeholder, r in client_decrypted_res.items():
                if r == 0:
                    clientsAvailPlaceholderTargets[clientID].append((placeholder, clientEncTargetTargetAmtVector[1])) # for server to keep track the available placeholders of client
                    clientsPlaceholderTargetMapEncRealTarget[clientID][placeholder] = clientEncTargetTargetAmtVector[0] # placeholder-to-encRealLabel map
                    del placeholderTargetMapEncRealTargetCopy[placeholder]
                    break

    # Run concurrent requests using threads
    client_ids = ['c' + str(client) for client in clients]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_client, client_ids, clients_enc_info) 


    # global counts
    sums = {}
    for client, items in clientsAvailPlaceholderTargets.items():
        for placeholder, placeholderAmt in items:
            if placeholder not in sums:
                sums[placeholder] = 0
            sums[placeholder] = sums[placeholder] + placeholderAmt

    noises = {}
    for placeholder, encPlaceholderSum in sums.items():
        # server adds random value to encrypted global count, serialized sum vector and send to client for decryption 
        noise = random.randint(126, 5234)
        noises[placeholder] = noise
        sums[placeholder] = (encPlaceholderSum + noise).serialize()

    res = requests.post(f'http://127.0.0.1:{basePort + random.choice(clients)}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(sums), 'mapping_stage': 'False'})
    res = pickle.loads(res.content)
    for placeholder, placeholderGlobalSum in res.items():
        clientDecrypted_placeholderSum = placeholderGlobalSum[0] - noises[placeholder]
        sums[placeholder] = clientDecrypted_placeholderSum


    # calculate the normalized label proportion for each client label
    for client, items in clientsAvailPlaceholderTargets.items():
        noises = {}
        clientAvailPlaceholderTarget = {}
        for i, (placeholder, placeholderAmt) in enumerate(items):
            inverseTotal = 1 / sums[placeholder]
            noise = random.random()
            noises[placeholder] = noise
            encNormalizedLabelProp = ((placeholderAmt * inverseTotal) + noise).serialize() # Normalize by dividing by the sum
            clientAvailPlaceholderTarget[placeholder] = encNormalizedLabelProp
        res = requests.post(f'http://127.0.0.1:{basePort + int(client[1:])}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(clientAvailPlaceholderTarget), 'mapping_stage': 'False'})
        clientDecrypted_normalizedLabelProp = pickle.loads(res.content)
        for i, (placeholder, noisyPlaceholderAmtProp) in enumerate(clientDecrypted_normalizedLabelProp.items()):
            clientsAvailPlaceholderTargets[client][i] = (placeholder, noisyPlaceholderAmtProp[0] - noises[placeholder])  # Assign decrypted normalized value back

    return sums, clientsAvailPlaceholderTargets, clientsPlaceholderTargetMapEncRealTarget


def send_PlaceholderMapToRealLabel(client, mapping):
    serialized_data = {placeholder: encRealLabel.serialize() for placeholder, encRealLabel in mapping.items()}
    status = requests.post(f'http://127.0.0.1:{basePort + int(client[1:])}/placeholderToRealLabelMapping', files={'mapping':pickle.dumps(serialized_data)})
    return status 


async def start_federated_learning():
    epochs = args.epochs
    clients = list(range(1, args.client_num + 1))

    send_generateEncryptContext_request(clients) 
    serialz_clients_enc_info = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda client: pickle.loads(requests.post(f'http://127.0.0.1:{basePort + int(client)}/encryptLabels').content), clients))
    serialz_clients_enc_info.extend(results)

    sums, clientsAvailPlaceholderTargets, clientsPlaceholderTargetMapEncRealTarget = computePlaceholders(clients, serialz_clients_enc_info)
    for client, mapping in clientsPlaceholderTargetMapEncRealTarget.items():    
        send_PlaceholderMapToRealLabel(client, mapping) 

    st = time.time()
    for epoch in range(1, epochs+1):
        # call init()
        targets, clientsAvailPlaceholderTargets, consecutiveClientsInQueue = initVars(sums, clientsAvailPlaceholderTargets)
        probSelectedClientsAvailForSort_forEachTarget = getCurrentClientsAvailByPlaceholder(clientsAvailPlaceholderTargets)  # all clients are considered for probabilistic selection at first iteration
        firstClient = True

        await send_prepareTrainData_request(clients)
        
        while (len(targets) != 0) or (len(consecutiveClientsInQueue) != 0):
            print(f'Remaining Global Training Labels To Train: {len(targets)}')
            # print(f'Remaining Global Training Labels To Train: {Counter(targets)}')
            # continously monitor the assigned clients in queue, whenever there are less than 2 clients in queue, assign clients for the next window targets to train so that the last client in current window knows which next client to send the parameters to
            if (len(targets)!=0) and (len(consecutiveClientsInQueue) <= 2):
                while (len(consecutiveClientsInQueue) <= 2):
                    windowTargets = targets[0:30] # extract the labels subset from global training labels list
                    targets = targets[30:] # update global training labels list
                    
                    sortedClients_forEachWindowTarget = sortClients(windowTargets, clientsAvailPlaceholderTargets, probSelectedClientsAvailForSort_forEachTarget) # assign the suitable client to train on a target in such a way that minimize model parameters passing 
                    
                    windowTargets_selectedClients = [value[0] for value in sortedClients_forEachWindowTarget.values()] # get the selected first client for each target of window
                    consecutiveClientsAndTargets = groupConsecutiveClients(windowTargets_selectedClients, windowTargets) # group consecutive clients together 
                    consecutiveClientsInQueue.extend(consecutiveClientsAndTargets) # add clients with their respective targets to queue for training
        
                    probSelectedClientsAvailForSort_forEachTarget = updateProbabilisticSelectedClientsLists(windowTargets_selectedClients, windowTargets, clientsAvailPlaceholderTargets) # update probabilistic selected clients list for next window targets sorting
                    
                    if (len(targets)==0):
                        break

            current_client = consecutiveClientsInQueue.pop(0)
            current_client['next_assigned_client'] = 's0' if (len(consecutiveClientsInQueue) == 0) else consecutiveClientsInQueue[0]['assigned_client'] # Server prepares info to send current client in queue: if the assigned client is the last one in queue, assign server address as the endpoint to pass the training completed model else provide the next client address to the current client

            if firstClient:
                buffer = io.BytesIO()
                torch.save(glb_model.state_dict(), buffer, _use_new_zipfile_serialization=False)
                buffer.seek(0)
                requests.post(
                    f"http://127.0.0.1:{basePort + int(current_client['assigned_client'][1:])}/currentGlobalModelParams",
                    headers={"Content-Type": "application/octet-stream"},
                    data=buffer.getvalue()
                )
                buffer.close() 
                firstClient = False
            
            res = await send_localTrain_request(current_client['assigned_client'], current_client['target_to_train'], current_client['next_assigned_client'])
            target_to_remove = res['target exhausted']
            target_data_noTrain = res['target no train']


            if target_to_remove:
                for target in target_to_remove:
                    clientsAvailPlaceholderTargets[current_client['assigned_client']] = [placeholderPropTuple for placeholderPropTuple in clientsAvailPlaceholderTargets[current_client['assigned_client']] if placeholderPropTuple[0] != target]
                    
                    for clientInQueue in consecutiveClientsInQueue[:]: # if the queue contains current client and got assignned to train on the label it has exhausted, remove the client from queue and add the target labels back to global training labels list
                        if (clientInQueue['assigned_client'] == current_client['assigned_client']) and (target in clientInQueue['target_to_train']):
                            for targetToIncludeBackOcc in range(clientInQueue['target_to_train'].count(target)):
                                targets.insert(random.randint(0, len(targets)), target)
                                # targetsReassign += 1
                                print(f'target no train: {target}')
                            newClientTargetToTrain = [t for t in clientInQueue['target_to_train'] if t != target]
                            if len(newClientTargetToTrain) == 0:
                                consecutiveClientsInQueue.remove(clientInQueue) 
                            else:
                                clientInQueue['target_to_train'] = newClientTargetToTrain
                    
                # re-update probabilistic selected clients list for next window labels because there is changes in selected clients for the current window targets and the main clients available targets dictionary
                probSelectedClientsAvailForSort_forEachTarget = updateProbabilisticSelectedClientsLists(windowTargets_selectedClients, windowTargets, clientsAvailPlaceholderTargets)

            if target_data_noTrain: 
                for targetToIncludeBack in target_data_noTrain: # add targets not train back to the targets list random position
                    targets.insert(random.randint(0, len(targets)), targetToIncludeBack) 
                    # targetsReassign += 1

        et = time.time()
        epoch_acc = {"epoch": epoch, "result": []}
        clientsTestResult = await send_computeTestDataAcc_request(clients)
        epoch_acc['result'] = clientsTestResult

        epoch_avg_result = get_metrics_average(epoch_acc['result'])
        with open(args.resultFilePath, 'a') as file:
            # file.write(str(epoch_acc) + '\n')
            file.write(f"Epoch {epoch}:\n")
            file.write(epoch_avg_result)
            file.write("\n")
        
        print(f'total train time: {et - st}')
            
    model_params = pickle.dumps(glb_model.state_dict())
    # Save the serialized model parameters to a file in the current directory
    with open(f"SingleSampleLearning-ModelParams-ClassHold{args.labelOrDomainPerClientHold}-DirAlpha{args.dirichlet}-ClientNum{args.client_num}.pkl", 'wb') as f:
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
    await start_federated_learning()

    await server_task
        
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(begin())
