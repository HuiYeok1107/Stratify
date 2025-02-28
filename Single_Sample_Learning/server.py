import asyncio
import uvicorn
from fastapi import FastAPI, Request, Response, File, UploadFile
import httpx
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pickle
import random
import os 
import signal

from itertools import groupby
import itertools
from itertools import groupby
from itertools import chain 
from collections import Counter 
import numpy as np

from Model_Architecture.LeNet import *


import tenseal as ts
context = None 

app = FastAPI()

# create global model
glb_model = LeNet()

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
        tasks = [client_session.post(f"http://127.0.0.1:{5000 + client}/prepareTrainData") for client in clients]
        # Wait for all tasks to complete concurrently
        responses = await asyncio.gather(*tasks)

    return {"status": "Prepare training data signals sent"}

# def send_prepareTrainData_request(client):
#     print(f'clinet: {client}')
#     status = requests.post(f'http://127.0.0.1:{5000 + client}/prepareTrainData') 
#     return status 


# Dynamic sub-window.  Initial sub-window consider only the clients list for current i. To determine whether to increase the upper and lower bound of a sub-window
    # Increase upper bound: if any clients of current i exist in the list before i (i-1)
    # Increase lower bound: if the top freq client(s) of current i for the current upper and lower bound sub-window exist in the next list after i (i+1) and 
                            # continue increase when the updated current upper and lower bound sub-window top freq client(s) exist in subsequent consecutive lists (i+1+x) 
def sortClients(windowTargets, clientsLabels, probSelectedClientsAvailForSort_forEachTarget): ##
    print("")
    print('in sortClients func')
    # print(f'current clients avail by placeholder: {probSelectedClientsAvailForSort_forEachTarget}')
    # for each target (label/region) in window, identify which clients have data on that target
    listOfClientsWithAvailTarget_atWindowI = {target_position: [] for target_position in range(0, len(windowTargets))}
    for i, target in enumerate(windowTargets):
        for ckey, cTargetAmountAvailTuple in clientsLabels.items():
            if cTargetAmountAvailTuple: # avoid indexing empty list
                if (target in list(zip(*cTargetAmountAvailTuple))[0]) and (ckey in list(zip(*probSelectedClientsAvailForSort_forEachTarget[target]))[0]):  
                    listOfClientsWithAvailTarget_atWindowI[i].append(ckey)

    print(f'clinets avail at window i target: {listOfClientsWithAvailTarget_atWindowI}')
    print(f'windowTargets: {windowTargets}')
    print(f'clientsLabels: {clientsLabels}')
    print(f'probSelectedClientsAvailForSort_forEachTarget: {probSelectedClientsAvailForSort_forEachTarget}')
    
    windowI_subWindow_clientsCoverage = {}
    sortedClientsForEachWindowTargetI = {}
    for i in range(0, len(listOfClientsWithAvailTarget_atWindowI)):
        # print(f'current i: {i, listOfClientsWithAvailTarget_atWindowI[i]}')
        expandSubWindowLowerBound = True
        lowerSubWindowBound = 0
        start_idx = i 
        end_idx = i

        upperSubWindowBound = max(0, i-1) # prevent out-of-bound upper sub-window
        if set(listOfClientsWithAvailTarget_atWindowI[upperSubWindowBound]) & set(listOfClientsWithAvailTarget_atWindowI[i]): # increase upper bound by i-1 if the clients in i exist in i-1 list
            start_idx = upperSubWindowBound
        # print(f'start: {start_idx}')

        while expandSubWindowLowerBound:
            windowI_subWindow_clientsCoverage[i] = Counter(list(chain(*list(listOfClientsWithAvailTarget_atWindowI.values())[start_idx: end_idx+1]))) # calculate clients freq for current upper and lower bounds sub-window
            lowerSubWindowBound += 1
            lowerSubWindowBound = min(len(listOfClientsWithAvailTarget_atWindowI)-1-i, lowerSubWindowBound) # prevent out-of-bound lower sub-window
            # if set(listOfClientsWithAvailTarget_atWindowI[i]) & set(listOfClientsWithAvailTarget_atWindowI[i+lowerSubWindowBound]):
            print(f'windowI_subWindow_clientsCoverage: {windowI_subWindow_clientsCoverage[i]}')   
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
            # else:
            #     expandSubWindowLowerBound = False
        #     print(f'end: {end_idx}')
        #     print(list(listOfClientsWithAvailTarget_atWindowI.values())[start_idx: end_idx+1])

        # print(windowI_subWindow_clientsCoverage[i])

        sortedClientsForEachWindowTargetI[i] = sorted(listOfClientsWithAvailTarget_atWindowI[i], key=lambda client: windowI_subWindow_clientsCoverage[i][client], reverse=True)
        print(f"window {i} sorted clients: {sortedClientsForEachWindowTargetI[i]}")
        print('')
    return sortedClientsForEachWindowTargetI


def groupConsecutiveClients(windowTargets_selectedClients, windowTargets):
    # group consecutive identical client ID
    grouped_by_consecutive_clients = [list(group) for key, group in groupby(windowTargets_selectedClients)]
    consecutiveClientsAndTargetsList = []
    print(grouped_by_consecutive_clients)
    print(windowTargets)
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
            # print(windowTarget, 'in')
            clientsAvailForSort_forEachTarget[windowTarget] = [client_amt_tuple for client_amt_tuple in clientsAvailForSort_forEachTarget[windowTarget] if client_amt_tuple[0] != clientAssigned]
    
    for placeholder, clientPropTuples in clientsAvailForSort_forEachTarget.items(): # for each placeholder's clients available list, include or exclude this client from the list based on the client's normalized placeholder proportion among the clients. This is to avoid having only the client with higher amount of label data remaining near the end of the placeholders list, and in which leads to model overfitting to this client domains data.
        # clients = []
        # while len(clients) == 0:
        #     clients = [(client, prop) for client, prop in clientPropTuples if random.choices([True, False], [prop, 1-prop])[0]] 
        # clientsAvailForSort_forEachTarget[placeholder] = clients

        selectedClients = list(set(random.choices(list(list(zip(*clientPropTuples))[0]), weights=list(list(zip(*clientPropTuples))[1]), k=len(clientPropTuples))))  # NEW CODE LOGIC TO TEST
        clientsAvailForSort_forEachTarget[placeholder] = [(client, prop) for client, prop in clientPropTuples if client in selectedClients]
        
    
    print('')
    print(f'Prob selected client for next win: {clientsAvailForSort_forEachTarget}')
    print('')
    
    return clientsAvailForSort_forEachTarget # return each placeholder clients available list prepared for clients sorting 

async def send_localTrain_request(assignedClient, placeholdersToTrain, nextClient): 
    async with httpx.AsyncClient(timeout=None) as client_session:
        trainResult = await client_session.post(
                f'http://127.0.0.1:{5000 + int(assignedClient[1:])}/train',
                files={
                    'to-train': pickle.dumps(placeholdersToTrain),
                    'next-client': pickle.dumps(nextClient)
                }
            )
    return trainResult


# # add in to client side, send model to next client
# # if is first client, send initialized global model and next client address; if is last client, send server address
# def send_localTrain_request(assignedClient, placeholdersToTrain, nextClient):
#     trainResult = requests.post(f'http://127.0.0.1:{5000 + int(assignedClient[1:])}/train', files={'to-train': pickle.dumps(placeholdersToTrain), 'next-client': pickle.dumps(nextClient)})
#     return trainResult

async def send_computeTestDataAcc_request(clients):
    async with httpx.AsyncClient(timeout=None) as client_session:
        tasks = [client_session.post(f'http://127.0.0.1:{basePort + client}/test') for client in clients]
        results = await asyncio.gather(*tasks)
        epochResults = [pickle.loads(result.content) for result in results]
    return epochResults

# def send_computeTestDataAcc_request(client):
#     result = requests.post(f'http://127.0.0.1:{5000 + client}/test')
#     return result

async def send_trainingCompleted_signal(clients):
    async with httpx.AsyncClient(timeout=None) as client_session:
        tasks = [client_session.post(f"http://127.0.0.1:{basePort + client}/federatedLearningCompleted") for client in clients]
        responses = await asyncio.gather(*tasks)
    return {"status": "Training completed signals sent"}


# def send_completedTraining_signal(client):
#     status = requests.post(f'http://127.0.0.1:{5000 + client}/federatedLearningCompleted')
#     return status

def updateClientsPool(data, clientsAvailPlaceholderTargets, targets, consecutiveClientsInQueue, current_client, windowTargets_selectedClients, windowTargets):
    target_to_remove = data['target exhausted']
    target_data_noTrain = data['target no train']
    # unpack target to remove and target not train lists from client
    if target_to_remove:
        for target in target_to_remove:
            clientsAvailPlaceholderTargets[current_client['assigned_client']] = [placeholderPropTuple for placeholderPropTuple in clientsAvailPlaceholderTargets[current_client['assigned_client']] if placeholderPropTuple[0] != target]

            for clientInQueue in consecutiveClientsInQueue[:]: ##TO-DO: update code in github if the queue contains current client and got assignned to train on the label it has exhausted, remove the client from queue and add the target labels back to targets list
                if (clientInQueue['assigned_client'] == current_client['assigned_client']) and (target in clientInQueue['target_to_train']):
                    for targetToIncludeBackOcc in range(clientInQueue['target_to_train'].count(target)):
                        targets.insert(random.randint(0, len(targets)), target)
                        # targetsReassign += 1
                        print(f'target no train: {target}')
                    newClientTargetToTrain = [t for t in clientInQueue['target_to_train'] if t != target]
                    if len(newClientTargetToTrain) == 0:
                        consecutiveClientsInQueue.remove(clientInQueue) ##TO-DO: update code in github
                    else:
                        clientInQueue['target_to_train'] = newClientTargetToTrain
        
    # re-update probabilistic selected clients list for next window targets sorting because there is changes in selected clients for the current window targets and the main clients available targets dictionary
    probSelectedClientsAvailForSort_forEachTarget = updateProbabilisticSelectedClientsLists(windowTargets_selectedClients, windowTargets, clientsAvailPlaceholderTargets)

    if target_data_noTrain: 
        print(f'target no train: {target_data_noTrain}')
        for targetToIncludeBack in target_data_noTrain: # add targets no train back to the targets list random position
            targets.insert(random.randint(0, len(targets)), targetToIncludeBack) 
            # targetsReassign += 1

@app.post('/currentGlobalModelParams')
async def currentGlobalModelParams(global_model_params, UploadFile=File(...)):
    global glb_model
    glb_model_params = pickle.loads(await global_model_params.read())
    glb_model.load_state_dict(glb_model_params)
    return {'message': "Model updated with latest global parameters."}


def send_generateEncryptContext_request(clients): ##
    global context 
    selectedClient = random.choice(clients)
    clients.remove(selectedClient)
    response = requests.post(f'http://127.0.0.1:{5000 + selectedClient}/generateEncryptContext', files={'clients': pickle.dumps(clients)})
    clients.append(selectedClient)
    clients.sort()
    print(f'in send context clients: {clients}')
    # data = response.json()
    data = response.content
    context = ts.context_from(data)

    # return response.status_code 

def computePlaceholders(serialz_clients_enc_info):
    global context 

    clients_enc_info = []
    for serialz_client_enc_info in serialz_clients_enc_info:
        c_enc_info2 = [[ts.ckks_vector_from(context, encSerialTargetVector), ts.ckks_vector_from(context, encSerialTargetAmtVector)] for encSerialTargetVector, encSerialTargetAmtVector in serialz_client_enc_info]
        
        clients_enc_info.append(c_enc_info2)
    
    uniqueList = []
    for encTarget, encTargetAmt in list(itertools.chain(*clients_enc_info)):
        add_elem = True  
        if uniqueList:
            for unique_elem in uniqueList:
                enc_res = encTarget - unique_elem
                # * Server needs to send the enc_res back to the client for decryption and get back the result
                # serialized_enc_res = enc_res.serialize()
                # client_decrypted_res = ts.ckks_vector_from(context, serialized_enc_res).decrypt()
                res = requests.post(f'http://127.0.0.1:{5000 + 1}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(enc_res.serialize())})
                client_decrypted_res = pickle.loads(res.content)

                # print(f"Res: {client_decrypted_res}")
                if abs(client_decrypted_res[0]) < 1e-5:  # Check if client_decrypted_res is close to zero
                    add_elem = False 
                    break
        if add_elem:
            # add a small encrypted value into the encrypted target to prevent transparent ciphertext issue in later comparison stage 
            encTarget = encTarget + ts.ckks_vector(context, [0.0000001]) 
            uniqueList.append(encTarget)
    
    placeholders = range(0, 200) #list(string.ascii_uppercase)
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
                # client_decrypted_res = enc_res.decrypt()

                res = requests.post(f'http://127.0.0.1:{5000 + int(clientID[1:])}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(enc_res.serialize())})
                client_decrypted_res = pickle.loads(res.content)

                if abs(client_decrypted_res[0]) < 1e-5:
                    print('in')
                    clientsAvailPlaceholderTargets[clientID].append((placeholder, clientEncTargetTargetAmtVector[1]))   
                    clientsPlaceholderTargetMapEncRealTarget[clientID][placeholder] = clientEncTargetTargetAmtVector[0]
                    break   

    print(clientsPlaceholderTargetMapEncRealTarget)

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

        res = requests.post(f'http://127.0.0.1:{5000 + int(clientID[1:])}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(encPlaceholderSum.serialize())})
        clientDecrypted_placeholderSum = pickle.loads(res.content)[0] - noise
        sums[placeholder] = clientDecrypted_placeholderSum

        # serializedEncPlaceholderSum = encPlaceholderSum.serialize()
        # clientDecrypted_placeholderSumWithNoise = ts.ckks_vector_from(context, serializedEncPlaceholderSum).decrypt()[0] 
        # clientDecrypted_placeholderSum = clientDecrypted_placeholderSumWithNoise - noise
        # sums[placeholder] = clientDecrypted_placeholderSum
    # End.

    # calculate the normalized label proportion for each client label
    for client, items in clientsAvailPlaceholderTargets.items():
        for i, (placeholder, placeholderAmt) in enumerate(items):
            inverseTotal = 1 / sums[placeholder]
            encNormalizedLabelProp = placeholderAmt * inverseTotal  # Normalize by dividing by the sum

            res = requests.post(f'http://127.0.0.1:{5000 + int(client[1:])}/decryptIntermediateComparisonResult', files={'enc_comparison_val': pickle.dumps(encNormalizedLabelProp.serialize())})
            clientDecrypted_normalizedLabelProp = pickle.loads(res.content)[0]

            # serialEncNormalizedLabelProp = encNormalizedLabelProp.serialize() # server serialized encrypted normalized value result and send to the respective client for decryption
            # clientDecrypted_normalizedLabelProp = ts.ckks_vector_from(context, serialEncNormalizedLabelProp).decrypt()[0] # client decrypt encrypted result and send back to server
            clientsAvailPlaceholderTargets[client][i] = (placeholder, clientDecrypted_normalizedLabelProp)  # Assign decrypted normalized value back

    print(clientsAvailPlaceholderTargets)
    print(sums)
    return sums, clientsAvailPlaceholderTargets, clientsPlaceholderTargetMapEncRealTarget

def send_PlaceholderMapToRealLabel(client, mapping):
    serialized_data = {placeholder: encRealLabel.serialize() for placeholder, encRealLabel in mapping.items()}
    status = requests.post(f'http://127.0.0.1:{5000 + int(client[1:])}/placeholderToRealLabelMapping', files={'mapping':pickle.dumps(serialized_data)})
    return status 

async def start_federated_learning():
    epochs = 1
    clients = list(range(1, 20+1))

    send_generateEncryptContext_request(clients) ##

    serialz_clients_enc_info = []
    for client in clients: # send the request using loop instead of threading because the server needs to know the encrypted label is sent from which client so that the assigned placeholder can be sent to the correct client in later stage
        response = requests.post(f'http://127.0.0.1:{5000 + int(client)}/encryptLabels')
        serialz_client_enc_info = pickle.loads(response.content)
        serialz_clients_enc_info.append(serialz_client_enc_info)

    sums, clientsAvailPlaceholderTargets, clientsPlaceholderTargetMapEncRealTarget = computePlaceholders(serialz_clients_enc_info) ##
    print(f'clientsPlaceholderTargetMapEncRealTarget: {clientsPlaceholderTargetMapEncRealTarget}')
    for client, mapping in clientsPlaceholderTargetMapEncRealTarget.items():    
        send_PlaceholderMapToRealLabel(client, mapping) ##

    for epoch in range(1, epochs+1):
        # os.kill(os.getpid(), signal.SIGINT)
        # call init()
        targets, clientsAvailPlaceholderTargets, consecutiveClientsInQueue = initVars(sums, clientsAvailPlaceholderTargets)
        probSelectedClientsAvailForSort_forEachTarget = getCurrentClientsAvailByPlaceholder(clientsAvailPlaceholderTargets)  # all clients are considered for probabilistic selection at first iteration
        firstClient = True

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(send_prepareTrainData_request, client) for client in clients]

            for future in as_completed(futures):
                response = future.result()
                print(response.json())
        
        while (len(targets) != 0) or (len(consecutiveClientsInQueue) != 0):
            print(f'length targets: {Counter(targets)}')
            # continously monitor the assigned clients in queue, whenever there are less than 2 clients in queue, assign clients for the next window targets to train so that the last client in current window knows which next client to send the parameters to
            if (len(targets)!=0) and (len(consecutiveClientsInQueue) <= 2):
                while (len(consecutiveClientsInQueue) <= 2):
                    windowTargets = targets[0:15] # extract the current 10 labels from list
                    targets = targets[15:] # updated targets list
                    
                    sortedClients_forEachWindowTarget = sortClients(windowTargets, clientsAvailPlaceholderTargets, probSelectedClientsAvailForSort_forEachTarget) # assign the suitable client to train on a target in such a way that minimize model parameters passing 
                    
                    windowTargets_selectedClients = [value[0] for value in sortedClients_forEachWindowTarget.values()] # get the selected first client for each target of window
                    consecutiveClientsAndTargets = groupConsecutiveClients(windowTargets_selectedClients, windowTargets) # group consecutive clients together 
                    consecutiveClientsInQueue.extend(consecutiveClientsAndTargets) # add clients with their respective targets to queue for training
        
                    probSelectedClientsAvailForSort_forEachTarget = updateProbabilisticSelectedClientsLists(windowTargets_selectedClients, windowTargets, clientsAvailPlaceholderTargets) # update probabilistic selected clients list for next window targets sorting
            
                    print(f'client in queue: {consecutiveClientsInQueue}')
                    
                    if (len(targets)==0):
                        break

            current_client = consecutiveClientsInQueue.pop(0)
            current_client['next_assigned_client'] = 's0' if (len(consecutiveClientsInQueue) == 0) else consecutiveClientsInQueue[0]['assigned_client'] # Server prepares info to send current client in queue: if the assigned client is the last one in queue, assign server address as the endpoint to pass the training completed model else provide the next client address to the current client

            if firstClient:
                serialz_glb_model_params = pickle.dumps(glb_model.state_dict())
                requests.post(f"http://127.0.0.1:{5000 + int(current_client['assigned_client'][1:])}/currentGlobalModelParams", files={'glb-model-params': serialz_glb_model_params})
                res = send_localTrain_request(current_client['assigned_client'], current_client['target_to_train'], current_client['next_assigned_client'])
                res_data = res.json()
                target_to_remove = res_data['target exhausted']
                target_data_noTrain = res_data['target no train']
                firstClient = False
            else:
                res = send_localTrain_request(current_client['assigned_client'], current_client['target_to_train'], current_client['next_assigned_client'])
                res_data = res.json()
                target_to_remove = res_data['target exhausted']
                target_data_noTrain = res_data['target no train']

            # updateClientsPool(res.json(), clientsAvailPlaceholderTargets, targets, consecutiveClientsInQueue, current_client, windowTargets_selectedClients, windowTargets)
            if target_to_remove:
                for target in target_to_remove:
                    clientsAvailPlaceholderTargets[current_client['assigned_client']] = [placeholderPropTuple for placeholderPropTuple in clientsAvailPlaceholderTargets[current_client['assigned_client']] if placeholderPropTuple[0] != target]

                    # for i, clientInQueue in enumerate(consecutiveClientsInQueue[:]): ##TO-DO: update code in github if the queue contains current client and got assignned to train on the label it has exhausted, remove the client from queue and add the target labels back to targets list
                    #     if (clientInQueue['assigned_client'] == current_client['assigned_client']) and (target in clientInQueue['target_to_train']):
                    #         for targetToIncludeBack in clientInQueue['target_to_train']:
                    #             targets.insert(random.randint(0, len(targets)), targetToIncludeBack) 
                    #             targetsReassign += 1
                                
                    #         consecutiveClientsInQueue.remove(clientInQueue) ##TO-DO: update code in github
                    
                    for clientInQueue in consecutiveClientsInQueue[:]: ##TO-DO: update code in github if the queue contains current client and got assignned to train on the label it has exhausted, remove the client from queue and add the target labels back to targets list
                        if (clientInQueue['assigned_client'] == current_client['assigned_client']) and (target in clientInQueue['target_to_train']):
                            for targetToIncludeBackOcc in range(clientInQueue['target_to_train'].count(target)):
                                targets.insert(random.randint(0, len(targets)), target)
                                # targetsReassign += 1
                                print(f'target no train: {target}')
                            newClientTargetToTrain = [t for t in clientInQueue['target_to_train'] if t != target]
                            if len(newClientTargetToTrain) == 0:
                                consecutiveClientsInQueue.remove(clientInQueue) ##TO-DO: update code in github
                            else:
                                clientInQueue['target_to_train'] = newClientTargetToTrain
                    
                # re-update probabilistic selected clients list for next window targets sorting because there is changes in selected clients for the current window targets and the main clients available targets dictionary
                probSelectedClientsAvailForSort_forEachTarget = updateProbabilisticSelectedClientsLists(windowTargets_selectedClients, windowTargets, clientsAvailPlaceholderTargets)

            if target_data_noTrain: 
                print(f'target no train: {target_data_noTrain}')
                for targetToIncludeBack in target_data_noTrain: # add targets not train back to the targets list random position
                    targets.insert(random.randint(0, len(targets)), targetToIncludeBack) 
                    # targetsReassign += 1
    
        epoch_acc = {"epoch": epoch, "result": []}
        clientsTestResult = await send_computeTestDataAcc_request(clients)
        epoch_acc['result'] = clientsTestResult

        test_accs = []
        test_BA = []
        train_accs = []
        train_BA = []
        for client in epoch_acc['result']:
            client_name = list(client.keys())[0]
            test_BA.append(client[client_name]['Test macro balanced accuracy'])
            train_BA.append(client[client_name]['Train macro balanced accuracy'])
            test_accs.append(client[client_name]['Test accuracy'])
            train_accs.append(client[client_name]['Train accuracy'])

        with open("mnist-solution1.txt", 'a') as file:
            file.write(str(epoch_acc) + '\n')
            file.write('\n')
            file.write(
                f"Average metrics: Test Acc: {np.mean(test_accs)} "
                f"Test Balance Acc: {np.mean(test_BA)} "
                f"Train Acc: {np.mean(train_accs)} "
                f"Train Balance Acc: {np.mean(train_BA)}\n"
            )


    await send_trainingCompleted_signal(clients)
    os.kill(os.getpid(), signal.SIGTERM)

if __name__ == '__main__':
    threading.Thread(target=start_federated_learning, daemon=True).start()

    uvicorn.run(app, port=5000)  