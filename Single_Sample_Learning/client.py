import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import uvicorn
from fastapi import FastAPI, Response,  File, UploadFile
import pickle

from Model_Architecture.LeNet import *
from nonIID_dataPartition import *
import os 
import signal
from itertools import islice

import numpy as np
import torch.optim as optim

import tenseal as ts 

device = torch.device('cuda')

model = LeNet()
optimizer = optim.Adam(model.parameters(), lr=0.0001, eps=1e-6)
criterion = nn.CrossEntropyLoss()  # Equivalent to sparse_categorical_crossentropy        

epoch_total_correct = 0
epoch_total_samples = 0
train_losses = [] 
train_labels = []
train_preds = []

context = None
PlaceholderMaptoRealTarget = None ##

def create_flask_app(rank, port, clientTrainData, clientTestData, clientTrainLabelDataCount):
    app = FastAPI() 

    @app.route('/federatedLearningCompleted', methods=['POST'])
    def federatedLearningCompleted():
        os.kill(os.getpid(), signal.SIGINT)

    trainDataByLabels = {}
    testData = clientTestData


    def preprocess_trainImage(img):
        # transform_train = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        # ])
        # transformed_img = transform_train(img)
        transformed_img = img / 255.0
        return transformed_img

    def preprocess_testImage(img):
        # transform_test = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])
        # transformed_img = transform_test(img)
        transformed_img = img / 255.0
        return transformed_img
    
    # change name to start of new epoch & get batch size from server
    @app.post('/prepareTrainData')
    def prepare_trainData():
        clientData_copy = clientTrainData.copy()
        clientData_copy['images'] = clientData_copy['images'].apply(preprocess_trainImage)
        # convert client data by label to each data-label generator
        for label in clientData_copy['labels'].unique():
            trainDataByLabels[label] = iter(clientData_copy.loc[clientData_copy['labels'] == label, ['images', 'labels']].sample(frac=1, replace=False).itertuples(index=False, name=None))
        
        return {"message": f"Train data is ready by process {rank}"}

    def prepare_testData():
        # global testData
        testData['images'] = testData['images'].apply(preprocess_testImage)
    
    prepare_testData()

    @app.route('/generateEncryptContext', methods=['POST']) ##
    async def generate_EncryptContext(clients: UploadFile = File(...)):
        global context 

        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=2)
        context.generate_galois_keys()
        context.global_scale = 2**40
        
        clientsSerialzContext = context.serialize(save_secret_key=True) # all clients need to use the same context for the server to perform operations on compatible encrypted contents
        serverSerialzContext = context.serialize(save_secret_key=True) # context without secret key so that the server will not be able to decrypt the clients encrypted contents but with the context to perform operations on the encrypted contents
        
        clientsAddrs = pickle.loads(await clients.read())

        with ThreadPoolExecutor() as executor: 
            futures = [executor.submit(lambda client: requests.post(f'http://127.0.0.1:{5000 + int(client)}/receiveEncryptContext', files={"serialized_client_context": clientsSerialzContext}), client) for client in clientsAddrs] ##

            for future in as_completed(futures):
                response = future.result()
                print(response.json())

        # return jsonify({"message": "Encryption Context Generated.", "serialized_server_context": serverSerialzContext}), 200
        return Response(serverSerialzContext, content_type='application/octet-stream')
    

    @app.post('/receiveEncryptContext') 
    async def receive_EncryptionContext(serialized_client_context: UploadFile = File(...)):
        global context
        context = ts.context_from(await serialized_client_context.read(), n_threads=2)
        return {"message": "received context"}
    
    

    @app.post('/decryptIntermediateComparisonResult') 
    async def decryptComparisonResult(enc_comparison_val: UploadFile = File(...)):
        global context
        encComparisonValue = ts.ckks_vector_from(context, pickle.loads(await enc_comparison_val.read()))
        comparisonValue = encComparisonValue.decrypt()
        return Response(pickle.dumps(comparisonValue), content_type='application/octet-stream')


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


    @app.post('/currentGlobalModelParams')
    async def currentGlobalModelParams():
        global model
        serializ_glb_model_params = await glb_model_params.read()
        glb_model_params = pickle.loads(serializ_glb_model_params)
        model.load_state_dict(glb_model_params)
        return {"message": "Model updated with latest global parameters."}


    def model_performance(labels, preds, totalTargetClass):
        conf_matrix = np.zeros((totalTargetClass, totalTargetClass), dtype=int)
        for pred, label in zip(preds, labels):
            conf_matrix[label, pred] += 1

        # Initialize lists to store precision, recall, and F1 scores for each class
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

        # Calculate metrics for each class
        for i in range(conf_matrix.shape[0]):
            TP = conf_matrix[i, i]  # True Positives
            FP = conf_matrix[:, i].sum() - TP  # False Positives
            FN = conf_matrix[i, :].sum() - TP  # False Negatives

            # Precision: TP / (TP + FP)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

            # Recall: TP / (TP + FN)
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Append results
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

        total_correct = conf_matrix.diagonal().sum()  # Sum of all true positives
        total_samples = conf_matrix.sum()  # Total number of predictions
        normal_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # Calculate Macro-Averaged Precision, Recall, and F1
        macro_precision = np.mean([p for p in precision_per_class if p != 0])
        macro_balanceAcc = np.mean([r for r in recall_per_class if r != 0])
        macro_f1 = np.mean([f for f in f1_per_class if f != 0])

        return conf_matrix, precision_per_class, recall_per_class, f1_per_class, normal_accuracy, macro_precision, macro_balanceAcc, macro_f1


    @app.route('/test', methods=['POST'])
    def test():
        # global model
        global model, epoch_total_correct, epoch_total_samples, train_losses, train_labels, train_preds

        glbModelParam = pickle.loads(requests.get('http://127.0.0.1:5000/get_glb_params').content)
        model.load_state_dict(glbModelParam)
        model.eval()
        
        test_losses = []
        test_labels = []
        test_preds = []
        total = len(testData)
        correct = 0
        with torch.no_grad():
            for i in range(len(testData)):
                input = torch.from_numpy(testData.iloc[i]['images']).permute(2, 0, 1).unsqueeze(0).float().to(device)
                label = torch.tensor([testData.iloc[i]['labels']]).to(device)
                pred = model(input)
                _, predicted = pred.max(1)
                correct += predicted.eq(label).sum().item()
                test_losses.append(criterion(pred, torch.tensor([label]).to(device)).item())
                test_labels.append(label.item())
                test_preds.append(predicted.item())


        # epoch_testAcc = 100. * correct / total
        # epoch_trainAcc = 100. * epoch_total_correct / epoch_total_samples

        # # reset train accuracy trackers back to 0
        # epoch_total_correct = 0
        # epoch_total_samples = 0 
        train_losses = []
        train_labels = []
        train_preds = []

        conf_matrix, precision_per_class, recall_per_class, f1_per_class, accuracy, macro_precision, macro_balanceAcc, macro_f1 = model_performance(train_labels, train_preds, 10) ##num_classes
        test_conf_matrix, test_precision_per_class, test_recall_per_class, test_f1_per_class, test_accuracy, test_macro_precision, test_macro_balanceAcc, test_macro_f1 = model_performance(test_labels, test_preds, 10) ##num_classes


        # return jsonify({f"client {rank}": {"Train Acc": epoch_trainAcc, "test correct": correct, "test total": total, "Test Acc": epoch_testAcc}})
        return jsonify({f"client {rank}": {"Train confusion matrix": conf_matrix.tolist(), 
                                           "Train precision per class": precision_per_class, 
                                           "Train recall per class": recall_per_class, 
                                           "Train f1 per class": f1_per_class,
                                           "Train accuracy": accuracy,
                                           "Train macro precision": macro_precision,
                                           "Train macro balanced accuracy": macro_balanceAcc,
                                           "Train macro f1": macro_f1,
                                           "Train avg loss": np.mean(train_losses),
                                           "Test confusion matrix": test_conf_matrix.tolist(), 
                                           "Test precision per class": test_precision_per_class, 
                                           "Test recall per class": test_recall_per_class, 
                                           "Test f1 per class": test_f1_per_class,
                                           "Test accuracy": test_accuracy,
                                           "Test macro precision": test_macro_precision,
                                           "Test macro balanced accuracy": test_macro_balanceAcc,
                                           "Test macro f1": test_macro_f1,
                                           "Test avg loss": np.mean(test_losses),
                                           }})


    @app.route('/train', methods=['POST'])
    def train():
        global model, criterion, epoch_total_correct, epoch_total_samples, PlaceholderMaptoRealTarget, train_losses, train_labels, train_preds
        model.train().to(device)
        target_exhausted = []
        target_noTrain = []

        print('in clinet train func')
        # receive target to train and next assigned client address from server
        to_train = pickle.loads(request.files['to-train'].read())
        nextClient = pickle.loads(request.files['next-client'].read())
        print(f'to_train: {to_train}, nextClient: {nextClient}')

        # train model on the target data required by the server in sequence 
        for targetPlaceholder in to_train:
            targetRealLabel = PlaceholderMaptoRealTarget[targetPlaceholder]
            sample = list(islice(trainDataByLabels[targetRealLabel], 1))
            if sample:
                x, y = sample[0]
                optimizer.zero_grad()
                output = model(torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device))
                loss = criterion(output, torch.tensor([y]).to(device))
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output, 1)
                # epoch_total_correct = epoch_total_correct + (predicted == torch.tensor([y]).to(device)).sum().item()
                # epoch_total_samples = epoch_total_samples + torch.tensor([y]).size(0)
                train_labels.append(y)
                train_preds.append(predicted.item())
                train_losses.append(loss.item())

            else:
                target_exhausted.append(targetPlaceholder)
                target_noTrain.append(targetPlaceholder) 
        
        
        # send updated model param to next assigned client
        serialz_glb_model_params = pickle.dumps(model.state_dict())
        requests.post(f'http://127.0.0.1:{5000 + int(nextClient[1:])}/currentGlobalModelParams', data={'global_model_params': serialz_glb_model_params})

        # return target exhausted and target no train list back to server
        return {'target exhausted': list(set(target_exhausted)), 'target no train': target_noTrain}
    

    # Start the Flask app on the assigned port
    uvicorn.run(app, port=port)  


def start_training_process(rank, port, clientTrainData, clientTestData, clientTrainLabelDataCount):
    print(f"Process {rank} started. Running Flask app on port {port}")
    create_flask_app(rank, port, clientTrainData, clientTestData, clientTrainLabelDataCount)


if __name__ == "__main__":
    num_processes = 20 # total clients
    base_port = 5001  # Starting port
    mp.set_start_method('spawn', force=True)

    # Spawn multiple processes, each with its own Flask app
    processes = []
    for rank in range(num_processes):
        port = base_port + rank
        p = mp.Process(target=start_training_process, args=(rank, port, clients_traindf[rank], clients_testdf[rank], clientsTrainLabelDataCount[rank]))
        p.start()
        processes.append(p)

    # Optionally join processes (this will block the main process)
    for p in processes:
        p.join()