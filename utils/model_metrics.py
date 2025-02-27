import numpy as np


def model_performance(labels, preds, totalTargetClass):
    conf_matrix = np.zeros((totalTargetClass, totalTargetClass), dtype=int)
    for pred, label in zip(preds, labels):
        conf_matrix[label, pred] += 1

    f1_per_class = []
    ba_per_class = []
    correct_per_class = []
    total_per_class = []

    # Calculate metrics for each class
    for i in range(conf_matrix.shape[0]):
        if conf_matrix[i, :].sum() != 0: # excluded metrics calculation for class that a client does not hold
            TP = conf_matrix[i, i]  
            FP = conf_matrix[:, i].sum() - TP  
            FN = conf_matrix[i, :].sum() - TP  
            TN = conf_matrix.sum() - (TP + FP + FN)  
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0 
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

            balanced_acc = (recall + specificity) / 2
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            f1_per_class.append(f1)
            ba_per_class.append(balanced_acc)
            correct_per_class.append(TP)
            total_per_class.append(conf_matrix[i, :].sum())
        

    total_correct = conf_matrix.diagonal().sum()  # sum of all true positives
    total_samples = conf_matrix.sum()  # total number of predictions
    normal_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Macro-Averaged 
    macro_avg_f1 = np.mean([f for f in f1_per_class])
    macro_avg_ba_allclasses = np.mean([ba for ba in ba_per_class]) # note: balance accuracy is only for dataset with 3 classes & above

    # Weighted overall ba per class (total i class sample * ba for i class + ...) / total all classes samples
    weighted_ba = np.sum([ba * total_samples for ba, total_samples in zip(ba_per_class, total_per_class)]) / np.sum(total_per_class)
    weighted_f1 = np.sum([f * total_samples for f, total_samples in zip(f1_per_class, total_per_class)]) / np.sum(total_per_class)

    conf_matrix = list([[int(value) for value in row] for row in conf_matrix.tolist()])

    return conf_matrix, normal_accuracy, macro_avg_f1, weighted_f1, macro_avg_ba_allclasses, weighted_ba, total_samples


def get_metrics_average(clients_model_metrics):
    train_accs, train_weight_accs, test_accs, test_weight_accs = [], [], [], []
    train_macro_BA, train_weight_BA, test_macro_BA, test_weight_BA = [], [], [], []
    train_macro_f1, train_weight_f1, test_macro_f1, test_weight_f1 = [], [], [], []
    total_train_samples, total_test_samples = 0, 0
    
    for client in clients_model_metrics:
        client_name = list(client.keys())[0]
        train_accs.append(client[client_name]['train_normal_accuracy'])
        train_weight_accs.append(client[client_name]['train_normal_accuracy'] * client[client_name]['train_size'])
        train_macro_BA.append(client[client_name]['train_macro_avg_ba_allclasses'])
        train_weight_BA.append(client[client_name]['train_weighted_ba'] * client[client_name]['train_size'])
        train_macro_f1.append(client[client_name]['train_macro_avg_f1'])
        train_weight_f1.append(client[client_name]['train_weighted_f1'] * client[client_name]['train_size'])

        total_train_samples += client[client_name]['train_size']
        
        test_accs.append(client[client_name]['test_normal_accuracy'])
        test_weight_accs.append(client[client_name]['test_normal_accuracy'] * client[client_name]['test_size'])
        test_macro_BA.append(client[client_name]['test_macro_avg_ba_allclasses'])
        test_weight_BA.append(client[client_name]['test_weighted_ba'] * client[client_name]['test_size'])
        test_macro_f1.append(client[client_name]['test_macro_avg_f1'])
        test_weight_f1.append(client[client_name]['test_weighted_f1'] * client[client_name]['test_size'])
        
        total_test_samples += client[client_name]['test_size']

    
    return (
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
    )