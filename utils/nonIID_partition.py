import pandas as pd
import numpy as np
import random

def classHold_nonIID_partition(df, clientsNonIIDLabels):
    """
    Generate Clients Non-IID Data

    Args:
        df (DataFrame): The whole data frame to be split across clients
        clientsNonIIDLabels (List): The non-IID labels hold by each client, each nested list contains the labels hold by a client e.g., for 3 clients and labels range 0 to 5 [[3], [0, 1, 2, 5], [1, 2, 4]]

    Returns:
        clients_df (List of DataFrames): The non-IID data created for each client
        clientsLabelDataCount (List of Dicts): Each dict contains items on the labels count of a client, i.e., Label(key)-Count(value)
    """
    
    df = df.sample(frac=1)    
    # determine the amount of data per split for a label based on the number of client with that label
    label_dataSizePerSplit = {}
    for label in df['label'].unique(): 
        num_split = sum(clientNonIIDLabels.count(label) for clientNonIIDLabels in clientsNonIIDLabels) # total number of client with this label
        dataSize_perSplit, remainder = divmod(len(df.loc[df['label']==label,:]), num_split) # split the label's records evenly based on the total number of client with the label
        label_dataSizePerSplit[label] = dataSize_perSplit

    # partition the labels data for each client
    clients_df = []
    clientsLabelDataCount = []
    for i, client_labels in enumerate(clientsNonIIDLabels):
        clients_df.append(pd.DataFrame())
        clientsLabelDataCount.append({})
        for client_label in client_labels:
            client_label_data = df.loc[df['label']==client_label, :].sample(n=label_dataSizePerSplit[client_label], replace=False)
            clients_df[i] = pd.concat([clients_df[i], client_label_data])
            clientsLabelDataCount[i][client_label] = len(client_label_data)
            df.drop(client_label_data.index, inplace=True)
        clients_df[i] = clients_df[i].sample(frac=1)
    
    return clients_df, clientsLabelDataCount
 

def dirichlet_nonIID_label_partition(df, dirProp, num_clients):
    df = df.sample(frac=1) 
    clients_df = []
    clientsLabelDataCount = []
    for _ in range(num_clients):
        clients_df.append(pd.DataFrame())
        clientsLabelDataCount.append({})

    for label in df['label'].unique(): 
        total_samples = len(df.loc[df['label']==label])
        proportions = dirProp[label]
        samples_per_client = (proportions * total_samples).astype(int)
        for i in range(0, num_clients):
            if samples_per_client[i] != 0:
                client_label_data = df.loc[df['label']==label, :].sample(n=samples_per_client[i], replace=False)
                clients_df[i] = pd.concat([clients_df[i], client_label_data])
                clientsLabelDataCount[i][label] = len(client_label_data)
                df.drop(client_label_data.index, inplace=True)
        if len(df.loc[df['label']==label]) != 0:
            non_zero_clients_indices = [i for i, sample in enumerate(samples_per_client) if sample != 0]
            random_client_index = random.choice(non_zero_clients_indices)

            remaining_label_data = df.loc[df['label']==label][:]
            clients_df[random_client_index] = pd.concat([clients_df[random_client_index], remaining_label_data])
            clientsLabelDataCount[random_client_index][label] += len(remaining_label_data)
            df.drop(remaining_label_data.index, inplace=True)

    for i in range(num_clients):
        clients_df[i] = clients_df[i].sample(frac=1, replace=False)
    
    return clients_df, clientsLabelDataCount

def assignClientLabel(total_labels, label_per_client, total_clients):
    labels = list(range(0, total_labels))
    labels_c = labels.copy() # to ensure each label appears at least once among clients
    random.shuffle(labels_c)
    split = round(len(labels_c) / total_clients)
    
    clientsNonIIDLabels = []
    label_distribution = {label: 0 for label in labels}  # Track overall label counts

    for i in range(0, total_clients):
        nonIIDLabels = []
        clientFlag = True
        dominant_labels = random.sample(labels, k=random.randint(1, max(1, total_labels // 2)))  # Random dominant labels for client
    
        while len(nonIIDLabels) < label_per_client:    
            # Replenish the labels if they run out
            if len(labels) < (total_labels * (label_per_client / 100)):
                labels = list(range(0, total_labels))
    
            # Assign labels from shuffled set to ensure diversity
            if labels_c and clientFlag:
                split_labels = labels_c[:split]
                labels_c = labels_c[split:]
                nonIIDLabels.extend(split_labels)
                clientFlag = False
                if i+1 == total_clients and labels_c:
                    nonIIDLabels.extend(labels_c[:])
    
            # Assign remaining labels with imbalanced probabilities
            if len(nonIIDLabels) < label_per_client:
                if random.random() < 0.7:  # 70% chance to select from dominant labels
                    nonIIDLabels.append(random.choice(dominant_labels))
                else:  # 30% chance to pick any label
                    nonIIDLabels.append(random.choice(labels))
    
            # Ensure no duplicates in the client's labels
            nonIIDLabels = list(set(nonIIDLabels))
    
        # Update the global label distribution
        for label in nonIIDLabels:
            label_distribution[label] += 1
    
        clientsNonIIDLabels.append(nonIIDLabels)
    
        print("Label counts across all clients:", label_distribution)
    
    return clientsNonIIDLabels


def assignClientDomain(total_domains, domain_per_client, total_clients, total_labels):
    domains = list(range(0, total_domains))
    domains_c = domains.copy() # to ensure each label appears at least once among clients
    random.shuffle(domains_c)
    split = max(1, round(len(domains_c) / total_clients))
    dominant_domains = random.sample(domains, k=random.randint(1, max(1, total_domains // 2)))
    
    clientsNonIIDDomains = []
    for i in range(0, total_clients):
        nonIIDLabels = []
        clientFlag = True
        # print(f'client{i}')
        while len(nonIIDLabels) != domain_per_client:    
            if len(domains) <= round((total_domains * (domain_per_client / 100))):
                # print('in2')
                domains = list(range(0, total_domains))
    
            if domains_c != [] and clientFlag:
                split_domains = domains_c[0:split]
                domains_c = domains_c[split:]
                nonIIDLabels.extend(split_domains)
                clientFlag = False
                
            # if len(nonIIDLabels) != domain_per_client:
            #     nonIIDLabels.append(domains.pop(random.randrange(len(domains))))
            # nonIIDLabels = list(set(nonIIDLabels))
            if len(nonIIDLabels) != domain_per_client:
                if random.random() < 0.7:  # 70% chance to select from dominant labels
                    nonIIDLabels.append(random.choice(dominant_domains))
                else:  # 30% chance to pick any label
                    nonIIDLabels.append(domains.pop(random.randrange(len(domains))))
            nonIIDLabels = list(set(nonIIDLabels))
            # print(len(nonIIDLabels))
        clientsNonIIDDomains.append(nonIIDLabels)
    
    clients_labelsDomains = []
    for i in range(total_clients):
        clients_labelsDomains.append({label: clientsNonIIDDomains[i] for label in range(total_labels)})
    print(clients_labelsDomains)
    
    return clients_labelsDomains
    

def domainHold_nonIID_partition(df, clients_labelsDomains):
    # count number of client that holds a specific label's domain
    label_domain_unique_value_counts = {}
    for client in clients_labelsDomains:
        for label, domains in client.items():
            if label not in label_domain_unique_value_counts:
                label_domain_unique_value_counts[label] = {}
            for domain in domains:
                if domain not in label_domain_unique_value_counts[label]:
                    label_domain_unique_value_counts[label][domain] = 0
                label_domain_unique_value_counts[label][domain] += 1


    # determine the amount of data per split based on the number of clients holding that label
    labelDomain_dataSizePerSplit = {}
    for label, labelDomainsCounts in label_domain_unique_value_counts.items():
        labelDomain_dataSizePerSplit[label] = {}
        for labelDomain, clientCount in labelDomainsCounts.items():
            labelDomain_dataSizePerSplit[label][labelDomain] = divmod(len(df.loc[(df['label']==label) & (df['domain']==labelDomain),:]), clientCount)[0]


    # partition the label domain data for each client
    clients_df = []
    clientsLabelDataCount = []
    for i, client_labelsDomains in enumerate(clients_labelsDomains):
        clients_df.append(pd.DataFrame())
        clientsLabelDataCount.append({})
        for clabel, clabelDomains in client_labelsDomains.items():
            dataAmt = 0
            for clabelDomain in clabelDomains:
                client_data = df.loc[(df['label']==clabel) & (df['domain']==clabelDomain), ['image', 'label', 'domain']].sample(n=labelDomain_dataSizePerSplit[clabel][clabelDomain], replace=False)
                clients_df[i] = pd.concat([clients_df[i], client_data])
                dataAmt += len(client_data)
                df.drop(client_data.index, inplace=True)
            clientsLabelDataCount[i][clabel] = dataAmt

    return clients_df, clientsLabelDataCount


def dirichlet_nonIID_domain_partition(df, dirProp, num_clients):
    df = df.sample(frac=1) 
    clients_df = []
    clientsLabelDataCount = []
    for _ in range(num_clients):
        clients_df.append(pd.DataFrame())
        clientsLabelDataCount.append({})

    for label in df['label'].unique(): 
        for domain in df['domain'].unique():
            print(label, domain)
            total_samples = len(df.loc[(df['label']==label) & (df['domain']==domain)])
            proportions = dirProp[domain]
            samples_per_client = (proportions * total_samples).astype(int)
            print(proportions)
            print(samples_per_client)
            for i in range(0, num_clients):
                if samples_per_client[i] != 0:
                    client_label_data = df.loc[(df['label']==label) & (df['domain']==domain), :].sample(n=samples_per_client[i], replace=False)
                    clients_df[i] = pd.concat([clients_df[i], client_label_data])
                    if label not in clientsLabelDataCount[i]:
                        clientsLabelDataCount[i][label] = 0
                        clientsLabelDataCount[i][label] += len(client_label_data)
                    else:
                        clientsLabelDataCount[i][label] += len(client_label_data)
                    df.drop(client_label_data.index, inplace=True)
            if len(df.loc[(df['label']==label) & (df['domain']==domain)]) != 0:
                non_zero_clients_indices = [i for i, sample in enumerate(samples_per_client) if sample != 0]
                random_client_index = random.choice(non_zero_clients_indices)
                
                remaining_label_data = df.loc[(df['label']==label) & (df['domain']==domain)][:]
                clients_df[random_client_index] = pd.concat([clients_df[random_client_index], remaining_label_data])
                clientsLabelDataCount[random_client_index][label] += len(remaining_label_data)
                df.drop(remaining_label_data.index, inplace=True)

    for i in range(num_clients):
        clients_df[i] = clients_df[i].sample(frac=1, replace=False)
    
    return clients_df, clientsLabelDataCount


# nonIID_Partition = {
#     "classHold": classHold_nonIID_partition,
#     "domainHold": domainHold_nonIID_partition,
#     "classDirichlet": dirichlet_nonIID_label_partition,
#     "domainDirichlet": dirichlet_nonIID_domain_partition

# }






