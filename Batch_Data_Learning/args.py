import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Training setting')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--labelOrDomainPerClientHold', default=0, type=int)
    parser.add_argument('--dirichlet', default=0, type=float)
    parser.add_argument('--client_num', default=5, type=int)

    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--lr_scheduler', default=0, type=int)

    parser.add_argument('--resultFilePath', default='result.txt', type=str)
    parser.add_argument('--start_port', default=5000, type=int)

    args = parser.parse_args()

    return args