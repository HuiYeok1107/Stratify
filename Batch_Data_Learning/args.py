import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Training setting')
    parser.add_argument('--dataset', default='mnist', type=str, help="Dataset to train on. Supported: 'mnist', 'cifar10', 'cifar100', 'tinyimagenet' (only for batch learning), 'covtype', 'pacs', 'digitdg'.")
    parser.add_argument('--labelOrDomainPerClientHold', default=0, type=int, help="Number of classes or domains each client holds.")
    parser.add_argument('--dirichlet', default=0, type=float, help="Enable Dirichlet partition. 1: Yes 0: No.")
    parser.add_argument('--client_num', default=5, type=int, help="Number of FL clients participate.")

    parser.add_argument('--epochs', default=30, type=int, help="Number of communication rounds or training epochs.")
    parser.add_argument('--augmentation', default=0, type=int, help="Enable train data augmentation. 1: Yes 0: No.")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size per iteration for batch learning.")
    parser.add_argument('--optimizer', default='adam', type=str, help="Optimizer to use. Supported: 'adam', 'sgd'.")
    parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument('--weight_decay', default=0.0, type=float, help="Weight decay to use in optimizer.")
    parser.add_argument('--momentum', default=0.0, type=float, help="Momentum factor (only for SGD optimizer)")
    parser.add_argument('--eps', default=0, type=int, help="Epsilon value for optimizers like Adam")
    parser.add_argument('--lr_scheduler', default=0, type=int, help="Enable cyclical learning rate scheduler. 1: Yes 0: No.")
    parser.add_argument('--grad_clip', default=0.0, type=float, help="Gradients clipping value")


    parser.add_argument('--resultFilePath', default='result.txt', type=str, help="The file to write model performance metrics on.")
    parser.add_argument('--start_port', default=5000, type=int, help="Starting port number. Each client will be assigned to each unique port, incrementing from this port.")

    args = parser.parse_args()

    return args