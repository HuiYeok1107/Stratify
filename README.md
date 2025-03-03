# Fed-GT
This repository contains the implementation of the algorithms proposed in the paper:  
**"Embrace Non-IID Data: Rethinking the Collaboration Approach in Federated Learning"** by []  
📄 [Paper Link](#) 

## Installation
```bash
# Clone the repository
git clone https://github.com/huiyeok1107/Fed-GT.git
cd Fed-GT

# Create a virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
## Experiment
### Arguments

- **`--dataset`** (`str`, default=`mnist`): Dataset to train on. Options: `mnist`, `cifar10`, `cifar100`, `tinyimagenet` (only for batch learning), `covtype`, `pacs`, `digitdg`.  
- **`--labelOrDomainPerClientHold`** (`str`, default=`0`): Number of classes or domains each client holds.  
- **`--dirichlet`** (`str`, default=`0`): Enable Dirichlet partition (`1` for Yes, `0` for No).  
- **`--client_num`** (`str`, default=`5`): Number of federated learning (FL) clients.  
- **`--epochs`** (`int`, default=`30`): Number of communication rounds or training epochs.  
- **`--augmentation`** (`int`, default=`0`): Enable training data augmentation (`1` for Yes, `0` for No).  
- **`--batch_size`** (`int`, default=`128`): Batch size per iteration (for batch learning).  
- **`--lr`** (`float`, default=`0.001`): Learning rate for the optimizer.  
- **`--optimizer`** (`str`, default=`adam`): Optimizer to use (`adam`, `sgd`, etc.).  
- **`--weight_decay`** (`float`, default=`0.0`): Weight decay (L2 regularization) for the optimizer.  
- **`--momentum`** (`float`, default=`0.0`): Momentum for optimizers like SGD.  
- **`--eps`** (`float`, default=`0.0`): Epsilon value for optimizers like Adam.  
- **`--lr_scheduler`** (`int`, default=`0`): Enable cyclical learning rate scheduler (`1` for Yes, `0` for No).  
- **`--grad_clip`** (`float`, default=`0.0`): Gradient clipping value.  
- **`--resultFilePath`** (`str`, default=`result.txt`): File path to store model performance metrics.  
- **`--startport`** (`int`, default=`5000`): Starting port number. Each client gets a unique port, incrementing from this base port.
- 
| Argument       | Type   | Default  | Description |
|---------------|--------|----------|-------------|
| `--dataset`   | `str`  | `mnist`  | Dataset to train on. Options: `mnist`, `cifar10`, `cifar100`, 'tinyimagenet` (only for batch learning), `covtype`, `pacs`, `digitdg`. |
| `--labelOrDomainPerClientHold`   | `str`  | `0`  | Number of classes or domains each client holds. |
| `--dirichlet`   | `str`  | `0`  | Enable Dirichlet partition. 1: Yes 0: No. |
| `--client_num`   | `str`  | `5`  | Number of FL clients participate. |
| `--epochs`    | `int`  | `30`     | Number of communication rounds or training epochs. |
| `--augmentation`    | `int`  | `0`  | Enable train data augmentation. 1: Yes 0: No. |
| `--batch_size` | `int` | `128`    | Batch size per iteration for batch learning. |
| `--lr`        | `float` | `0.001` | Learning rate for the optimizer. |
| `--optimizer` | `str`  | `adam`   | Optimizer to use (`adam`, `sgd`, etc.). |
| `--weight_decay` | `float` | `0.0` | Weight decay (L2 regularization) for the optimizer. |
| `--momentum`  | `float` | `0.0`   | Momentum for optimizers like SGD. |
| `--eps` | `float` | `0.0` | Epsilon value for optimizers like Adam |
| `--lr_scheduler`  | `int` | `0`   | Enable cyclical learning rate scheduler. 1: Yes 0: No. |
| `--grad_clip`  | `float` | `0.0`   | Gradients clipping value |
| `--resultFilePath`  | `str` | `result.txt`   | The file to write model performance metrics on. |
| `--startport` | `int`  | `5000`   | Starting port number. Each client will be assigned to each unique port, incrementing from this port. |

Note: in this implementation, each client is spawned as a separate process to simulate the federated learning (FL) training environment using a single machine.  However, due to context switching between processes, the training time in this simulation does not accurately reflect real-world FL training, especially as the number of clients increases. For an accurate measure of training time, we recommend deploying each client on a separate cloud instance or physical machine to avoid process scheduling overhead. You can use the example script in ... to run the training with cloud instances. 

### Batch-Data Per Iteration Learning


### Single-Sample Per Iteration Learning
