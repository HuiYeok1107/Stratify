# Stratify Official Site
Released on May 04, 2026

This repository contains the official implementation of the algorithms proposed in the paper:  
**"Stratify: Rethinking Federated Learning for Non-IID Data through Balanced Sampling"**  

📄 [Paper Link](https://www.sciencedirect.com/science/article/abs/pii/S0031320326008654) 


## Installation
```bash
# Clone the repository
git clone https://github.com/huiyeok1107/Stratify.git
cd Statify

# Create a virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
## Experiment
Below are the command-line arguments you can use to customize the training process.
| Argument       | Type   | Default  | Description |
|---------------|--------|----------|-------------|
| `dataset`   | `str`  | `mnist`  | Dataset to train on. Options: `mnist`, `cifar10`, `cifar100`, `tinyimagenet` (only for batch learning), `covtype`, `pacs`, `digitdg`. |
| `labelOrDomainPerClientHold`   | `str`  | `0`  | Number of classes or domains each client holds. |
| `dirichlet`   | `str`  | `0`  | Enable Dirichlet partition. 1: Yes 0: No. |
| `client_num`   | `str`  | `5`  | Number of FL clients participate. |
| `uniform_SLS`   | `int`  | 0 | Enable uniform ** SLS. 1: Yes 0: No.|
| `uniformClientSelection`   | `int`  | 1 | Enable uniform client selection. 1: Yes 0: No.|
| `epochs`    | `int`  | `30`     | Number of communication rounds or training epochs. |
| `augmentation`    | `int`  | `0`  | Enable train data augmentation. 1: Yes 0: No. |
| `batch_size` | `int` | `128`    | Batch size per iteration for batch learning. |
| `lr`        | `float` | `0.001` | Learning rate for the optimizer. |
| `optimizer` | `str`  | `adam`   | Optimizer to use. Options: `adam`, `sgd`. |
| `weight_decay` | `float` | `0.0` | Weight decay (L2 regularization) for the optimizer. |
| `momentum`  | `float` | `0.0`   | Momentum for optimizers like SGD. |
| `eps` | `float` | `0.0` | Epsilon value for optimizers like Adam |
| `lr_scheduler`  | `int` | `0`   | Enable cyclical learning rate scheduler. 1: Yes 0: No. |
| `grad_clip`  | `float` | `0.0`   | Gradients clipping value |
| `resultFilePath`  | `str` | `result.txt`   | The file to write model performance metrics on. |
| `startport` | `int`  | `5000`   | Starting port number. Each client will be assigned to each unique port, incrementing from this port. |

<p align="justify">❗ <b>IMPORTANT:</b> in this implementation, each client is spawned as a separate process to simulate a federated learning training environment on a single machine. Hence, please ensure that `--client_num` is less than the available CPU cores on your machine to avoid system crashes. Due to context switching between processes, the training time in this simulation does not accurately reflect real-world FL training, especially as the number of clients increases. For an accurate measurement of training time, we recommend deploying each client on a separate cloud instance or physical machine to avoid process scheduling overhead. <p>

For a complete list of dataset-specific training hyperparameters, please refer to x.txt.

### Batch-Data Per Iteration Learning
1. In the first terminal, start the client processes:
   ```bash
   python Batch_Data_Learning/client.py --dataset 'mnist' --labelOrDomainPerClientHold 5 --client_num 3 --epochs 1 --batch_size 64 --optimizer 'adam'
   ```
   **Wait for all client processes to initialize before proceeding.**
   
2. In the second terminal, start the server:
   ```bash
   python Batch_Data_Learning/server.py --dataset 'mnist' --labelOrDomainPerClientHold 5 --client_num 3 --epochs 1 --batch_size 64 --optimizer 'adam'
   ```
### Single-Sample Per Iteration Learning
1. In the first terminal, start the client processes:
   ```bash
   python Single_Sample_Learning/client.py --dataset 'mnist' --labelOrDomainPerClientHold 5 --client_num 3 --epochs 1 --batch_size 1 --optimizer 'adam'
   ```
   **Wait for all client processes to initialize before proceeding.**
   
2. In the second terminal, start the server:
   ```bash
   python Single_Sample_Learning/server.py --dataset 'mnist' --labelOrDomainPerClientHold 5 --client_num 3 --epochs 1 --batch_size 1 --optimizer 'adam'
   ```

   ## Citation

```bibtex
@article{HY2026,
  title={Stratify: Rethinking Federated Learning for Non-IID Data through Balanced Sampling},
  author={Wong, Hui Yeok and Lim, Chee Kau and Chan, Chee Seng},
  journal={Pattern Recognition},
  pages={Accepted},
  year={2026},
  publisher={Elsevier}
}
```

## Feedback
Suggestions and opinions on this work are greatly welcomed. Please contact the authors by sending an email to
`limck at um.edu.my` or `cs.chan at um.edu.my`.

## License and Copyright
The project is open source under BSD-3 license (see the ``` LICENSE ``` file). 

&#169;2026 Universiti Malaya.

