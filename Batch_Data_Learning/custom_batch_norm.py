import torch
import torch.nn as nn
import requests
import pickle
from Batch_Data_Learning.args import args_parser
global args
args = args_parser()
serverPort = args.start_port 
batchSize = 400

class CustomBatchNormManualFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, gamma, beta, mean, var, eps=1e-5):

    #normalization
    center_input = input - mean[None, :, None, None]
    denominator = var[None, :, None, None] + eps
    denominator = denominator.sqrt()
    in_hat = center_input/denominator

    #scale and shift
    out = gamma[None, :, None, None] * in_hat + beta[None, :, None, None]
        
    #store constants
    ctx.save_for_backward(gamma, denominator, in_hat, input)
    ctx.epsilon = eps
    
    return out


  @staticmethod
  def backward(ctx, grad_output):
    global batchSize, serverPort
    
    gamma, denominator, in_hat, input = ctx.saved_tensors
    
    _, C, H, W = grad_output.shape
    N = batchSize 
    n = N * H * W

    den_inv = 1/denominator
    
    #gradient of the input
    if ctx.needs_input_grad[0]:
        
        grad_in_hat = grad_output * gamma[None, :, None, None]
        
        term_1 = grad_in_hat * (N * W * H)

        local_term_2 = torch.sum(grad_in_hat, dim=[0, 2, 3])
        # term_2 = getOverallTerm2(local_term_2)
        result = requests.post(f'http://127.0.0.1:{serverPort}/computeBatchTerm2', data=pickle.dumps(local_term_2.detach()))
        term_2 = pickle.loads(result.content)

        local_part_term_3 = torch.sum(grad_in_hat * in_hat, dim=[0, 2, 3])
        # part_term_3 = getOverallPartTerm3(local_part_term_3)
        result = requests.post(f'http://127.0.0.1:{serverPort}/computeBatchPartTerm3', data=pickle.dumps(local_part_term_3.detach()))
        part_term_3 = pickle.loads(result.content)
    
        term_3 = in_hat * part_term_3[None, :, None, None]
      
        grad_input = (1/(N * W * H)) * den_inv * (term_1 - term_2[None, :, None, None] - term_3)

    else:
        grad_input = None

    #gradient of gamma
    if ctx.needs_input_grad[1]:
        grad_gamma = torch.sum(torch.mul(grad_output, in_hat), dim=[0, 2, 3])
    else:
        grad_gamma = None
        
    #gradient of beta
    if ctx.needs_input_grad[2]:
        grad_beta = grad_output.sum(dim=[0, 2, 3])
        # print(f'grad beta: {grad_beta}')
    else:
        grad_beta = None

    # return gradients of the three tensor inputs and None for forward input that do not need gradents (e.g., the constant eps)
    return grad_input, grad_gamma, grad_beta, None, None, None


class CustomBatchNormManualModule(nn.Module):

  def __init__(self, n_neurons, eps=1e-5, momentum=0.1):
    super(CustomBatchNormManualModule, self).__init__()

    #save parameters
    self.n_neurons = n_neurons
    self.eps = eps
    self.momentum = momentum

    # self.running_mean = torch.zeros(self.n_neurons)
    # self.running_var = torch.ones(self.n_neurons)

    self.register_buffer('running_mean', torch.zeros(self.n_neurons, dtype=torch.float))
    self.register_buffer('running_var', torch.ones(self.n_neurons, dtype=torch.float))

    #Initialize gamma and beta
    self.gamma = nn.Parameter(torch.ones(self.n_neurons, dtype=torch.float))
    self.beta = nn.Parameter(torch.zeros(self.n_neurons, dtype=torch.float))

  def forward(self, input):
    global batchSize, serverPort

    _, C, H, W = input.shape
    N = batchSize
    # print('len placeholderBatch')
    print(N)
    n = N * H * W
    # print(f'n: {n}')
    batch_normalization = CustomBatchNormManualFunction()
    if self.training:
      # batch mean
      localSum = input.sum(dim=[0, 2, 3])
      # batchMean = calculateBatchMean(localSum, n)
      result = requests.post(f'http://127.0.0.1:{serverPort}/computeBatchMean', files={'localSum': pickle.dumps(localSum.detach()), 'n': pickle.dumps(n)})
      batchMean = pickle.loads(result.content)
    #   print(f'batch mean: {batchMean}')

      # batch variance
      localStdDSum = ((input - batchMean[None, :, None, None]) ** 2).sum(dim=[0, 2, 3])
      # batchVar = calculateBatchVar(localStdDSum, n)
      result = requests.post(f'http://127.0.0.1:{serverPort}/computeBatchVar', files={'localStdv': pickle.dumps(localStdDSum.detach()), 'n': pickle.dumps(n)})
      batchVar = pickle.loads(result.content)
    #   print(f'batchVar: {batchVar}')

      # with torch.no_grad():
      #   self.running_mean = self.momentum * batchMean + (1 - self.momentum) * self.running_mean
      #   self.running_var = self.momentum * batchVar * n / (n - 1) + (1 - self.momentum) * self.running_var

      out = batch_normalization.apply(input, self.gamma, self.beta, batchMean, batchVar, self.eps)
    else:
      out = batch_normalization.apply(input, self.gamma, self.beta, self.running_mean, self.running_var, self.eps)

    return out
  