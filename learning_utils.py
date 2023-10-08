import torch
from torch import random as torch_random
from numpy import random as np_random
import random
def set_seed (seed):
    random.seed(seed)
    np_random.seed(seed)
    torch_random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    """Print Progress per Batch"""
    def __init__(self, num_batches, meters, prefix="", print_fn=print):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.print_fn = print_fn

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.print_fn('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def write_progress_updates_to_files(file_name, entry):
    with open(file_name, 'a') as f: 
        f.write(f"""{entry} \n""")

def print_cuda_mem_stats(gpu_id):
    """Report Progress for certain GPU_id"""
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory/(1024**3)
    allocated_mem = torch.cuda.memory_allocated(gpu_id)/(1024**3)
    allocated_cache = torch.cuda.memory_reserved(gpu_id)/(1024**3)
    print(f""" Cuda Allocated memory: {allocated_mem} \n Cuda Reserved memory {allocated_cache} \n From total of {total_memory} GBytes""")

def print_grad_stats(model):
    param_name= []
    grad_shape = []
    grad_mean = []
    grad_max = []
    for n, p in model.named_parameters():
        if (p.requires_grad):
            param_name.append(n)
            grad_shape.append(p.grad.shape)
            grad_mean.append(p.grad.abs().mean().item())
            grad_max.append(p.grad.abs().max().item())
    print('grad stats'+ '-'*20)
    for p, shape, mean, max_grad in zip(param_name,grad_shape,grad_mean,grad_max):
        print(f'{p}:  mean: {mean:.3e} - max: {max_grad:.3e}')

class to_device():
    """A transformation that moves a tensor to a particular device. It is particularly useful for getting used for nn_optimize modules """
    def __init__(self, device):
        self.device = device
    def __call__(self, data):
        ret  = []
        if isinstance(data, tuple) or isinstance(data, list):
            ret  = []
            for item in data:
                if isinstance(item, torch.Tensor):
                    ret.append(item.to(self.device))
                else: 
                    ret.append(item)
            return tuple(ret)
        elif  isinstance(data, torch.Tensor):
            return (data.to(self.device))
        else:
            raise Exception("The data can't be moved to the device")

def collate_train_samples(batch):
    """ =
    Used as collate_fn for pytorch dataloaders to maintain data_ident (doc_sample or pkg_train_sample) as the third item in the tuple 
    Args:
        batch: list of tuples  (x, y, sample)
    """ 
    batch_x_lst, batch_y_lst, batch_ident_lst = [], [], []
    for i in range(len(batch)):
        batch_x_lst.append(batch[i][0])
        batch_y_lst.append(batch[i][1])
        batch_ident_lst.append(batch[i][2])
    batch_x = torch.cat(batch_x_lst, dim=0)
    batch_y = torch.tensor(batch_y_lst)
    return (batch_x, batch_y, batch_ident_lst)