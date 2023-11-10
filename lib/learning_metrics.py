import torch
def md_classification_accuracy (y_hat, y, ignore_index = None):
    """calculate accuracy of estimates"""
    with torch.no_grad():
        y_hat_top_1 = torch.argmax(y_hat,dim=1)
        y_hat_top_1 = y_hat_top_1.flatten()
        y = y.flatten()
        if ignore_index is not None:
            ignore_index_fltr = [y != ignore_index]
            y = y[ignore_index_fltr]
            y_hat_top_1 = y_hat_top_1[ignore_index_fltr]
        correct = torch.eq(y,y_hat_top_1).float().sum()
        return correct/torch.numel(y), torch.numel(y)