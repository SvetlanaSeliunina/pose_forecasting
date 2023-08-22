import torch
import os


def mpjpe_error(batch_pred, batch_gt):
    batch_pred = batch_pred.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)

    return torch.mean(torch.norm(batch_gt - batch_pred, 2, 1))


def get_log_dir(out_dir):
    dirs = [x[0] for x in os.walk(out_dir)]
    if len(dirs) < 2:
        log_dir = os.path.join(out_dir, 'exp0')
        os.mkdir(log_dir)
    else:
        log_dir = os.path.join(out_dir, 'exp%i' % (len(dirs)-1))
        os.mkdir(log_dir)

    return log_dir
