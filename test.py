from utils import *
from unit_vector_skeleton_test import *


@torch.no_grad()
def test(model, train_loader, test_loader, extra_loader, dev, input_n, output_n, autoreg=False):
    train_loss, test_loss, extra_loss = 0, 0, 0
    train_seq, test_seq, extra_seq = [], [], []
    device = dev

    n = 0
    for cnt, data in enumerate(train_loader):
        # batch = batch.to(device)
        batch = data[0].to(device)
        batch_dim = batch.shape[0]
        n += batch_dim
        lengths = data[1].to(device)
        original_batch = data[2].to(device)
        original_gt = original_batch[:, input_n:input_n + output_n, :]

        sequences_train = batch[:, 0:input_n, :]
        sequences_gt = batch[:, input_n:input_n + output_n, :]

        if autoreg:
            sequences_predict = torch.Tensor().to(device)
            for i in range(output_n):
                if i == 0:
                    input_train = sequences_train
                elif i >= input_n:
                    input_train = sequences_predict[:, (i - input_n):, :]
                else:
                    input_train = sequences_train[:, i:, :]
                    input_train = torch.cat((input_train, sequences_predict), 1)
                pred_one = model(input_train)[:, 0, :].unsqueeze(1)
                sequences_predict = torch.cat((sequences_predict, pred_one), 1)
        else:
            sequences_predict = model(sequences_train)
        data_in = [sequences_predict, lengths]
        denorm_predict = Denormalize(data_in)
        loss = mpjpe_error(denorm_predict, original_gt)
        # loss = mpjpe_error(sequences_predict, sequences_gt)
        train_loss += loss * batch_dim

        if cnt == len(train_loader)-1:
            train_input = batch.cpu()
            train_seq.append(train_input)
            train_output = torch.cat((sequences_train, sequences_predict), 1).cpu()
            train_seq.append(train_output)
    train_loss = train_loss.detach().cpu() / n

    n = 0
    for cnt, data in enumerate(test_loader):
        # batch = batch.to(device)
        batch = data[0].to(device)
        batch_dim = batch.shape[0]
        n += batch_dim
        lengths = data[1].to(device)
        original_batch = data[2].to(device)
        original_gt = original_batch[:, input_n:input_n + output_n, :]

        sequences_train = batch[:, 0:input_n, :]
        sequences_gt = batch[:, input_n:input_n + output_n, :]

        if autoreg:
            sequences_predict = torch.Tensor().to(device)
            for i in range(output_n):
                if i == 0:
                    input_train = sequences_train
                elif i >= input_n:
                    input_train = sequences_predict[:, (i - input_n):, :]
                else:
                    input_train = sequences_train[:, i:, :]
                    input_train = torch.cat((input_train, sequences_predict), 1)
                pred_one = model(input_train)[:, 0, :].unsqueeze(1)
                sequences_predict = torch.cat((sequences_predict, pred_one), 1)
        else:
            sequences_predict = model(sequences_train)
        data_in = [sequences_predict, lengths]
        denorm_predict = Denormalize(data_in)
        loss = mpjpe_error(denorm_predict, original_gt)
        # loss = mpjpe_error(sequences_predict, sequences_gt)
        test_loss += loss * batch_dim
        if cnt == len(test_loader)-1:
            test_input = batch.cpu()
            test_seq.append(test_input)
            test_output = torch.cat((sequences_train, sequences_predict), 1).cpu()
            test_seq.append(test_output)
    test_loss = test_loss.detach().cpu() / n

    n = 0
    for cnt, data in enumerate(extra_loader):
        # batch = batch.to(device)
        batch = data[0].to(device)
        batch_dim = batch.shape[0]
        n += batch_dim
        lengths = data[1].to(device)
        original_batch = data[2].to(device)
        original_gt = original_batch[:, input_n:input_n + output_n, :]

        sequences_train = batch[:, 0:input_n, :]
        sequences_gt = batch[:, input_n:input_n + output_n, :]

        if autoreg:
            sequences_predict = torch.Tensor().to(device)
            for i in range(output_n):
                if i == 0:
                    input_train = sequences_train
                elif i >= input_n:
                    input_train = sequences_predict[:, (i - input_n):, :]
                else:
                    input_train = sequences_train[:, i:, :]
                    input_train = torch.cat((input_train, sequences_predict), 1)
                pred_one = model(input_train)[:, 0, :].unsqueeze(1)
                sequences_predict = torch.cat((sequences_predict, pred_one), 1)
        else:
            sequences_predict = model(sequences_train)
        data_in = [sequences_predict, lengths]
        denorm_predict = Denormalize(data_in)
        loss = mpjpe_error(denorm_predict, original_gt)
        # loss = mpjpe_error(sequences_predict, sequences_gt)
        extra_loss += loss * batch_dim
        if cnt == len(extra_loader) - 1:
            extra_input = batch.cpu()
            extra_seq.append(extra_input)
            extra_output = torch.cat((sequences_train, sequences_predict), 1).cpu()
            extra_seq.append(extra_output)
    extra_loss = extra_loss.detach().cpu() / n

    return train_loss.item(), test_loss.item(), extra_loss.item(), train_seq, test_seq, extra_seq
