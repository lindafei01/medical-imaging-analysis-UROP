import torch
import numpy as np
import os
import copy
import datetime
import socket
import time
import opacus
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from opacus import PrivacyEngine


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# data preprocessing
def mnist_noniid(labels, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards = int(num_users * 5)
    num_imgs = int(50000 / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    np.random.seed(0)
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 5, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def load_mnist(num_users):
    train = datasets.MNIST(root="~/data/", train=True, download=True, transform=transforms.ToTensor())
    train_data = train.data.float()[:50000]
    train_label = train.targets[:50000]
    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data - mean) / std

    test = datasets.MNIST(root="~/data/", train=False, download=True, transform=transforms.ToTensor())
    test_data = test.data.float()
    test_label = test.targets
    test_data = (test_data - mean) / std

    # split MNIST (training set) into non-iid data sets
    non_iid = []
    user_dict = mnist_noniid(train_label, num_users)
    for i in range(num_users):
        idx = user_dict[i]
        d = train_data[idx].flatten(1)
        targets = train_label[idx].float()
        non_iid.append((d, targets))
    non_iid.append((test_data.flatten(1).float(), test_label.float()))
    return non_iid

class MLP(nn.Module):
    """Neural Networks"""
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        seed = 1
        torch.manual_seed(seed)
        n_hidden_1 = 256
        n_hidden_2 = 256
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(),
            nn.Linear(n_hidden_2, out_dim)
        )
        view_weights = self.model.state_dict()

    def forward(self, x):
        x = self.model(x)
        return x

class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """

    def __init__(self, model, output_size, data, E, batch_size, q, clip, eps, delta, device = None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
        self.data = data
        self.output_size = output_size
        self.model_name = model
        self.device = device
        self.BATCH_SIZE = batch_size
        self.torch_dataset = TensorDataset(torch.tensor(data[0]),
                                           torch.tensor(data[1]))
        self.data_size = len(self.torch_dataset)
        self.data_loader = DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        self.E = E
        self.clip = clip
        self.q = q
        self.eps = eps
        self.delta = delta
        self.model = model(data[0].shape[1], output_size).to(self.device)

        # compute noise using Opacus
        self.privacy_engine = PrivacyEngine(secure_mode=False)
        # >>>>>>>>>>>>>> MOVE THIS TO RECV >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # self.model, self. optimizer, self.data_loader = self.privacy_engine.make_private(
        #     module=self.model,
        #     optimizer=torch.optim.SGD(self.model.parameters(), lr=0.2, momentum=0.0),
        #     data_loader=self.data_loader,
        #     noise_multiplier=1.0,  # sigma
        #     max_grad_norm=self.clip,
        # )
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def recv(self, model_param):
        """receive global model from aggregator (server)"""
        self.model = self.model_name(self.data[0].shape[1], self.output_size).to(self.device)
        self.model.load_state_dict(copy.deepcopy(model_param))
        self.model, self. optimizer, self.data_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=torch.optim.SGD(self.model.parameters(), lr=0.2, momentum=0.0),
            data_loader=self.data_loader,
            noise_multiplier=1.0,  # sigma
            max_grad_norm=self.clip,
        )

    def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.0)  # THIS LINE HAS NO EFFECT
        epoch_loss = []
        batch_loss = []

        for e in range(self.E):
            # randomly select q fraction samples from data
            # according to the privacy analysis of Opacus
            # training "Lots" are sampled by poisson sampling

            idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]
            sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
            sample_data_loader = DataLoader(
                dataset=sampled_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True
            )

            for batch_x, batch_y in sample_data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                pred_y = self.model(batch_x.float())
                loss = criterion(pred_y, batch_y.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss += [sum(batch_loss) / len(batch_loss)]

        # eps, best_alpha = self.privacy_engine.accountant.get_privacy_spent(
        #     delta=self.delta
        # )
        eps, best_alpha = self.privacy_engine.accountant.get_privacy_spent(
            delta=self.delta
        )
        print(
            f"Client loss in this round: {np.mean(batch_loss):.6f}\t"
            f"(ε = {eps:.4f}, δ = {self.delta}) for α = {best_alpha}\n"
        )

        client_loss = sum(epoch_loss) / len(epoch_loss)
        return client_loss

class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model from clients
        2. Aggregate local models
        3. Compute global model, broadcast global model to clients
    """

    def __init__(self, fl_param):
        super(FLServer, self).__init__()
        self.device = fl_param['device']
        self.client_num = fl_param['client_num']
        self.C = fl_param['C']  # (float) C in [0, 1]
        self.data = []
        self.target = []
        self.T = fl_param['tot_T']  # total number of global iterations (communication rounds)

        self.lr = fl_param['lr']
        self.clip = fl_param['clip']
        self.eps = fl_param['eps']
        self.delta = fl_param['delta']
        for sample in fl_param['data'][self.client_num:]:
            self.data += [torch.tensor(sample[0]).to(self.device)]  # test set
            self.target += [torch.tensor(sample[1]).to(self.device)]  # target label

        self.input_size = int(self.data[0].shape[1])
        self.clients = [FLClient(fl_param['model'],
                                 fl_param['output_size'],
                                 fl_param['data'][i],
                                 fl_param['E'],
                                 fl_param['batch_size'],
                                 fl_param['q'],
                                 fl_param['clip'],
                                 fl_param['eps'],
                                 fl_param['delta'],
                                 self.device)
                        for i in range(self.client_num)]

        self.global_model = fl_param['model'](self.input_size, fl_param['output_size']).to(self.device)
        self.weight = np.array([client.data_size * 1.0 for client in self.clients])
        self.broadcast(self.global_model.state_dict())

    def aggregated(self, idxs_users):
        """FedAvg - Update model using weights"""
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:  # initialize to all zero
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                new_par[name] += par[name] * (w / self.C)

        # rewrite model parameters key name
        # map = {'_module.model.1.weight': 'model.1.weight', '_module.model.1.bias': 'model.1.bias',
        #        '_module.model.3.weight': 'model.3.weight', '_module.model.3.bias': 'model.3.bias',
        #        '_module.model.5.weight': 'model.5.weight', '_module.model.5.bias': 'model.5.bias'}
        # for k in list(new_par):
        #     new_par[map[k]] = new_par.pop(k)

        self.global_model.load_state_dict(copy.deepcopy(new_par))
        return self.global_model.state_dict().copy()

    def broadcast(self, new_par):
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv(new_par)

    def test_acc(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for i in range(len(self.data)):
            t_pred_y = self.global_model(self.data[i])
            _, predicted = torch.max(t_pred_y, 1)
            correct += (predicted == self.target[i]).sum().item()
            tot_sample += self.target[i].size(0)
        acc = correct / tot_sample
        return acc

    def global_update(self):
        idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
        clients_loss = []
        i = 0
        for idx in idxs_users:
            i += 1
            print(f"The Client {i} (idx = {idx}) is Training:\n*******************")
            clients_loss += [self.clients[idx].update()]
        loss = sum(clients_loss) / len(clients_loss)
        self.broadcast(self.aggregated(idxs_users))
        acc = self.test_acc()
        # torch.cuda.empty_cache()
        return loss, acc

    def set_lr(self, lr):
        for c in self.clients:
            c.lr = lr

# initialize clients numbers
client_num = 10
# load data
d = load_mnist(client_num)


lr = 0.2
fl_param = {
    'output_size': 10,              # number of units in output layer
    'client_num': client_num,       # number of clients
    'C': 1,
    'model': MLP,                   # model
    'data': d,                      # dataset
    'q': 0.1,                       # sampling rate
    'tot_T': 3,                    # number of aggregation times
    'E': 10,                        # number of local iterations
    'batch_size': 50,
    'lr': 0.2,                      # learning rate
    'clip': 1,                      # clipping norm
    'eps': 50,                      # privacy budget for each global communication
    'delta': 1e-5,                  # approximate differential privacy: (epsilon, delta)-DP
    'device': device
}
import warnings
warnings.filterwarnings("ignore")
start_time = time.time()
fl_entity = FLServer(fl_param).to(device)
print('Currently performing FL with DP ---------------------------:')
print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
acc = []
loss = []
for t in range(fl_param['tot_T']):
    fl_entity.set_lr(lr)
    ''' Update the local model according to weights update '''
    loss_, acc_ = fl_entity.global_update()
    loss += [loss_]
    acc += [acc_]
    print("Round = {:d}, loss={:.4f}, acc = {:.4f}".format(t+1, loss[-1], acc[-1]))
print(f'Total time: {time.time() - start_time :.2f} s.')