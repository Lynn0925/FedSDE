import os

from utils import Logger
from utils.models import *
import torch
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model
from utils.util import Ensemble_A
from collections import defaultdict

class FedServer(object):
    def __init__(self, args, client_list, dataset_id, model_name):
        """
        Initialize the server for federated learning.
        :param client_list: List of the connected clients in networks
        :param dataset_id: Dataset name for the application scenario
        :param model_name: Machine learning model name for the application scenario
        """
        # Initialize the dict and list for system settings
        self.client_state = {}
        self.client_loss = {}
        self.client_n_data = {}
        self.selected_clients = []
        # batch size for testing
        self._batch_size = 200
        self.client_list = client_list

        # Initialize the test dataset
        self.testset = None

        # Initialize the hyperparameter for federated learning in the server
        self.round = 0
        self.n_data = 0
        self._dataset_id = dataset_id

        # Testing on GPU
        gpu = args.gpu_id
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

        # Initialize the global machine learning model
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel, im_size=self._image_dim)

        self.rng = np.random.default_rng(args.sys_i_seed)
        self._args = args

    def load_testset(self, testset):
        """
        Server loads the test dataset.
        :param data: Dataset for testing.
        """
        self.testset = testset

    def state_dict(self):
        """
        Server returns global model dict.
        :return: Global model dict
        """
        return self.model.state_dict()

    def test(self):
        """
        Server tests the model on test dataset.
        """
        test_loader = DataLoader(self.testset, batch_size=200, shuffle=False)
        self.model.to(self._device)
        self.model.eval()
        accuracy_collector = 0
        for step, (x, y) in enumerate(test_loader.dataset):
            with torch.no_grad():
                b_x = x.to(self._device)  # Tensor on GPU
                b_y = y.to(self._device)  # Tensor on GPU

                test_output = self.model(b_x)
                pred_y = torch.max(test_output, 1)[1].to(self._device).data.squeeze()
                accuracy_collector = accuracy_collector + sum(pred_y == b_y)
                # print(b_y)
                # print(pred_y)
                # print(test_output)
                # break
        accuracy = accuracy_collector / len(self.testset)

        return accuracy.cpu().numpy()

    def select_clients(self, connection_ratio=1):
        """
        Server selects a fraction of clients.
        :param connection_ratio: connection ratio in the clients
        """
        # select a fraction of clients
        self.selected_clients = []
        for client_id in self.client_list:
            b = self.rng.binomial(np.ones(1).astype(int), connection_ratio)
            if b:
                self.selected_clients.append(client_id)
        # print(self.selected_rsu_ids)
        return self.selected_clients

    def agg(self):
        """
        Server aggregates models from connected clients.
        :return: model_state: Updated global model after aggregation
        :return: avg_loss: Averaged loss value
        :return: n_data: Number of the local data points
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        # Initialize a model for aggregation
        model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel,
                           im_size=self._image_dim)
        model_state = model.state_dict()
        avg_loss = 0

        # Aggregate the local updated models from selected clients
        for i, name in enumerate(self.selected_clients):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                if i == 0:
                    model_state[key] = self.client_state[name][key] * self.client_n_data[name] / self.n_data
                else:
                    model_state[key] = model_state[key] + self.client_state[name][key] * self.client_n_data[
                        name] / self.n_data

            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        # Server load the aggregated model as the global model
        self.model.load_state_dict(model_state)
        self.round = self.round + 1
        n_data = self.n_data

        return model_state, avg_loss, n_data

    def rec(self, name, state_dict, n_data, loss):
        """
        Server receives the local updates from the connected client k.
        :param name: Name of client k
        :param state_dict: Model dict from the client k
        :param n_data: Number of local data points in the client k
        :param loss: Loss of local training in the client k
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss

    def flush(self):
        """
        Flushing the client information in the server
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}

    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.model.state_dict(), path)


class EnsembleServer(FedServer):
    def __init__(self, args, client_list, dataset_id, model_name):
        super().__init__(args, client_list, dataset_id, model_name)
        self.client_model = {}
        self.model = None
        # Clients' model and dataset
        self.client_dict = {}
        self.client_model = {}
        self.client_data = {}
        self.cls_2_client = defaultdict(list)
        self.client_cls_score = {}

    def train(self):
        print("Ensemble training")
        
        self.model = Ensemble_A(list(self.client_model.values()))
        acc = self.test()
        L = Logger()
        logger = L.get_logger()
        logger.info('Ensemble:' + '| Accuracy: %.4f ' % acc)

    def load_cls_record(self, cls_record):
        """
        Client loads the statistic of local label.
        :param cls_record: class number record
        """
        self._cls_record = {}
        self._cls_record = {int(k): v for k, v in cls_record.items()}
        self.pmf_dict = defaultdict(lambda: np.zeros(self._num_class))
        self.data_amt_dict = {}
        for client_id, record in self._cls_record.items():
            for p in range(self._num_class):
                self.pmf_dict[client_id][p] = record.get(float(p), 0) / sum(record.values())

            self.data_amt_dict[client_id] = sum(record.values()) / sum(
                [sum(_r.values()) for _r in self._cls_record.values()])
