import copy
import gc
import logging
import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

from .models import *
from .utils import *
from .client import Client
from .losses import *
import src.moco.builder

logger = logging.getLogger(__name__)
def calculate_macro_metrics(all_labels, all_predictions, num_classes):
    """
    Calculate macro-averaged precision, recall and F1
    without requiring scikit-learn.
    """

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for class_id in range(num_classes):

        true_positive = 0
        false_positive = 0
        false_negative = 0

        for true_label, predicted_label in zip(
            all_labels,
            all_predictions
        ):

            if true_label == class_id and predicted_label == class_id:
                true_positive += 1

            elif true_label != class_id and predicted_label == class_id:
                false_positive += 1

            elif true_label == class_id and predicted_label != class_id:
                false_negative += 1

        if true_positive + false_positive > 0:
            precision = (
                true_positive /
                (true_positive + false_positive)
            )
        else:
            precision = 0.0

        if true_positive + false_negative > 0:
            recall = (
                true_positive /
                (true_positive + false_negative)
            )
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = (
                2.0 * precision * recall /
                (precision + recall)
            )
        else:
            f1 = 0.0

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    precision_macro = precision_sum / num_classes
    recall_macro = recall_sum / num_classes
    f1_macro = f1_sum / num_classes

    return (
        precision_macro,
        recall_macro,
        f1_macro
    )

class Server(object):
    def __init__(
        self,
        writer,
        model_config={},
        global_config={},
        data_config={},
        init_config={},
        fed_config={},
        optim_config={},
        cl_config={},
        loss_config={},
        adaptive_gamma_config={},
        temp_queue_config={},
        logit_adjustment_config={}
    ): 
        self.clients = None
        self._round = 0
        self.writer = writer
        self.modelname = model_config["name"]

        self.model = src.moco.builder.MoCo(
                            eval(model_config["name"]), model_config["name"], cl_config["cl_dim"], cl_config["cl_K"], 
                            cl_config["cl_temp"], model_config["num_classes"])
        self.criterion_cl = CCLoss(gamma=cl_config["cl_gamma"], temperature=cl_config["cl_temp"], 
                                        K=cl_config["cl_K"], num_classes=model_config["num_classes"]).cuda()

        
        self.seed = global_config["seed"]
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]
        self.diri = data_config["diri"]
        self.alpha = data_config["alpha"]

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]

        self.criterion = fed_config["criterion"]
        self.optimizer = fed_config["optimizer"]
        self.optim_config = optim_config
        self.loss_config = loss_config
        self.adaptive_gamma_config = adaptive_gamma_config
        self.temp_queue_config = temp_queue_config
        self.logit_adjustment_config = logit_adjustment_config

    def setup(self, **init_kwargs):
        assert self._round == 0

        # initialize weights of the model
        torch.manual_seed(self.seed)

        init_net(self.model, **self.init_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client
        local_datasets, test_dataset = create_datasets_CL(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid, self.diri, self.alpha, self.batch_size)
        
        # assign dataset to each client
        self.clients = self.create_clients(local_datasets)

        self.data = test_dataset
        self.dataloader = DataLoader(test_dataset,batch_size=self.batch_size,shuffle=False,drop_last=False)
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer,
            optim_config=self.optim_config,
            criterion_cl = self.criterion_cl,
            loss_config=self.loss_config,
            adaptive_gamma_config=self.adaptive_gamma_config,
            temp_queue_config=self.temp_queue_config,
            logit_adjustment_config=self.logit_adjustment_config
            )
        # Save client-wise adaptive gamma information
        os.makedirs("results", exist_ok=True)
        
        gamma_file = "results/client_gamma.csv"
        
        with open(gamma_file, "w", newline="") as f:
        
            writer = csv.writer(f)
        
            writer.writerow([
                "client_id",
                "num_samples",
                "entropy",
                "gamma"
            ])
        
            for client in self.clients:
            
                writer.writerow([
                    client.id,
                    len(client.data),
                    client.entropy,
                    client.gamma
                ])
        self.transmit_model()
        
    def create_clients(self, local_datasets):
        """Initialize each Client instance."""
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, device=self.device)
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send global model to clients, keeping inactive client models on CPU."""

        # Create a CPU copy of the global model.
        # This prevents all client models from occupying GPU memory.
        global_model_cpu = copy.deepcopy(self.model).cpu()

        if sampled_client_indices is None:
            # Initial transmission or final transmission
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                if client.model is not None:
                    client.model = None

                # Each client receives an independent model stored on CPU
                client.model = copy.deepcopy(global_model_cpu)

            message = (
                f"[Round: {str(self._round).zfill(4)}] "
                f"...successfully transmitted models to all "
                f"{str(self.num_clients)} clients!"
            )

        else:
            # Send global model only to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                if self.clients[idx].model is not None:
                    self.clients[idx].model = None

                # Store client model on CPU
                self.clients[idx].model = copy.deepcopy(global_model_cpu)

            message = (
                f"[Round: {str(self._round).zfill(4)}] "
                f"...successfully transmitted models to "
                f"{str(len(sampled_client_indices))} selected clients!"
            )

        del global_model_cpu
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(message)
        logging.info(message)

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = 0

        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update_cl(
                round_number=self._round
            )
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size
    
    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.clients[selected_index].client_update_cl(
            round_number=self._round
        )

        client_size = len(self.clients[selected_index])

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        self.model.queue_ptr[0] = 0

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
            selected_total_size = sum(selected_total_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = [0.1 for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
        
    def evaluate_global_model(self):

        self.model.eval()
        self.model.to(self.device)

        test_loss = 0.0
        correct = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for data, labels in self.dataloader:

                data = data.float().to(self.device)
                labels = labels.long().to(self.device)

                outputs = self.model(data)

                test_loss += eval(self.criterion)()(
                    outputs,
                    labels
                ).item()

                predicted = outputs.argmax(dim=1)

                correct += (
                    predicted == labels
                ).sum().item()

                all_predictions.extend(
                    predicted.cpu().tolist()
                )

                all_labels.extend(
                    labels.cpu().tolist()
                )

        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)

        test_accuracy = correct / len(self.data)

        (
            test_precision,
            test_recall,
            test_f1
        ) = calculate_macro_metrics(
            all_labels,
            all_predictions,
            num_classes=21
        )

        return (
            test_loss,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1
        )
    def fit(self):
        """Execute the whole process of federated learning."""

        self.results = {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "round_time": []
        }

        # Create results directory
        os.makedirs("results", exist_ok=True)

        metrics_file = "results/metrics.csv"

        # Create CSV and write header
        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "round",
                "loss",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "round_time_sec"
            ])

        total_start = time.perf_counter()

        for r in range(self.num_rounds):

            round_start = time.perf_counter()

            self._round = r + 1

            self.train_federated_model()

            (
                test_loss,
                test_accuracy,
                test_precision,
                test_recall,
                test_f1
            ) = self.evaluate_global_model()

            # Calculate round execution time
            round_time = time.perf_counter() - round_start

            # Store results
            self.results["loss"].append(test_loss)
            self.results["accuracy"].append(test_accuracy)
            self.results["precision"].append(test_precision)
            self.results["recall"].append(test_recall)
            self.results["f1"].append(test_f1)
            self.results["round_time"].append(round_time)

            # Save metrics to CSV
            with open(metrics_file, "a", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    self._round,
                    test_loss,
                    test_accuracy,
                    test_precision,
                    test_recall,
                    test_f1,
                    round_time
                ])

            # TensorBoard
            self.writer.add_scalars(
                "Loss",
                {
                    f"[{self.dataset_name}] "
                    f"C_{self.fraction}, "
                    f"E_{self.local_epochs}, "
                    f"B_{self.batch_size}, "
                    f"IID_{self.iid}": test_loss
                },
                self._round
            )

            self.writer.add_scalars(
                "Accuracy",
                {
                    f"[{self.dataset_name}] "
                    f"C_{self.fraction}, "
                    f"E_{self.local_epochs}, "
                    f"B_{self.batch_size}, "
                    f"IID_{self.iid}": test_accuracy
                },
                self._round
            )

            # Logging
            message = (
                f"[Round: {str(self._round).zfill(4)}] "
                f"Evaluate global model's performance...!"
                f"\n\t[Server] ...finished evaluation!"
                f"\n\t=> Loss: {test_loss:.4f}"
                f"\n\t=> Accuracy: {100.0 * test_accuracy:.2f}%"
                f"\n\t=> Precision: {test_precision:.4f}"
                f"\n\t=> Recall: {test_recall:.4f}"
                f"\n\t=> F1: {test_f1:.4f}"
                f"\n\t=> Round time: {round_time:.2f} sec\n"
            )

            print(message)
            logging.info(message)

            del message
            gc.collect()

        # Total experiment time
        total_time = time.perf_counter() - total_start

        # Save experiment summary
        with open("results/experiment_summary.txt", "w") as f:

            f.write(
                f"Total training time (sec): {total_time:.4f}\n"
            )

            f.write(
                f"Total rounds: {self.num_rounds}\n"
            )

            f.write(
                f"Average round time (sec): "
                f"{sum(self.results['round_time']) / len(self.results['round_time']):.4f}\n"
            )

            f.write(
                f"Final loss: "
                f"{self.results['loss'][-1]:.6f}\n"
            )

            f.write(
                f"Final accuracy: "
                f"{self.results['accuracy'][-1]:.6f}\n"
            )

            f.write(
                f"Final precision: "
                f"{self.results['precision'][-1]:.6f}\n"
            )

            f.write(
                f"Final recall: "
                f"{self.results['recall'][-1]:.6f}\n"
            )

            f.write(
                f"Final F1: "
                f"{self.results['f1'][-1]:.6f}\n"
            )

        self.transmit_model()