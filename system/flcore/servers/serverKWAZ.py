import time
from collections import defaultdict

from clients.clientKWAZ import clientKWAZ
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import BaseHeadSplit


class FedKWAZ(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientKWAZ)
        self.selected_clients = self.select_clients()
        for client_id in range(self.num_clients):
            self.uploaded_ids.append(client_id)
            global_model = BaseHeadSplit(args, 0).to(args.device)
            print(f"Client {client_id}: Proxy architecture: {type(global_model.base).__name__}")
            role = 'Client_' + str(client_id)
            save_item(global_model, role, 'global_model', self.save_folder_name)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            for client in self.selected_clients:
                client.train(i + 1)

            self.receive_ids()
            self.receive_protosAndlogits()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def receive_protosAndlogits(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        uploaded_protos1 = []
        uploaded_logits1 = []
        uploaded_protos2 = []
        uploaded_logits2 = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos1 = load_item(client.role, 'protos1', client.save_folder_name)
            logits1 = load_item(client.role, 'logits1', client.save_folder_name)
            uploaded_protos1.append(protos1)
            uploaded_logits1.append(logits1)
            protos2 = load_item(client.role, 'protos2', client.save_folder_name)
            logits2 = load_item(client.role, 'logits2', client.save_folder_name)
            uploaded_protos2.append(protos2)
            uploaded_logits2.append(logits2)

        global_logits1 = logit_aggregation(uploaded_logits1)
        global_protos1 = proto_aggregation(uploaded_protos1)
        save_item(global_protos1, self.role, 'global_protos1', self.save_folder_name)
        save_item(global_logits1, self.role, 'global_logits1', self.save_folder_name)
        global_logits2 = logit_aggregation(uploaded_logits2)
        global_protos2 = proto_aggregation(uploaded_protos2)
        save_item(global_protos2, self.role, 'global_protos2', self.save_folder_name)
        save_item(global_logits2, self.role, 'global_logits2', self.save_folder_name)

def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


def logit_aggregation(local_logits_list):
    agg_logits_label = defaultdict(list)
    for local_logits in local_logits_list:
        for label in local_logits.keys():
            agg_logits_label[label].append(local_logits[label])

    for [label, logit_list] in agg_logits_label.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            agg_logits_label[label] = logit / len(logit_list)
        else:
            agg_logits_label[label] = logit_list[0].data

    return agg_logits_label
