import copy
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item



class clientKWAZ(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.KL = nn.KLDivLoss()
        self.T = 4.0
        self.loss_mse = nn.MSELoss()

        self.mix_alpha = 0.1
        self.patch_size = 16
        self.updata_epoch = 30

        alps = args.alphas.split(',')
        self.alphas = list([])
        for alp in alps:
            self.alphas.append(float(alp))

        beta = args.alphas.split(',')
        self.betas = list([])
        for beta in beta:
            self.betas.append(float(beta))

        patchs = args.ps.split(',')
        self.ps = list([])
        for p_s in patchs:
            self.ps.append(int(p_s))

        self.mixalpha = self.mix_alpha
        self.mixbeta = self.mix_alpha
        self.mixbeta_g = self.mix_alpha
        self.mixpatch_size = self.patch_size
        self.mixpatchb_size = self.patch_size
        self.mixpatchb_size_g = self.patch_size

    def train(self, k):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.learning_rate)

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        if k % self.updata_epoch == 0:
            self.mixalpha, self.mixpatch_size = self.search_alpha(model, global_model, trainloader)
            self.mixbeta, self.mixpatchb_size = self.search_beta(model, global_model, trainloader)
            self.mixbeta_g, self.mixpatchb_size_g = self.search_beta(global_model, model, trainloader)

        model.train()
        global_model.train()
        global_logits1 = load_item('Server', 'global_logits1', self.save_folder_name)
        global_protos1 = load_item('Server', 'global_protos1', self.save_folder_name)
        logits1 = defaultdict(list)
        protos1 = defaultdict(list)
        global_logits2 = load_item('Server', 'global_logits2', self.save_folder_name)
        global_protos2 = load_item('Server', 'global_protos2', self.save_folder_name)
        logits2 = defaultdict(list)
        protos2 = defaultdict(list)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = model.base(x)
                output = model.head(rep)
                rep_g = global_model.base(x)
                output_g = global_model.head(rep_g)

                loss_ce = self.loss(output, y)
                if global_logits2 != None:
                    logit_new = copy.deepcopy(output.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_logits2[y_c]) != type([]):
                            logit_new[i, :] = global_logits2[y_c].data
                    loss_ce += self.loss(output, logit_new.softmax(dim=1))

                if global_protos2 is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos2[y_c]) != type([]):
                            proto_new[i, :] = global_protos2[y_c].data
                    loss_ce += self.loss_mse(proto_new, rep)

                loss_ce_g = self.loss(output_g, y)
                if global_logits1 != None:
                    logit_new = copy.deepcopy(output_g.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_logits1[y_c]) != type([]):
                            logit_new[i, :] = global_logits1[y_c].data
                    loss_ce_g += self.loss(output_g, logit_new.softmax(dim=1))
                if global_protos1 is not None:
                    proto_new = copy.deepcopy(rep_g.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos1[y_c]) != type([]):
                            proto_new[i, :] = global_protos1[y_c].data
                    loss_ce_g += self.loss_mse(proto_new, rep_g)

                optimizer.zero_grad()
                optimizer_g.zero_grad()
                loss_ce.backward(retain_graph=True)
                loss_ce_g.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                optimizer.step()
                optimizer_g.step()

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = model.base(x)
                output = model.head(rep)
                rep_g = global_model.base(x)
                output_g = global_model.head(rep_g)
                loss_ce = self.loss(output, y)
                loss_kl = F.kl_div(F.log_softmax(output / self.T, dim=1), F.softmax(output_g / self.T, dim=1),
                                   reduction='batchmean') * self.T * self.T
                mses = self.loss_mse(rep_g, rep)
                loss = mses + loss_kl + loss_ce
                loss_ce_g = self.loss(output_g, y)
                loss_kl_g = F.kl_div(F.log_softmax(output_g / self.T, dim=1), F.softmax(output / self.T, dim=1),
                                     reduction='batchmean') * self.T * self.T
                mset = self.loss_mse(rep, rep_g)
                loss_g = mset + loss_kl_g + loss_ce_g

                mix_input = self.hapm(x, self.mixalpha, self.mixpatch_size)
                mixrep = model.base(mix_input)
                with torch.no_grad():
                    mixrep_g = global_model.base(mix_input)
                loss_rep1 = self.loss_mse(mixrep_g, mixrep)
                loss += loss_rep1
                mixrep_g = global_model.base(mix_input)
                mixrep = model.base(mix_input).detach()
                loss_rep_g = self.loss_mse(mixrep, mixrep_g)
                loss_g += loss_rep_g

                mix_input = self.hapm(x, self.mixbeta, self.mixpatchb_size)
                mixlogit = model(mix_input)
                with torch.no_grad():
                    mixlogit_g = global_model(mix_input)
                loss_kl = F.kl_div(F.log_softmax(mixlogit / self.T, dim=1), F.softmax(mixlogit_g / self.T, dim=1),
                                   reduction='batchmean', ) * self.T * self.T
                loss += loss_kl

                mix_input_pred_g = self.hapm(x, self.mixbeta_g, self.mixpatchb_size_g)
                mixlogit_g = global_model(mix_input_pred_g)
                with torch.no_grad():
                    mixlogit = model(mix_input_pred_g)
                loss_kl_pred_g = F.kl_div(F.log_softmax(mixlogit_g / self.T, dim=1),
                                          F.softmax(mixlogit / self.T, dim=1),
                                          reduction='batchmean') * self.T * self.T
                loss_g += loss_kl_pred_g
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    logits1[y_c].append(output[i, :].detach().data)
                    protos1[y_c].append(rep[i, :].detach().data)
                    logits2[y_c].append(output_g[i, :].detach().data)
                    protos2[y_c].append(rep_g[i, :].detach().data)
                optimizer.zero_grad()
                optimizer_g.zero_grad()
                loss.backward(retain_graph=True)
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                optimizer.step()
                optimizer_g.step()
        save_item(agg_log(logits1), self.role, 'logits1', self.save_folder_name)
        save_item(agg_pro(protos1), self.role, 'protos1', self.save_folder_name)
        save_item(agg_log(logits2), self.role, 'logits2', self.save_folder_name)
        save_item(agg_pro(protos2), self.role, 'protos2', self.save_folder_name)
        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(global_model, self.role, 'global_model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def hapm(self, x, alpha, path_size):
        batch_size, channel, h, w = x.shape

        if alpha == 0:
            alpha = 0.1
        if path_size == 0:
            path_size = 16
        num_patches = path_size

        grid_size = int(num_patches ** 0.5)
        patch_h_base = h // grid_size
        patch_w_base = w // grid_size

        i_indices = torch.arange(grid_size, device=x.device)
        j_indices = torch.arange(grid_size, device=x.device)
        i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')

        h_starts = i_grid * patch_h_base
        h_ends = (i_grid + 1) * patch_h_base
        w_starts = j_grid * patch_w_base
        w_ends = (j_grid + 1) * patch_w_base

        jitter_h = torch.randint(-patch_h_base // 4, patch_h_base // 4, (grid_size, grid_size), device=x.device)
        jitter_w = torch.randint(-patch_w_base // 4, patch_w_base // 4, (grid_size, grid_size), device=x.device)
        h_starts = torch.clamp(h_starts + jitter_h, 0, h)
        h_ends = torch.clamp(h_ends + jitter_h, 0, h)
        w_starts = torch.clamp(w_starts + jitter_w, 0, w)
        w_ends = torch.clamp(w_ends + jitter_w, 0, w)

        perm = torch.randperm(batch_size, device=x.device)

        lam_dist = torch.distributions.beta.Beta(alpha, alpha)
        lam = lam_dist.sample((grid_size * grid_size,)).to(x.device)

        mixed_x = x.clone()
        mask_sum = torch.zeros_like(x)

        for idx in range(grid_size * grid_size):
            i = idx // grid_size
            j = idx % grid_size

            h_start = h_starts[i, j]
            h_end = h_ends[i, j]
            w_start = w_starts[i, j]
            w_end = w_ends[i, j]

            mask = torch.zeros((batch_size, channel, h, w), device=x.device)
            mask[:, :, h_start:h_end, w_start:w_end] = 1.0

            overlap = mask_sum[:, :, h_start:h_end, w_start:w_end]
            mask[:, :, h_start:h_end, w_start:w_end] *= (overlap == 0).float()
            mask_sum += mask

            lam_patch = lam[idx].view(1, 1, 1, 1)

            mixed_patch = lam_patch * x + (1 - lam_patch) * x[perm]
            mixed_x = mixed_x * (1 - mask) + mixed_patch * mask

        return mixed_x

    def search_alpha(self, model_s, model_t, train_loader):
        best_a = 0.1
        best_ps = 16
        a = self.alphas
        ps = self.ps

        maxdifs = 0.0

        model_s.eval()
        model_t.eval()

        for i in range(len(a)):
            for j in range(len(ps)):

                if i == 0 and j == 0: continue

                alpha = a[i]
                patch_size = ps[j]

                difs = 0.0

                for idx, (x, y) in enumerate(train_loader):
                    x = x.to(self.device)

                    mix_input = self.hapm(x, alpha, patch_size)

                    with torch.no_grad():
                        mixrep_s = model_s.base(mix_input)
                        mixrep_t = model_t.base(mix_input)

                    loss_kl = self.loss_mse(mixrep_s, mixrep_t)

                    difs += loss_kl.item()

                if difs > maxdifs:
                    maxdifs = difs
                    best_a = alpha
                    best_ps = patch_size

        return best_a, best_ps

    def search_beta(self, model_s, model_t, train_loader):
        best_b = 0.1
        best_ps = 16
        b = self.betas
        ps = self.ps

        maxdifs = 0.0

        model_s.eval()
        model_t.eval()

        for i in range(len(b)):
            for j in range(len(ps)):

                if i == 0 and j == 0: continue

                beta = b[i]
                patch_size = ps[j]

                difs = 0.0

                for idx, (x, y) in enumerate(train_loader):
                    x = x.to(self.device)

                    mix_input = self.hapm(x, beta, patch_size)

                    with torch.no_grad():
                        mixlogit_s = model_s(mix_input)
                        mixlogit_t = model_t(mix_input)

                    loss_kl = F.kl_div(F.log_softmax(mixlogit_s / self.T, dim=1), F.softmax(mixlogit_t / self.T, dim=1),
                                       reduction='batchmean', ) * self.T * self.T

                    difs += loss_kl.item()

                if difs > maxdifs:
                    maxdifs = difs
                    best_b = beta
                    best_ps = patch_size

        return best_b, best_ps


def agg_log(logits):
    for [label, logit_list] in logits.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            logits[label] = logit / len(logit_list)
        else:
            logits[label] = logit_list[0]

    return logits


def agg_pro(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos
