# %%
import sys

sys.path.append("..")

# %%
import numpy as np
import pandas as pd

import os
import warnings
import cv2
import timm  # used for pretrained models

import albumentations  # used for image augmentations
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import math

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


# %%
class Config:
    DATA_DIR = "../data/train_images"
    TRAIN_CSV = "../data/train.csv"

    IMG_SIZE = 512
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    EPOCHS = 15  # Try 15 epochs
    BATCH_SIZE = 20

    NUM_WORKERS = 32
    DEVICE = "cuda"

    CLASSES = 11014
    SCALE = 30
    MARGIN = 0.5

    MODEL_NAME = "eca_nfnet_l0"
    FC_DIM = 512
    SCHEDULER_PARAMS = {
        "lr_start": 1e-5,
        "lr_max": 1e-5 * 32,
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }


# %%
class ShopeeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.root_dir = Config.DATA_DIR
        self.transform = transform
        self.length = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.root_dir, row.image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = row.label_group

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return {"image": image, "label": torch.tensor(label).long()}


def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(Config.IMG_SIZE, Config.IMG_SIZE, always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(0.09, 0.6), p=0.5
            ),
            albumentations.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(p=1.0),
        ]
    )


# %% [markdown]
# Learning Rate Scheduler
#
# 这个自定义学习率调度器 ShopeeScheduler 通过以下方式动态调整学习率：
#
# 在前 lr_ramp_ep 个 epoch 内，学习率从 lr_start 线性增加到 lr_max。
# 在接下来的 lr_sus_ep 个 epoch 内，学习率保持在 lr_max。
# 之后，学习率按 lr_decay 指数衰减到 lr_min。

# %%
# credit : https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images?scriptVersionId=58269290&cellId=22


class ShopeeScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        lr_start=5e-6,
        lr_max=1e-5,
        lr_min=1e-6,
        lr_ramp_ep=5,
        lr_sus_ep=0,
        lr_decay=0.8,
        last_epoch=-1,
    ):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        super(ShopeeScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            self.last_epoch += 1
            return [self.lr_start for _ in self.optimizer.param_groups]

        lr = self._compute_lr_from_epoch()
        self.last_epoch += 1

        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs

    def _compute_lr_from_epoch(self):
        if self.last_epoch < self.lr_ramp_ep:
            lr = (
                self.lr_max - self.lr_start
            ) / self.lr_ramp_ep * self.last_epoch + self.lr_start

        elif self.last_epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max

        else:
            lr = (self.lr_max - self.lr_min) * self.lr_decay ** (
                self.last_epoch - self.lr_ramp_ep - self.lr_sus_ep
            ) + self.lr_min
        return lr


# %%
# credit : https://github.com/Yonghongwei/Gradient-Centralization


def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x


class Ranger(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,  # lr
        alpha=0.5,
        k=5,
        N_sma_threshhold=5,  # Ranger options
        betas=(0.95, 0.999),
        eps=1e-5,
        weight_decay=0,  # Adam options
        # Gradient centralization on or off, applied to conv layers only or conv + fc layers
        use_gc=True,
        gc_conv_only=False,
        gc_loc=True,
    ):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        if not lr > 0:
            raise ValueError(f"Invalid Learning Rate: {lr}")
        if not eps > 0:
            raise ValueError(f"Invalid eps: {eps}")

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(
            lr=lr,
            alpha=alpha,
            k=k,
            step_counter=0,
            betas=betas,
            N_sma_threshhold=N_sma_threshhold,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        # level of gradient centralization
        # self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}"
        )
        if self.use_gc and self.gc_conv_only == False:
            print(f"GC applied to both conv and fc layers")
        elif self.use_gc and self.gc_conv_only == True:
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        # loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        "Ranger optimizer does not support sparse gradients"
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if (
                    len(state) == 0
                ):  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    # print("Initializing slow buffer...should not see this at load from saved model!")
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state["slow_buffer"] = torch.empty_like(p.data)
                    state["slow_buffer"].copy_(p.data)

                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # GC operation for Conv layers and FC layers
                # if grad.dim() > self.gc_gradient_threshold:
                #    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                if self.gc_loc:
                    grad = centralized_gradient(
                        grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only
                    )

                state["step"] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # compute mean moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                buffered = self.radam_buffer[int(state["step"] % 10)]

                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    buffered[2] = step_size

                # if group['weight_decay'] != 0:
                #    p_data_fp32.add_(-group['weight_decay']
                #                     * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg

                if group["weight_decay"] != 0:
                    G_grad.add_(p_data_fp32, alpha=group["weight_decay"])
                # GC operation
                if self.gc_loc == False:
                    G_grad = centralized_gradient(
                        G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only
                    )

                p_data_fp32.add_(G_grad, alpha=-step_size * group["lr"])
                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state["step"] % group["k"] == 0:
                    # get access to slow param tensor
                    slow_p = state["slow_buffer"]
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)

        return loss


# %%
class Mish_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]

        v = 1.0 + i.exp()
        h = v.log()
        grad_gh = 1.0 / h.cosh().pow_(2)

        # Note that grad_hv * grad_vx = sigmoid(x)
        # grad_hv = 1./v
        # grad_vx = i.exp()

        grad_hx = i.sigmoid()

        grad_gx = grad_gh * grad_hx  # grad_hv * grad_vx

        grad_f = torch.tanh(F.softplus(i)) + i * grad_gx

        return grad_output * grad_f


class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print("Mish initialized")
        pass

    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)


def replace_activations(model, existing_layer, new_layer):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_activations(
                module, existing_layer, new_layer
            )

        if type(module) is existing_layer:
            layer_new = new_layer
            model._modules[name] = layer_new
    return model


# %%
class ArcMarginProduct(nn.Module):
    """
    ArcFace Layer, which can be directly integrated into the last layer of the network.
    """

    def __init__(
        self,
        in_features,
        out_features,
        scale=30.0,
        margin=0.50,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.ce = nn.CrossEntropyLoss()

        self.reset_parameters()

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output, self.ce(output, label)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)


class ShopeeModel(nn.Module):
    def __init__(
        self,
        n_classes=Config.CLASSES,
        model_name=Config.MODEL_NAME,
        fc_dim=Config.FC_DIM,
        margin=Config.MARGIN,
        scale=Config.SCALE,
        use_fc=True,
        pretrained=True,
    ):
        super(ShopeeModel, self).__init__()
        print(f"Building Model Backbone for {model_name} model")

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if model_name == "resnext50_32x4d":
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif "efficientnet" in model_name:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif "nfnet" in model_name:
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=0.1)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self.reset_parameters()
            final_in_features = fc_dim

        self.final = ArcMarginProduct(
            final_in_features, n_classes, scale=scale, margin=margin
        )

    def forward(self, image, label):
        features = self.extract_features(image)
        logits, loss = self.final(features, label)
        return logits, loss

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x


# %%
def train_fn(model, loader, optimizer, scheduler, i):
    model.train()
    fin_loss = 0.0
    tk = tqdm(loader, desc="Epoch" + " [TRAIN] " + str(i + 1))

    for t, data in enumerate(tk):
        for k, v in data.items():
            data[k] = v.to(Config.DEVICE)

        optimizer.zero_grad()
        _, loss = model(**data)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        fin_loss += loss.item()

        tk.set_postfix(
            {
                "loss": "%.6f" % float(fin_loss / (t + 1)),
                "LR": optimizer.param_groups[0]["lr"],
            }
        )

    scheduler.step()
    return fin_loss / len(loader)


# %%
from torch.nn import DataParallel


def run_training():
    df = pd.read_csv(Config.TRAIN_CSV)

    labelencoder = LabelEncoder()
    df["label_group"] = labelencoder.fit_transform(df["label_group"])

    trainset = ShopeeDataset(df, transform=get_train_transforms())

    trainloader = DataLoader(
        trainset,
        batch_size=Config.BATCH_SIZE,
        pin_memory=True,
        num_workers=Config.NUM_WORKERS,
        shuffle=True,
        drop_last=True,
    )

    model = ShopeeModel()
    model.to(Config.DEVICE)

    existing_layer = torch.nn.SiLU
    new_layer = Mish()
    model = replace_activations(
        model, existing_layer, new_layer
    )  # in eca_nfnet_l0 SiLU() is used, but it will be replace by Mish()

    model = DataParallel(model)

    optimizer = Ranger(model.parameters(), lr=Config.SCHEDULER_PARAMS["lr_start"])
    scheduler = ShopeeScheduler(optimizer, **Config.SCHEDULER_PARAMS)

    for i in range(Config.EPOCHS):
        avg_loss_train = train_fn(model, trainloader, optimizer, scheduler, i)
        state_dict = model.module.state_dict()
        torch.save(state_dict, f"model_{i}_{avg_loss_train}.pt")


run_training()
