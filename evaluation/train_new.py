import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import CTCLabelConverter, AttnLabelConverter, Averager, adjust_learning_rate, cos_adjust_learning_rate
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation, benchmark_all_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(opt, log):
    """dataset preparation"""
    # train dataset. for convenience
    if opt.select_data == "synth":
        select_data = ["synth"]

    # set batch_ratio for each data.
    if opt.batch_ratio:
        batch_ratio = opt.batch_ratio.split("-")
    else:
        batch_ratio = [round(1 / len(select_data), 3)] * len(select_data)

    train_loader = Batch_Balanced_Dataset(
        opt, opt.train_data, select_data, batch_ratio, log
    )

    AlignCollate_valid = AlignCollate(opt, mode="test")
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=opt.valid_data, opt=opt, mode="test"
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid,
        pin_memory=False,
    )
    log.write(valid_dataset_log)
    print("-" * 80)
    log.write("-" * 80 + "\n")

    """ model configuration """
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
        opt.sos_token_index = converter.dict["[SOS]"]
        opt.eos_token_index = converter.dict["[EOS]"]
    opt.num_class = len(converter.character)

    model = Model(opt)

    # weight initialization
    for name, param in model.named_parameters():
        if "localization_fc2" in name:
            print(f"Skip {name} as it is already initialized")
            continue
        try:
            if "bias" in name:
                init.constant_(param, 0.0)
            elif "weight" in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if "weight" in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != "":
        load_log = f"### loading pretrained model from {opt.saved_model}\n"

        pretrained_state_dict_qk = torch.load(opt.saved_model)
        pretrained_state_dict_qk = pretrained_state_dict_qk['state_dict']
        pretrained_state_dict = {}
        for name in pretrained_state_dict_qk:
            print(name)
            if "encoder_q" in name:
                rename = name.replace("encoder_q.", "")
                if opt.decoder == "A":
                    pretrained_state_dict[rename] = pretrained_state_dict_qk[name]
                elif opt.decoder == "BA":
                    if not "SequenceModeling" in name:
                        pretrained_state_dict[rename] = pretrained_state_dict_qk[name]
                else:
                    raise NotImplementedError

        for name, param in model.named_parameters():
            try:
                if not name in pretrained_state_dict.keys():
                    raise KeyError
                if not opt.random_init:
                    param.data.copy_(
                        pretrained_state_dict[name].data
                    )  # load from pretrained model
                param.requires_grad = False  # Freeze
                load_log += f"pretrained layer (freezed): {name}\n"
                
            except:
                load_log += f"non-pretrained layer: {name}\n"

        print(load_log)
        log.write(load_log + "\n")

    log.write(repr(model) + "\n")

    """ setup loss """
    if "CTC" in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        # ignore [PAD] token
        criterion = torch.nn.CrossEntropyLoss(ignore_index=converter.dict["[PAD]"]).to(
            device
        )

    # loss averager
    train_loss_avg = Averager()

    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print(f"Trainable params num: {sum(params_num)}")
    log.write(f"Trainable params num: {sum(params_num)}\n")
    [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            filtered_parameters,
            lr=opt.lr,
            momentum=opt.sgd_momentum,
            weight_decay=opt.sgd_weight_decay,
        )
    elif opt.optimizer == "adadelta":
        optimizer = torch.optim.Adadelta(
            filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps, weight_decay=1e-4
        )
    elif opt.optimizer == "adam":
        optimizer = torch.optim.Adam(filtered_parameters, lr=opt.lr)
    elif opt.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(filtered_parameters, lr=opt.lr, weight_decay=opt.adamw_weight_decay)
    print("Optimizer:")
    print(optimizer)
    log.write(repr(optimizer) + "\n")

    if opt.optimizer == "AdamW":
        pass
    elif "super" in opt.schedule:
        if opt.optimizer == "sgd":
            cycle_momentum = True
        else:
            cycle_momentum = False

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=opt.lr,
            cycle_momentum=cycle_momentum,
            div_factor=20,
            final_div_factor=1000,
            total_steps=opt.num_iter,
        )
        print("Scheduler:")
        print(scheduler)
        log.write(repr(scheduler) + "\n")

    """ final options """
    opt_log = "------------ Options -------------\n"
    args = vars(opt)
    for k, v in args.items():
        if str(k) == "character" and len(str(v)) > 500:
            opt_log += f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}\n"
        else:
            opt_log += f"{str(k)}: {str(v)}\n"
    opt_log += "---------------------------------------\n"
    print(opt_log)
    log.write(opt_log)
    log.close()

    """ start training """
    start_iter = 0

    start_time = time.time()
    best_score = -1

    # training loop
    for iteration in tqdm(
        range(start_iter + 1, opt.num_iter + 1),
        total=opt.num_iter,
        position=0,
        leave=True,
    ):
        image_tensors, labels = train_loader.get_batch()

        image = image_tensors.to(device)
        labels_index, labels_length = converter.encode(
            labels, batch_max_length=opt.batch_max_length
        )
        batch_size = image.size(0)

        # default recognition loss part
        if "CTC" in opt.Prediction:
            preds = model(image)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
            loss = criterion(preds_log_softmax, labels_index, preds_size, labels_length)
        else:
            preds = model(image, labels_index[:, :-1])  # align with Attention.forward
            target = labels_index[:, 1:]  # without [SOS] Symbol
            loss = criterion(
                preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
            )

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), opt.grad_clip
        )  # gradient clipping with 5 (Default)
        optimizer.step()
        train_loss_avg.add(loss)

        
        if opt.optimizer == "AdamW":
            cos_adjust_learning_rate(optimizer, iteration, opt)
        else:
            if "super" in opt.schedule:
                scheduler.step()
            else:
                adjust_learning_rate(optimizer, iteration, opt)

        # validation part.
        # To see training progress, we also conduct validation when 'iteration == 1'
        if iteration % opt.val_interval == 0 or iteration == 1:
            # for validation log
            with open(f"./saved_models/{opt.exp_name}/log_train.txt", "a") as log:
                model.eval()
                with torch.no_grad():
                    (
                        valid_loss,
                        current_score,
                        preds,
                        confidence_score,
                        labels,
                        infer_time,
                        length_of_data,
                    ) = validation(model, criterion, valid_loader, converter, opt)
                model.train()

                # keep best score (accuracy or norm ED) model on valid dataset
                # Do not use this on test datasets. It would be an unfair comparison
                # (training should be done without referring test set).
                if current_score > best_score:
                    best_score = current_score
                    torch.save(
                        model.state_dict(),
                        f"./saved_models/{opt.exp_name}/best_score.pth",
                    )

                # validation log: loss, lr, score (accuracy or norm ED), time.
                lr = optimizer.param_groups[0]["lr"]
                elapsed_time = time.time() - start_time
                valid_log = f"\n[{iteration}/{opt.num_iter}] Train_loss: {train_loss_avg.val():0.5f}, Valid_loss: {valid_loss:0.5f}"
                valid_log += f'{"Current_score":17s}: {current_score:0.2f}, Current_lr: {lr:0.7f}\n'
                valid_log += f'{"Best_score":17s}: {best_score:0.2f}, Infer_time: {infer_time:0.1f}, Elapsed_time: {elapsed_time:0.1f}'

                # show some predicted results
                dashed_line = "-" * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"
                for gt, pred, confidence in zip(
                    labels[:5], preds[:5], confidence_score[:5]
                ):
                    if "Att" in opt.Prediction:
                        gt = gt[: gt.find("[EOS]")]
                        pred = pred[: pred.find("[EOS]")]

                    predicted_result_log += f"{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
                predicted_result_log += f"{dashed_line}"
                valid_log = f"{valid_log}\n{predicted_result_log}"
                print(valid_log)
                log.write(valid_log + "\n")

                opt.writer.add_scalar(
                    "train/train_loss", float(f"{train_loss_avg.val():0.5f}"), iteration
                )
                opt.writer.add_scalar("train/lr", float(f"{lr:0.7f}"), iteration)
                opt.writer.add_scalar(
                    "train/elapsed_time", float(f"{elapsed_time:0.1f}"), iteration
                )
                opt.writer.add_scalar(
                    "valid/valid_loss", float(f"{valid_loss:0.5f}"), iteration
                )
                opt.writer.add_scalar(
                    "valid/current_score", float(f"{current_score:0.2f}"), iteration
                )
                opt.writer.add_scalar(
                    "valid/best_score", float(f"{best_score:0.2f}"), iteration
                )

                train_loss_avg.reset()

    """ Evaluation at the end of training """
    print("Start evaluation on benchmark testset")
    """ keep evaluation model and result logs """
    os.makedirs(f"./result/{opt.exp_name}", exist_ok=True)
    os.makedirs(f"./evaluation_log", exist_ok=True)
    saved_best_model = f"./saved_models/{opt.exp_name}/best_score.pth"
    model.load_state_dict(torch.load(f"{saved_best_model}"))

    opt.eval_type = "benchmark"
    model.eval()
    with torch.no_grad():
        total_accuracy, eval_data_list, accuracy_list = benchmark_all_eval(
            model, criterion, converter, opt
        )

    opt.writer.add_scalar(
        "test/total_accuracy", float(f"{total_accuracy:0.2f}"), iteration
    )
    for eval_data, accuracy in zip(eval_data_list, accuracy_list):
        accuracy = float(accuracy)
        opt.writer.add_scalar(f"test/{eval_data}", float(f"{accuracy:0.2f}"), iteration)

    print(
        f'finished the experiment: {opt.exp_name}, "CUDA_VISIBLE_DEVICES" was {opt.CUDA_VISIBLE_DEVICES}'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        default="data_CVPR2021/training/label/",
        help="path to training dataset",
    )
    parser.add_argument(
        "--valid_data",
        default="data_CVPR2021/validation/",
        help="path to validation dataset",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of data loading workers"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument(
        "--num_iter", type=int, default=200000, help="number of iterations to train for"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=20000,
        help="Interval between each validation",
    )
    parser.add_argument(
        "--log_multiple_test", action="store_true", help="log_multiple_test"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=5, help="gradient clipping value. default=5"
    )
    """ Optimizer """
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer |sgd|adadelta|adam|"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="learning rate, default=1.0 for Adadelta, 0.0005 for Adam",
    )
    parser.add_argument(
        "--sgd_momentum", default=0.9, type=float, help="momentum for SGD"
    )
    parser.add_argument(
        "--sgd_weight_decay", default=0.000001, type=float, help="weight decay for SGD"
    )
    parser.add_argument(
        "--adamw_weight_decay", default=0.05, type=float, help="weight decay for adamw"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=20000, help="number of iterations warm up for AdamW"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.95,
        help="decay rate rho for Adadelta. default=0.95",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8, help="eps for Adadelta. default=1e-8"
    )
    parser.add_argument(
        "--schedule",
        default="super",
        nargs="*",
        help="(learning rate schedule. default is super for super convergence, 1 for None, [0.6, 0.8] for the same setting with ASTER",
    )
    parser.add_argument(
        "--lr_drop_rate",
        type=float,
        default=0.1,
        help="lr_drop_rate. default is the same setting with ASTER",
    )
    """ Model Architecture """
    parser.add_argument("--model_name", type=str, required=True, help="TRA|TRC")
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=3,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )
    """ Data processing """
    parser.add_argument(
        "--select_data",
        type=str,
        default="label",
        help="select training data. default is `label` which means 11 real labeled datasets",
    )
    parser.add_argument(
        "--batch_ratio",
        type=str,
        help="assign ratio for each selected data in the batch",
    )
    parser.add_argument(
        "--total_data_usage_ratio",
        type=str,
        default="1.0",
        help="total data usage ratio, this ratio is multiplied to total number of data.",
    )
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    parser.add_argument(
        "--character",
        type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        help="character label",
    )
    parser.add_argument(
        "--NED", action="store_true", help="For Normalized edit_distance"
    )
    parser.add_argument(
        "--Aug",
        type=str,
        default="None",
        help="whether to use augmentation |None|light|",
    )
    
    """ exp_name and etc """
    parser.add_argument("--exp_name", help="Where to store logs and models")
    parser.add_argument(
        "--manual_seed", type=int, default=111, help="for random seed setting"
    )
    parser.add_argument(
        "--saved_model", default="", help="path to model to continue training"
    )
    parser.add_argument(
        "--random_init", action="store_true", help="For random init"
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="A",
        help="select the module to be decoder, |A|BA|",
    )
    parser.add_argument(
        "--part",
        type=str,
        default="label_100",
        help="select the label ratio, |label_10|label_100|label_1|",
    )

    opt = parser.parse_args()

    if opt.model_name == "TRA":  # TRA
        opt.Transformation = "TPS"
        opt.FeatureExtraction = "ResNet"
        opt.SequenceModeling = "None"
        opt.Prediction = "Attn"
    
    elif opt.model_name == "TRC":  # TRC
        opt.Transformation = "TPS"
        opt.FeatureExtraction = "ResNet"
        opt.SequenceModeling = "None"
        opt.Prediction = "CTC"

    """ Seed and GPU setting """
    os.environ['PYTHONHASHSEED'] = str(opt.manual_seed)
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)  # if you are using multi-GPU.
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = False  # It fasten training.
    cudnn.deterministic = True

    opt.gpu_name = "_".join(torch.cuda.get_device_name().split())
    if sys.platform == "linux":
        opt.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        opt.CUDA_VISIBLE_DEVICES = 0  # for convenience
    opt.num_gpu = torch.cuda.device_count()
    if opt.num_gpu > 1:
        print(
            "We recommend to use 1 GPU, check your GPU number, you would miss CUDA_VISIBLE_DEVICES=0 or typo"
        )
        print("To use multi-gpu setting, remove or comment out these lines")
        sys.exit()

    if sys.platform == "win32":
        opt.workers = 0

    """ directory and log setting """
    if not opt.exp_name:
        opt.exp_name = f"Seed{opt.manual_seed}-{opt.model_name}"

    os.makedirs(f"./saved_models/{opt.exp_name}", exist_ok=True)
    log = open(f"./saved_models/{opt.exp_name}/log_train.txt", "a")
    command_line_input = " ".join(sys.argv)
    print(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}"
    )
    log.write(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}\n"
    )
    os.makedirs(f"./tensorboard", exist_ok=True)
    opt.writer = SummaryWriter(log_dir=f"./tensorboard/{opt.exp_name}")

    train(opt, log)
