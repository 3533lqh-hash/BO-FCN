import argparse
import time
import datetime
import os
import shutil
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision import transforms
import optuna
from optuna.trial import TrialState
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # 模型和数据集参数
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcn', 'psp', 'deeplabv3'],
                        help='模型名称 (默认: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50', 'resnet101', 'resnet152'],
                        help='骨干网络名称 (默认: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc')
    parser.add_argument('--base-size', type=int, default=530, help='基础图像大小')
    parser.add_argument('--crop-size', type=int, default=460, help='裁剪图像大小')
    parser.add_argument('--workers', '-j', type=int, default=4, metavar='N', help='数据加载线程数')
    # 训练超参数
    parser.add_argument('--jpu', action='store_true', default=False, help='使用JPU')
    parser.add_argument('--use-ohem', type=bool, default=False, help='使用OHEM损失')
    parser.add_argument('--aux', action='store_true', default=False, help='使用辅助损失')
    parser.add_argument('--aux-weight', type=float, default=0.8, help='辅助损失权重')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N', help='训练批次大小 (默认: 8)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='起始训练轮数 (默认: 0)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='训练总轮数 (默认: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='学习率 (默认: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='动量 (默认: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='权重衰减 (默认: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0, help='预热迭代次数')
    parser.add_argument('--warmup-factor', type=float, default=0.1, help='预热学习率因子')
    parser.add_argument('--warmup-method', type=str, default='linear', help='预热方法')
    # CUDA设置
    parser.add_argument('--no-cuda', action='store_true', default=False, help='禁用CUDA训练')
    parser.add_argument('--local_rank', type=int, default=0)
    # 检查点和日志
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的文件路径')
    parser.add_argument('--save-dir', default='~/.torch/models', help='模型保存目录')
    parser.add_argument('--save-epoch', type=int, default=10, help='每隔多少轮保存一次模型')
    parser.add_argument('--log-dir', default='../runs/logs/', help='日志保存目录')
    parser.add_argument('--log-iter', type=int, default=10, help='每隔多少次迭代打印日志')
    # 验证设置
    parser.add_argument('--val-epoch', type=int, default=2, help='每隔多少轮验证一次')
    parser.add_argument('--skip-val', action='store_true', default=False, help='跳过验证')
    # Optuna 优化设置
    parser.add_argument('--optimize', action='store_true', default=10, help='启用 Optuna 超参数优化')
    parser.add_argument('--n-trials', type=int, default=50, help='Optuna 试验次数')
    args = parser.parse_args()

    # 默认设置
    if args.epochs is None:
        epoches = {'pascal_voc': 50}
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {'pascal_voc': 0.0001}
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
    return args

import pandas as pd  # 用于保存数据到 Excel

class Trainer:
    """训练器类"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # 初始化记录指标的列表
        self.metrics_log = []  # 用于保存每次训练和验证的损失、准确率和 mIoU

        # 数据预处理
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        # 数据加载器
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)
        self.train_loader = data.DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler,
                                            num_workers=args.workers, pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset, batch_sampler=val_batch_sampler,
                                          num_workers=args.workers, pin_memory=True)

        # 模型初始化
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)
        if args.resume:
            if os.path.isfile(args.resume):
                print(f'恢复训练，加载 {args.resume}...')
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # 损失函数
        self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                                               aux_weight=args.aux_weight, ignore_index=-1).to(self.device)

        # 优化器
        params_list = []
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})
        self.optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # 学习率调度器
        self.lr_scheduler = WarmupPolyLR(self.optimizer, max_iters=args.max_iters, power=0.9,
                                         warmup_factor=args.warmup_factor, warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        # 分布式训练
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)

        # 评估指标
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0

    def train(self):
        """训练模型"""
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info(f'开始训练，总轮数: {epochs} = 总迭代次数: {max_iters}')

        self.model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration += 1
            self.lr_scheduler.step()

            images, targets = images.to(self.device), targets.to(self.device)
            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 减少所有GPU的损失以用于日志记录
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            # 计算剩余时间
            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # 记录训练损失
            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(f"Iter: {iteration}/{max_iters} || Lr: {self.optimizer.param_groups[0]['lr']:.6f} || "
                            f"Loss: {losses_reduced.item():.4f} || Cost Time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))} || "
                            f"ETA: {eta_string}")
                # 保存训练损失到日志
                self.metrics_log.append({
                    "Epoch": iteration // self.args.iters_per_epoch,
                    "Iteration": iteration,
                    "Loss": losses_reduced.item(),
                    "pixAcc": None,  # 验证时更新
                    "mIoU": None     # 验证时更新
                })

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(iteration)
                self.model.train()

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(f"总训练时间: {total_training_str} ({total_training_time / max_iters:.4f}s / it)")

        # 保存指标到 Excel
        self.save_metrics_to_excel()

    def validation(self, iteration):
        """验证模型"""
        is_best = False
        self.metric.reset()
        model = self.model.module if self.args.distributed else self.model
        torch.cuda.empty_cache()
        model.eval()

        for i, (image, target, filename) in enumerate(self.val_loader):
            image, target = image.to(self.device), target.to(self.device)
            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info(f"样本: {i + 1}, 验证 pixAcc: {pixAcc:.3f}, mIoU: {mIoU:.3f}")

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best)

        # 更新日志中的验证指标
        self.metrics_log[-1]["pixAcc"] = pixAcc
        self.metrics_log[-1]["mIoU"] = mIoU

        synchronize()

    def save_metrics_to_excel(self):
        """保存指标到 Excel 文件"""
        df = pd.DataFrame(self.metrics_log)  # 将日志转换为 DataFrame
        excel_path = os.path.join(self.args.save_dir, "training_metrics.xlsx")
        df.to_excel(excel_path, index=False)
        logger.info(f"训练指标已保存到 {excel_path}")


def save_checkpoint(model, args, is_best=False):
    """保存检查点"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{args.model}_{args.backbone}_{args.dataset}.pth'
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = f'{args.model}_{args.backbone}_{args.dataset}_best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


def objective(trial, args):
    """Optuna 目标函数"""
    # 定义需要优化的超参数
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  # 学习率
    args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)  # 权重衰减
    args.momentum = trial.suggest_float("momentum", 0.8, 0.99)  # 动量
    args.epochs = trial.suggest_float("epochs", 1, 500, log=True)  # 训练总轮数
    args.warmup_factor = trial.suggest_float("warmup_factor", 0.01, 1, log=True)  # 预热学习因子

    # 初始化训练器
    trainer = Trainer(args)
    trainer.train()

    # 返回验证集上的 mIoU 作为优化目标
    return trainer.best_pred


def optimize_hyperparameters(args):
    """使用 Optuna 优化超参数"""
    study = optuna.create_study(direction="maximize")  # 目标是最大化 mIoU
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)  # 运行指定次数的试验

    # 输出最佳超参数
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    # 使用最佳超参数重新训练模型
    args.lr = study.best_params["lr"]
    args.weight_decay = study.best_params["weight_decay"]
    args.momentum = study.best_params["momentum"]
    args.epochs = study.best_params["epochs"]
    args.warmup_factor = study.best_params["warmup_factor"]
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()

    # 分布式设置
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus

    # 日志设置
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename=f'{args.model}_{args.backbone}_{args.dataset}_log.txt')
    logger.info(f"使用 {num_gpus} 个GPU")
    logger.info(args)

    # 如果启用优化，则运行 Optuna 优化
    if args.optimize:
        optimize_hyperparameters(args)
    else:
        # 否则直接训练
        trainer = Trainer(args)
        trainer.train()
    torch.cuda.empty_cache()



