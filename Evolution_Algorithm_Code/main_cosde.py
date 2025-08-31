import os
import time
import shutil
import logging
from datetime import datetime
from collections import OrderedDict
from itertools import islice
import numpy as np
import random as pyrandom
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from utils.snn_model import SEW
from utils.data import population_info as pop_info
from utils.tools import resume
from utils.tools.utility import *
from utils.tools.option_de import args, args_text, amp_autocast, obtain_loader
from utils.tools.de import de
from utils.tools.pso import PSOOptimizer
from utils.tools.spe import model_dict_to_vector, model_vector_to_dict
from utils.tools import val
from utils.tools.greedy_soup_ann import greedy_soup, test_ood, test_single_model_ood
from spikingjelly.clock_driven import functional

_logger = logging.getLogger('train')

def main():
    os.environ['WANDB_MODE'] = 'offline'
    setup_default_logging(log_path=args.log_dir)
    logging.basicConfig(level=logging.DEBUG, filename=args.log_dir, filemode='a')

    # rank may not exist outside DDP
    random_seed(args.seed, getattr(args, 'rank', 0))

    # loaders
    _, loader_eval, loader_de = obtain_loader(args)

    # val.validate expects args.slice_len
    if not hasattr(args, 'slice_len'):
        args.slice_len = getattr(args, 'de_slice_len', 0)

    # model
    model = SEW.resnet34(num_classes=args.num_classes, g="add", down='max', T=4)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if getattr(args, 'local_rank', 0) == 0:
        _logger.info(f"Creating model...{args.model}, number of params: {n_parameters}")
    model = model.cuda().to(memory_format=torch.channels_last)

    # seed population from checkpoints
    load_score = False
    models_path = []
    if getattr(args, 'pop_init', None):
        base = os.path.basename(args.pop_init)
        if base.split('_')[-1] == 'score.txt':
            load_score = True
            score, acc1, acc5, val_loss, en_metrics, models_path = pop_info.get_path_with_acc(args.pop_init)
        else:
            models_path = pop_info.get_path(args.pop_init)

    if not models_path:
        # fallback: replicate one ckpt if none given
        models_path = [
            r"E:\PSO\PSOCADE4SNN-main\finetune_hyperparameter\checkpoints\train\20250825-211457-exp_debug\exp_debug_0.pt",
        ] * max(1, args.popsize)

    population = []
    for resume_path in models_path:
        if getattr(args, 'local_rank', 0) == 0:
            print(resume_path)
        resume.load_checkpoint(model, resume_path, log_info=getattr(args, 'local_rank', 0) == 0)
        solution = model_dict_to_vector(model).detach()
        population.append(solution)

    # diversity injection if all clones
    try:
        if len(population) >= 2:
            _stack = torch.stack(population)
            if (_stack[1:] - _stack[:1]).abs().max().item() < 1e-12:
                _base_std = _stack.std().item()
                _noise_scale = max(1e-3 * (_base_std if _base_std > 0 else 1.0), 1e-4)
                population = [p + _noise_scale * torch.randn_like(p) for p in population]
                if getattr(args, 'local_rank', 0) == 0:
                    _logger.info(f"Population cloned; injected Gaussian noise with scale {_noise_scale:.2e}")
    except Exception as _e:
        if getattr(args, 'local_rank', 0) == 0:
            _logger.warning(f"Gaussian diversity injection skipped due to error: {_e}")

    # optional DDP
    if getattr(args, 'distributed', False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = NativeDDP(model, device_ids=[getattr(args, 'local_rank', 0)])

    # output dir snapshot
    if getattr(args, 'local_rank', 0) == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), args.exp_name])
        output_dir = get_outdir(output_base, 'train', exp_name, inc=True)
        args.output_dir = output_dir
        current_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.copytree(os.path.join(current_dir, 'utils'), os.path.join(output_dir, 'utils'))
        for filename in os.listdir(current_dir):
            if filename.endswith('.py') or filename.endswith('.sh'):
                shutil.copy(os.path.join(current_dir, filename), os.path.join(output_dir, filename))
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # baseline scoring
    args.popsize = len(models_path)
    acc1, acc5, val_loss = [torch.zeros(args.popsize).tolist() for _ in range(3)]

    if not load_score:
        score = score_func(model, population, loader_de, args)

        if getattr(args, 'test_ood', False):
            greedy_model_dict = greedy_soup(population, score, model, loader_de, args, amp_autocast=amp_autocast)
            model.load_state_dict(greedy_model_dict)
            greedy_metrics = val.validate(model, loader_eval, args, amp_autocast=amp_autocast)
            ood_metrics = test_ood(greedy_model_dict, args, model, population, args.popsize, amp_autocast=amp_autocast)

        for i in range(args.popsize):
            solution = population[i]
            model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
            model.load_state_dict(model_weights_dict)
            temp = val.validate(model, loader_eval, args, amp_autocast=amp_autocast)
            acc1[i], acc5[i], val_loss[i] = temp['top1'], temp['top5'], temp['loss']

        en_metrics = val.validate_ensemble(model, population, args.popsize, loader_eval, args, amp_autocast=amp_autocast)

        if getattr(args, 'pop_init', None):
            pop_info.write_path_with_acc(score, acc1, acc5, val_loss, en_metrics, models_path, args.pop_init)
    else:
        en_metrics = en_metrics

    if getattr(args, 'local_rank', 0) == 0:
        update_summary('baselines:', OrderedDict(en_metrics), os.path.join(args.output_dir, 'summary.csv'), write_header=True)
        if getattr(args, 'test_ood', False):
            update_summary('greedy_val:', greedy_metrics, os.path.join(args.output_dir, 'summary.csv'), write_header=True)
            update_summary('greedy_and_ensemble_ood:', ood_metrics, os.path.join(args.output_dir, 'summary.csv'), write_header=True)

    rowd = OrderedDict([('score', score), ('top1', acc1), ('top5', acc5), ('val_loss', val_loss)])
    if getattr(args, 'local_rank', 0) == 0:
        bestidx = score.index(max(score))
        _logger.info('epoch:{}, best_score:{:>7.4f}, best_idx:{}, score: {}'.format(0, max(score), bestidx, score))
        update_summary(0, rowd, os.path.join(args.output_dir, 'summary.csv'), write_header=True)

    print("score", score)

    # ---- PSO: tune (F, CR) before DE (maximization) ----
    f, cr = args.f_init, args.cr_init
    if getattr(args, 'use_pso', False):
        if getattr(args, 'local_rank', 0) == 0:
            _logger.info(f"Starting PSO to tune (F, CR) with pop={args.pso_popsize}, iters={args.pso_iters}, eval_gens={args.pso_eval_gens}")

        def _eval_particle(_particle):
            _F, _CR = float(_particle[0]), float(_particle[1])
            _temp_pop = [p.clone() for p in population]
            _best_scores = []
            for _ in range(max(1, args.pso_eval_gens)):
                _temp_pop, _upd, _scores = de(args.popsize, _F, _CR, _temp_pop, model, loader_de, args)
                try:
                    _best_scores.append(float(max(_scores)))
                except Exception:
                    _best_scores.append(0.0)
            return float(np.mean(_best_scores))  # maximize accuracy

        _pso = PSOOptimizer(pop_size=args.pso_popsize, F_bounds=(0.1, 0.9), CR_bounds=(0.1, 0.99), max_iters=args.pso_iters)
        gbest, gbest_fit = _pso.optimize(_eval_particle)
        f, cr = float(gbest[0]), float(gbest[1])
        if getattr(args, 'local_rank', 0) == 0:
            _logger.info(f"PSO selected F={f:.4f}, CR={cr:.4f} (fitness={gbest_fit:.4f})")
    else:
        if getattr(args, 'local_rank', 0) == 0:
            _logger.info(f"Using initial F={f:.4f}, CR={cr:.4f} without PSO")

    max_acc_train = 0
    max_acc_val = 0

    # ---- DE generations (use all) ----
    for epoch in range(args.de_epochs):
        epoch_time = AverageMeter()
        end = time.time()

        print("cr f", cr, f)
        population, update_label, score = de(args.popsize, f, cr, population, model, loader_de, args)

        # cosine drift for F/CR only if PSO disabled
        if not getattr(args, 'use_pso', False):
            fcr_min = 1e-6
            f = fcr_min + (args.f_init - fcr_min) * (1 + math.cos(math.pi * epoch / 40)) / 2 + pyrandom.uniform(fcr_min, args.f_init)
            cr = fcr_min + (args.cr_init - fcr_min) * (1 + math.cos(math.pi * epoch / 40)) / 2 + pyrandom.uniform(fcr_min, args.cr_init)

        if getattr(args, 'local_rank', 0) == 0:
            bestidx = score.index(max(score))
            _logger.info('epoch:{}, best_score:{:>7.4f}, best_idx:{}, score: {}'.format(epoch, max(score), bestidx, score))
            pop_tensor = torch.stack(population)
            if max(score) > max_acc_train:
                max_acc_train = max(score)
                print("Best in train_set update and train acc = ", max_acc_train)
                model_path = os.path.join(args.output_dir, f'train_best_{args.model}.pt')
                print('Saving model to', model_path)
                torch.save(model.state_dict(), model_path)
        else:
            update_label = list(range(args.popsize))
            pop_tensor = torch.stack(population)

        if getattr(args, 'distributed', False):
            torch.cuda.synchronize()
            dist.barrier()
            torch.distributed.broadcast_object_list(update_label, src=0)
            torch.distributed.broadcast(pop_tensor, src=0)

        population = list(pop_tensor)

        for i in range(args.popsize):
            if update_label[i] == 1:
                solution = population[i]
                model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
                model.load_state_dict(model_weights_dict)
                temp = val.validate(model, loader_eval, args, amp_autocast=amp_autocast)
                acc1[i], acc5[i], val_loss[i] = temp['top1'], temp['top5'], temp['loss']
                if acc1[i] > max_acc_val:
                    max_acc_val = acc1[i]
                    print("Best in train_set update and val acc = ", max_acc_val)
                    model_path = os.path.join(args.output_dir, f'val_best_{args.model}.pt')
                    print('Saving best val model to', model_path)
                    torch.save(model.state_dict(), model_path)

        if getattr(args, 'distributed', False):
            torch.cuda.synchronize()

        epoch_time.update(time.time() - end)
        end = time.time()

        if getattr(args, 'local_rank', 0) == 0:
            rowd = OrderedDict([('best_idx', bestidx), ('score', score), ('top1', acc1), ('top5', acc5), ('val_loss', val_loss)])
            update_summary(epoch, rowd, os.path.join(args.output_dir, 'summary.csv'), write_header=True)
            if args.de_epochs - args.popsize <= epoch <= args.de_epochs - 1:
                model_path = os.path.join(args.output_dir, f'{args.exp_name}_{epoch}.pt')
                print('Saving model to', model_path)
                torch.save(model.state_dict(), model_path)

    try:
        import wandb
        wandb.finish()
    except Exception:
        pass
    return


def score_func(model, population, loader_de, args):
    popsize = len(population)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc1_all = torch.zeros(popsize).tolist()
    acc5_all = torch.zeros(popsize).tolist()
    end = time.time()
    model.eval()
    torch.set_grad_enabled(False)
    slice_len = args.de_slice_len or len(loader_de)

    for batch_idx, (input, target) in enumerate(islice(loader_de, slice_len)):
        data_time_m.update(time.time() - end)
        for i in range(0, popsize):
            solution = population[i]
            model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
            model.load_state_dict(model_weights_dict)
            input, target = input.cuda(), target.cuda()
            input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                output, _ = model(input)
            from spikingjelly.clock_driven import functional
            functional.reset_net(model)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if getattr(args, 'distributed', False):
                acc1 = reduce_tensor(acc1, getattr(args, 'world_size', 1))
                acc5 = reduce_tensor(acc5, getattr(args, 'world_size', 1))
            acc1_all[i] += acc1
            acc5_all[i] += acc5
        batch_time_m.update(time.time() - end)
        end = time.time()

    if getattr(args, 'local_rank', 0) == 0:
        print('data_time: {time1.val:.3f} ({time1.avg:.3f})  '
              'batch_time: {time2.val:.3f} ({time2.avg:.3f})  '.format(time1=data_time_m, time2=batch_time_m))

    score = [i.cpu() / slice_len for i in acc1_all]
    return score


if __name__ == '__main__':
    main()
