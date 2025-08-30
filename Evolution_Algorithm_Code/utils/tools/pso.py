#PSOMODIFIED: DE code commented below and replaced by PSO
# from random import random
# from random import sample
# from random import uniform
# import numpy as np
# from utils.tools.utility import *
# import time
# from itertools import islice
# from utils.tools.spe import model_dict_to_vector, model_vector_to_dict
# from utils.tools.option_de import amp_autocast
# from spikingjelly.clock_driven import functional
# 
# 
# def score_func_de(model,indi1,indi2, loader_de, args):
#     batch_time_m = AverageMeter()
#     data_time_m = AverageMeter()
#     acc1_all = torch.zeros(2).tolist()
#     acc5_all = torch.zeros(2).tolist()
#     end = time.time()
#     population = [indi1,indi2]
#     model.eval()
#     torch.set_grad_enabled(False)
#     slice_len = args.de_slice_len or len(loader_de)
#     for batch_idx, (input, target) in enumerate(islice(loader_de, slice_len)):
#         data_time_m.update(time.time() - end)
#         for i in range(0, 2):
#             solution = population[i]
#             model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
#             model.load_state_dict(model_weights_dict)
#             input, target = input.cuda(), target.cuda()
#             input = input.contiguous(memory_format=torch.channels_last)
#             with amp_autocast():
#                 output,output_list = model(input)
#             # if batch_idx==0 and args.local_rank < 2 and i == 0:
#             #     _logger.info('Checking Data >>>> pop: {} input: {}'.format(i, input.flatten()[6000:6005]))
#             functional.reset_net(model)
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             if args.distributed:
#                 acc1 = reduce_tensor(acc1, args.world_size)
#                 acc5 = reduce_tensor(acc5, args.world_size)
#             acc1_all[i] += acc1
#             acc5_all[i] += acc5
#         batch_time_m.update(time.time() - end)
#         end = time.time()
# 
#     if args.local_rank == 0:
#         print('data_time: {time1.val:.3f} ({time1.avg:.3f})  '
#             'batch_time: {time2.val:.3f} ({time2.avg:.3f})  '.format(time1=data_time_m, time2=batch_time_m)) 
# 
#     score = [i.cpu()/slice_len for i in acc1_all]#!!!
#     return score
# 
# def de(popsize, mutate, recombination, population,model,loader_de,args):
#     #--- INITIALIZE A POPULATION (step #1) ----------------+
#     
# #     if not population:
# #         population = []
# #         for i in range(0,popsize):
# #             indv = []
# #             for j in range(len(bounds)):
# #                 indv.append(uniform(bounds[j][0],bounds[j][1]))
# #             population.append(indv)
# #         return population
#     update_label=[0 for i in range(popsize)]
#     gen_scores = [] # score keeping
#     new_population_f = []
#     device = population[0].device
#     dim = len(model_dict_to_vector(model))
#     for j in range(0, popsize):
#         candidates = list(range(0,popsize))
#         candidates.remove(j)
#         random_index = sample(candidates, 3)#随机采样
#         x_t = population[j]
#         # x_1 = np.array(population[random_index[0]].cpu())
#         # x_2 = np.array(population[random_index[1]].cpu())
#         # x_3 = np.array(population[random_index[2]].cpu())
#         # x_t = np.array(population[j].cpu()).copy()  # target individual
#         # x_diff = np.array(x_2) - np.array(x_3)
#         # # v_donor = np.clip(np.array(x_diff) * mutate + x_1, -bounds, bounds)
#         # v_donor = np.array(x_diff) * mutate + x_1
# 
#         # idx = np.random.choice(np.arange(0, len(x_1)), size=int(len(x_1) * recombination), replace=False)  # 重组下标
#         # noidx = np.delete(np.arange(0, len(x_1)), idx)  # 不需要重组的下标
#         # v_trial = np.arange(0, len(x_1),dtype=float)
#         # v_trial[idx.tolist()] = v_donor[idx.tolist()]
#         # v_trial[noidx.tolist()] = x_t[noidx.tolist()]
# # -------
#         x_new = population[random_index[2]] + mutate * (population[random_index[0]] - population[random_index[1]])
#         v_trial = torch.where(torch.rand(dim).to(device) < torch.ones(dim).to(device)*recombination, x_new, population[j])
# # -------
#         # model_vector_to_dict(model=model, weights_vector=v_trial)
# 
#         score  = score_func_de(model,v_trial,x_t,loader_de, args)
#         new_population_f.append(score[0])
#         if score[0]>score[1]:
# #                 print(j,"th individual update")
#             # population[j] = v_trial.copy()
#             population[j] = v_trial
#             gen_scores.append(score[0])
#             update_label[j]=1
#         else:   
# #                 print("not updata")
#             # population[j] = x_t.copy()
#             population[j] = x_t
#             gen_scores.append(score[1])
# 
# #             score_target = cost_func(x_t)
# #             if score_trial < score_target:
# #                 population[j] = v_trial.copy()
# #                 gen_scores.append(score_trial)
# #             else:
# #                 gen_scores.append(score_target)
#     # gen_best = max(gen_scores)                                  # fitness of best individual
#     # gen_sol = population[gen_scores.index(max(gen_scores))]     # solution of best individual
#     bestidx = gen_scores.index(max(gen_scores))
#     print(new_population_f)
#     print(update_label)
# 
#     return population,update_label,gen_scores

# ==================== PSO Implementation ====================

import random
import numpy as np
import torch
from utils.tools.utility import AverageMeter
from itertools import islice

# ========================
# PSO Optimizer for F and CR
# ========================

class PSOOptimizer:
    def __init__(self, pop_size, F_bounds, CR_bounds, max_iters,
                 w_min=0.4, w_max=0.9, c1_min=0.5, c1_max=2.0, c2_min=0.5, c2_max=2.0):
        self.pop_size = pop_size
        self.F_bounds = F_bounds
        self.CR_bounds = CR_bounds
        self.max_iters = max_iters

        # initialize particles
        self.population = [self.random_particle() for _ in range(pop_size)]
        self.velocities = [np.array([0.0, 0.0]) for _ in range(pop_size)]
        self.pbest = list(self.population)
        self.pbest_fitness = [-np.inf for _ in range(pop_size)]

        self.gbest = None
        self.gbest_fitness = -np.inf

        # hyperparameter bounds
        self.w_min, self.w_max = w_min, w_max
        self.c1_min, self.c1_max = c1_min, c1_max
        self.c2_min, self.c2_max = c2_min, c2_max

    def random_particle(self):
        F = random.uniform(*self.F_bounds)
        CR = random.uniform(*self.CR_bounds)
        return np.array([F, CR])

    def cosine_anneal(self, t, T, pmin, pmax):
        return pmin + (pmax - pmin) / 2 * (1 + np.cos(np.pi * t / T))

    def evaluate_fitness(self, acc, prec):
        return 0.5 * acc + 0.5 * prec

    def pareto_front(self, scores):
        # scores: list of (acc, prec)
        pareto = []
        for i, (a1, p1) in enumerate(scores):
            dominated = False
            for j, (a2, p2) in enumerate(scores):
                if j != i and (a2 >= a1 and p2 >= p1) and (a2 > a1 or p2 > p1):
                    dominated = True
                    break
            if not dominated:
                pareto.append(i)
        return pareto

    def optimize(self, eval_func):
        for t in range(self.max_iters):
            # update hyperparameters using cosine annealing
            w = self.cosine_anneal(t, self.max_iters, self.w_min, self.w_max)
            c1 = self.cosine_anneal(t, self.max_iters, self.c1_min, self.c1_max)
            c2 = self.cosine_anneal(t, self.max_iters, self.c2_min, self.c2_max)

            scores, fitnesses = [], []
            for i, particle in enumerate(self.population):
                acc, prec = eval_func(particle)
                fit = self.evaluate_fitness(acc, prec)
                scores.append((acc, prec))
                fitnesses.append(fit)
                if fit > self.pbest_fitness[i]:
                    self.pbest[i] = particle.copy()
                    self.pbest_fitness[i] = fit

            # Pareto front selection
            pareto_idx = self.pareto_front(scores)
            if pareto_idx:
                best_idx = max(pareto_idx, key=lambda idx: fitnesses[idx])
                if fitnesses[best_idx] > self.gbest_fitness:
                    self.gbest = self.population[best_idx].copy()
                    self.gbest_fitness = fitnesses[best_idx]

            # update particles
            for i in range(self.pop_size):
                r1, r2 = random.random(), random.random()
                self.velocities[i] = (w * self.velocities[i]
                                      + c1 * r1 * (self.pbest[i] - self.population[i])
                                      + c2 * r2 * (self.gbest - self.population[i]))
                self.population[i] = self.population[i] + self.velocities[i]
                self.population[i][0] = np.clip(self.population[i][0], *self.F_bounds)
                self.population[i][1] = np.clip(self.population[i][1], *self.CR_bounds)

        return self.gbest, self.gbest_fitness


# ========================
# Score Function (like score_func_de)
# ========================
def score_func_pso(model, indi, loader, args):
    #PSOMODIFIED: New score func using acc+precision
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_all, prec_all = [], []
    end = time.time()
    model.eval()
    torch.set_grad_enabled(False)
    slice_len = args.de_slice_len or len(loader)

    for batch_idx, (input, target) in enumerate(islice(loader, slice_len)):
        data_time_m.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        # Accuracy
        acc1 = (output.argmax(dim=1) == target).float().mean().item()
        # Precision
        true_pos = ((output.argmax(dim=1) == 1) & (target == 1)).sum().item()
        pred_pos = (output.argmax(dim=1) == 1).sum().item()
        precision = true_pos / pred_pos if pred_pos > 0 else 0.0
        acc_all.append(acc1)
        prec_all.append(precision)
        batch_time_m.update(time.time() - end)
        end = time.time()

    return np.mean(acc_all), np.mean(prec_all)
