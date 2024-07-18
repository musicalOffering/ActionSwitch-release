import yaml
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os

from dataset import OAD_Dataset
from model import OADModel
from yaml.loader import FullLoader
from torch.utils.data import DataLoader
from utils import get_aux_loss, soft_update, get_4state_proposals, get_hungarian_score, g
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='yamls/canonical.yaml')
    args = parser.parse_args()
    yaml_path = args.yaml_path
    with open(yaml_path, encoding='utf-8') as f:
        config = yaml.load(f, FullLoader)

    if not os.path.exists(config['model_save_path']):
        os.mkdir(config['model_save_path'])

    ceweight_str = "["
    for i in config["cross_entropy_weight"]:
        ceweight_str += str(i)
        ceweight_str += '_'
    ceweight_str = ceweight_str[:-1]
    ceweight_str += ']'
    identifier = f'{config["prefix"]}_{config["dataset"]}_{config["n_state"]}_p{config["penalty_coef"]}'
    tb_path = f'{identifier}'
    count = 1
    while(True):
        if not os.path.exists(os.path.join('runs', tb_path, str(count))):
            tb_path = os.path.join('runs', tb_path, str(count))
            break
        else:
            count += 1


    #Agent
    model = OADModel(config).to(config['device'])

    print(f'identifier: {identifier}')
    print(f'dt_iteration: {config["dt_iteration"]}')
    print(f'n_state: {config["n_state"]}')
    print(f'aux_loss: {config["aux_loss"]}')
    print(f'use_target: {config["use_target"]}')
    print(f'history_length: {config["history_length"]}')
    print(f'penalty_coef: {config["penalty_coef"]}')
    print(f'cross_entropy_coef: {config["cross_entropy_weight"]}')

    if config['use_target']:
        print('use target network...')
        target = OADModel(config).to(config['device'])
    scheduler = scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=model.opt, lr_lambda=g(1., 0., config['oad_final_lr']/config['oad_base_lr'], config['oad_warmup_iter'], config['oad_final_iter']))
    train_dataset = OAD_Dataset(config, mode='train')
    test_dataset = OAD_Dataset(config, mode='test')
    trainloader = DataLoader(train_dataset, batch_size=config['dt_batch_size'], shuffle=True, num_workers=config['num_workers'])
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    iter_cnt = 0
    save_cnt = 0
    eval_cnt = 0
    plot_cnt = 0
    f1_cnt = 0

    iter_cnt = 0
    tp_cum = 0
    p_cum = 0
    a_cum = 0

    for epoch in range(config['epoch']):
        print(f'training in epoch {epoch+1}')
        total_correct = 0
        for s, target_a_1, target_a_2 in tqdm(trainloader):
            iter_cnt += 1
            model.train()
            s = s.to(config['device'])
            target_a_1 = target_a_1.to(config['device'])
            target_a_2 = target_a_2.to(config['device'])
            logits, loss = model(s, target_a_1, target_a_2)
            if config['aux_loss'] == 'cross_entropy':
                penalty = get_aux_loss(logits)
            else:
                raise NotImplementedError('invalid aux_loss')
            loss += config['penalty_coef']*penalty
            answer_sheet = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            answer_1 = target_a_1.detach().cpu().numpy()
            answer_2 = target_a_2.detach().cpu().numpy()
            correct_1 = np.sum(np.where(answer_sheet == answer_1, 1, 0))
            correct_2 = np.sum(np.where(answer_sheet == answer_2, 1, 0))
            correct = np.max([correct_1, correct_2])
            total_correct += correct
            model.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)
            model.opt.step()
            scheduler.step()
            if config['use_target']:
                soft_update(model, target, config['tau'])
            loss = loss.detach()
        if epoch+1 == config['epoch']:
            print(f'eval in epoch{epoch+1}')
            with open(config['oracle_proposal_path'], 'rb') as f:
                oracle_proposal_dict = pickle.load(f)
            eval_cnt = 0
            tp_cum = 0
            p_cum = 0
            a_cum = 0
            for feature, name in tqdm(testloader):
                name = name[0]
                atsumari = []
                feature = feature.squeeze().to(config['device'])
                duration = len(feature)
                hs = [torch.zeros(1, config['n_recurrent_hidden']).to(config['device']) for _ in range(config['n_recurrent_layer'])]
                for i in range(duration):
                    with torch.no_grad():
                        snippet = feature[i].unsqueeze(0)
                        if config['use_target']:
                            target.eval()
                            hs, score = target.encode(snippet, hs)
                        else:
                            model.eval()
                            hs, score = model.encode(snippet, hs)
                        a = torch.argmax(score, dim=1).squeeze().cpu().numpy()
                        atsumari.append(a)
                oracle_proposals = oracle_proposal_dict[name]
                pred_proposals = get_4state_proposals(atsumari)
                hungarian_results = get_hungarian_score(oracle_proposals, pred_proposals)
                tp_cum += hungarian_results['tp']
                p_cum += hungarian_results['p']
                a_cum += hungarian_results['a']
                eval_cnt += 1
                if eval_cnt == config['eval_num']:
                    break
            precision = tp_cum/(p_cum+1e-8)
            recall = tp_cum/(a_cum+1e-8)
            f1 = (2*precision*recall)/(precision+recall+1e-8)
            print(f'hungarian f1, recall: {f1}, {recall}')
            print(f'lr: {scheduler.get_last_lr()}')
            
            if config['use_target']:
                torch.save(target.state_dict(), os.path.join(config['model_save_path'], f'{identifier}.pt'))
            else:
                torch.save(model.state_dict(), os.path.join(config['model_save_path'], f'{identifier}.pt'))
        
            
            
