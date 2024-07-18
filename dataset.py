import torch
import pickle
import math
import random
import yaml
import json
import numpy as np
import torch.utils.data as data
import os
import os.path as osp
import scipy.interpolate

from yaml.loader import FullLoader
from torch.utils.data import DataLoader

def resize_feature(input_data, new_size):
    assert len(input_data) > 1
    x = np.arange(len(input_data))
    f = scipy.interpolate.interp1d(x, input_data, axis=0)
    x_new = [i*float(len(input_data)-1)/(new_size-1) for i in range(new_size)]
    return f(x_new)

def calculate_iou(prediction:list, answer:list):
    intersection = -1
    s1, e1 = prediction
    s2, e2 = answer
    if s1 > s2:
        s1, s2 = s2, s1
        e1, e2 = e2, e1
    if e1 <= s2:
        intersection = 0
    else:
        if e2 <= e1:
            intersection = (e2 - s2)
        else:
            intersection = (e1 - s2)
    l1 = e1 - s1
    l2 = e2 - s2
    iou = intersection/(l1 + l2 - intersection + 1e-8)
    return iou

class Classifier_Dataset(data.Dataset):
    def __init__(self, config, subset):
        super().__init__()
        self.config = config
        self.subset = subset
        with open(config['annotation_path'], encoding='utf-8') as f:
            self.meta = json.load(f)
        self.idxes = self.get_db_idxes(subset)
        
    def __len__(self):
        return len(self.idxes)
        
    def __getitem__(self, index):
        name, _ = self.idxes[index]
        feature = np.load(os.path.join(self.config['feature_path'], f'{name}.npy'))
        original_feature_len = len(feature)
        duration = self.meta['database'][name]['duration']
        annotations = self.meta['database'][name]['annotations']
        cnt = 0
        randflag = False
        while True:
            annotation = random.choice(annotations)
            st, ed = annotation['segment']
            st_idx = int((st/duration)*original_feature_len)
            ed_idx = int((ed/duration)*original_feature_len)
            if ed <= duration and st_idx != ed_idx:
                break
            cnt += 1
            if cnt > 10:
                randflag = True
                break
        if random.random() > self.config['foreground_ratio'] or len(annotations) == 0 or randflag:
            #random
            if random.random() > self.config['long_instance_ratio']:
                #short instance
                instance_length = max(1, min(int(random.random() * self.config['unit_length']), len(feature)))
            else:
                #long instance
                instance_length = min(max(self.config['unit_length'], int(random.random() * self.config['max_length'])), len(feature))
            instance_start_idx = random.randint(0, len(feature) - instance_length)
            instance_end_idx = instance_start_idx+instance_length
            feature = feature[instance_start_idx:instance_end_idx]
            iou = 0
            label_idx = 0
            for annotation in annotations:
                st, ed = annotation['segment']
                st_idx = int((st/duration)*original_feature_len)
                ed_idx = int((ed/duration)*original_feature_len)
                tmp_iou = calculate_iou([instance_start_idx, instance_end_idx], [st_idx, ed_idx])
                if tmp_iou > iou:
                    iou = tmp_iou
                    if iou > self.config['iou_threshold']:
                        label_idx = annotation['labelIndex'] + 1
        else: 
            #foreground
            cnt = 0
            while True:
                st, ed = annotation['segment']
                st_idx = int((st/duration)*original_feature_len)
                ed_idx = int((ed/duration)*original_feature_len)
                if ed <= duration and st_idx != ed_idx:
                    break
                cnt += 1
                if cnt > 200:
                    print(name)
                    raise Exception()
            st, ed = annotation['segment']
            center = (st+ed)/2
            instance_length = (ed-st)
            start_candidate = [max(0, st - (instance_length/2)), center]
            end_candidate = [center, min(ed + (instance_length/2), duration)]
            cnt = 0
            while True:
                start = random.uniform(*start_candidate)
                end = random.uniform(*end_candidate)
                if calculate_iou([start, end], annotation['segment']) > self.config['iou_threshold']:
                    instance_start_idx = int((start/duration)*original_feature_len)
                    instance_end_idx = int((end/duration)*original_feature_len)
                    if instance_end_idx != instance_start_idx:
                        iou = calculate_iou([start, end], annotation['segment'])
                        label_idx = annotation['labelIndex'] + 1
                        break
                cnt += 1
                if cnt > 200:
                    print(name)
                    raise Exception()
            instance_start_idx = int((start/duration)*original_feature_len)
            instance_end_idx = int((end/duration)*original_feature_len)
            feature = feature[instance_start_idx:instance_end_idx]
        if len(feature) < self.config['unit_length']:
            #padding
            feature_len = len(feature)
            padding_size = self.config['unit_length'] - len(feature)
            padding = np.zeros([padding_size, feature.shape[1]])
            feature = np.concatenate([feature, padding], axis=0)
        elif len(feature) > self.config['unit_length']:
            #linear interpolate
            feature_len = self.config['unit_length']
            feature = resize_feature(feature, self.config['unit_length'])
        else:
            #pittari!
            feature_len = self.config['unit_length']
        mask = np.array([False for _ in range(self.config['unit_length'])])
        mask[feature_len:] = True
        feature = feature.astype(np.float32)
        return feature, mask, label_idx
    
    def get_db_idxes(self, subset):
        idxes = []
		# [ [file_name, cnt_idx] ... ]
        for name in self.meta['database']:
            if self.meta['database'][name]['subset'] == subset:
                feature_len = len(np.load(os.path.join(self.config['feature_path'], f'{name}.npy')))
                chunk_num = feature_len//self.config['unit_length']
                idxes.extend([[name, i] for i in range(chunk_num)])
        return idxes

class Sliced_Dataset(data.Dataset):
    def __init__(self, config, subset):
        super().__init__()
        self.config = config
        self.subset = subset
        with open(config['annotation_path'], encoding='utf-8') as f:
            self.meta = json.load(f)
        self.idxes = self.get_db_idxes(config['classifier_epoch_iter'])
        self.folder_list = os.listdir(osp.join(config['classifier_feature_path'], subset))
        folder_prob = dict()
        cnt = 0
        for folder in self.folder_list:
            path = osp.join(config['classifier_feature_path'], subset, folder)
            cnt += len(os.listdir(path))
        for folder in self.folder_list:
            path = osp.join(config['classifier_feature_path'], subset, folder)
            folder_prob[int(folder)] = len(os.listdir(path))/cnt
        keys = sorted(list(folder_prob.keys()))
        self.folder_prob = []
        for key in keys:
            self.folder_prob.append(folder_prob[key])

        
    def __len__(self):
        return len(self.idxes)
        
    def __getitem__(self, index):
        eqaul_sampling = self.config['classifier_equal_sampling']
        if eqaul_sampling or self.subset == 'test':
            folder = random.sample(self.folder_list, 1)[0]
            base_path = osp.join(self.config['classifier_feature_path'], self.subset, folder)
            filename = random.sample(os.listdir(base_path), 1)[0]
            full_path = osp.join(base_path, filename)
            tmp = np.load(full_path, allow_pickle=True).item()
            #print(tmp['feature'].shape, tmp['a_b'], tmp['labelIndex'])
        else:
            p = np.array(self.folder_prob)
            p = p**self.config['classifier_temperature']
            
            p /= sum(p)

            folder = str(np.random.choice(len(self.folder_list), 1, p=p)[0])
            base_path = osp.join(self.config['classifier_feature_path'], self.subset, folder)
            filename = random.sample(os.listdir(base_path), 1)[0]
            full_path = osp.join(base_path, filename)
            tmp = np.load(full_path, allow_pickle=True).item()
        original_feature = tmp['feature']
        a_b = tmp['a_b']
        labelIndex = tmp['labelIndex']
        if len(original_feature) > 5:
            while True:
                a = random.normalvariate(self.config['a'], self.config['a_std'])
                while a < 0 or a > 1:
                    a = random.normalvariate(self.config['a'], self.config['a_std'])
                b = random.normalvariate(self.config['b'], self.config['b_std'])
                while b < 0 or b > 1:
                    b = random.normalvariate(self.config['b'], self.config['b_std'])
                if a > b:
                    a, b = b, a
                st_idx, ed_idx = round(len(original_feature)*a), round(len(original_feature)*b)
                if ed_idx - st_idx > 0:
                    break
        else:
            a, b = a_b[0], a_b[1]
            st_idx, ed_idx = round(len(original_feature)*a), round(len(original_feature)*b)
            assert ed_idx - st_idx > 0
        feature = original_feature[st_idx:ed_idx]
        a, b = a_b[0], a_b[1]
        iou = calculate_iou([round(len(original_feature)*a), round(len(original_feature)*b)], [st_idx, ed_idx])
        if iou > self.config['iou_threshold']:
            label_idx = labelIndex + 1
        else: 
            label_idx = 0
        if len(feature) < self.config['unit_length']:
            #padding
            feature_len = len(feature)
            padding_size = self.config['unit_length'] - len(feature)
            padding = np.zeros([padding_size, feature.shape[1]])
            feature = np.concatenate([feature, padding], axis=0)
        elif len(feature) > self.config['unit_length']:
            #linear interpolate
            feature_len = self.config['unit_length']
            feature = resize_feature(feature, self.config['unit_length'])
        else:
            #pittari!
            feature_len = self.config['unit_length']
        mask = np.array([False for _ in range(self.config['unit_length'])])
        mask[feature_len:] = True
        feature = feature.astype(np.float32)
        return feature, mask, label_idx
    
    def get_db_idxes(self, length):
        idxes = [0 for _ in range(length)]
        return idxes

class OAD_Dataset(data.Dataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.mode = mode
        
        self.dt_iteration = config['dt_iteration']
        self.feature_path = config['feature_path']
        self.label_path = config['state_label_path']
        self.oracle_label_path = self.label_path
        self.n_feature = config['n_feature']
        self.history_length = config['history_length']
        self.length_proportional_sampling = config['length_proportional_sampling']
        self.n_state = config['n_state']
        if config['dataset'] == 'fineaction':
            print('load feature_len...')
            with open(config['feature_len_path'], 'r', encoding='utf-8') as f:
                self.feature_len = json.load(f)
        with open(config['annotation_path'], 'r', encoding='utf-8') as f:
            self.meta = json.load(f)['database']
        feature_names = []
        for key in self.meta:
            if self.meta[key]['subset'] == mode:
                feature_names.append(f'{key}.npy')
        self.feature_names = feature_names
        prob_dict = {}
        if self.length_proportional_sampling and self.mode == 'train':
            cum_length = 0
            for feature_name in feature_names:
                path = os.path.join(self.feature_path, feature_name)
                if config['dataset'] == 'fineaction':
                    vidname = feature_name[:-4]
                    feature_length = self.feature_len[vidname]
                else:
                    feature_length = len(np.load(path))
                prob_dict[feature_name] = feature_length
                cum_length += feature_length
            for feature_name in feature_names:
                prob_dict[feature_name] = prob_dict[feature_name]/cum_length
        elif self.mode != 'train':
            prob_dict = None
        else:
            for feature_name in feature_names:
                prob_dict[feature_name] = 1/len(feature_names)
        self.prob_dict = prob_dict

    def __len__(self):
        if self.mode == 'train':
            return self.dt_iteration
        else:
            return len(self.feature_names)

    def __getitem__(self, index):
        if self.mode == 'train':
            name = random.choices(list(self.prob_dict.keys()), weights=self.prob_dict.values(), k=1)[0]
            path = os.path.join(self.feature_path, name)
            feature = np.load(path, mmap_mode='r')
            label_path_1 = f'{self.label_path}_1'
            label_path_2 = f'{self.label_path}_1'
            label_1 = np.load(os.path.join(label_path_1, name))
            label_2 = np.load(os.path.join(label_path_2, name))
            if len(feature) <= self.history_length:
                s = np.zeros((self.history_length, self.n_feature), dtype=np.float32)
                target_a_1 = np.zeros((self.history_length,), dtype=np.int64)
                target_a_2 = np.zeros((self.history_length,), dtype=np.int64)
                start_idx = 0
                s[start_idx:len(feature)] = feature[start_idx:len(feature)]
                target_a_1[start_idx:len(feature)] = label_1[start_idx:len(feature)]
                target_a_2[start_idx:len(feature)] = label_2[start_idx:len(feature)]
            else:
                start_idx = random.randint(0, len(feature)-self.history_length)
                s = feature[start_idx:start_idx+self.history_length]
                target_a_1 = label_1[start_idx:start_idx+self.history_length]
                target_a_2 = label_2[start_idx:start_idx+self.history_length]
            s = s.astype(np.float32)
            target_a_1 = target_a_1.astype(np.int64)
            target_a_2 = target_a_2.astype(np.int64)
            assert len(s) == len(target_a_1) and len(s) == len(target_a_2)
            return s, target_a_1, target_a_2
        else:
            #test
            name = self.feature_names[index]
            path = os.path.join(self.feature_path, name)
            s = np.load(path, mmap_mode='r')
            s = s.astype(np.float32)
            name = name[:-4]
            return s, name