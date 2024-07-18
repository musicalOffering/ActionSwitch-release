import json
import os
from pydoc import classname
import torch
import yaml
import random
import argparse
import numpy as np

from yaml.loader import FullLoader
from model import Classifier
from dataset import resize_feature
from tqdm import tqdm
from utils import get_idx_and_confidence, get_idx_and_confidences


'''
IN: segment json
{
    results: {
        vid_name: [
            {
                "segment": [
                    2,
                    7
                ]
            },
            {
                ...
            },
        ]
    }
}

OUT: labeled json
{
    results: {
        vid_name: [
            {
                "label": "quarrel",
                "segment": [
                    2.33,
                    7.66
                ],
                "score": 0.90012
            },
            {
                ...
            },
        ]
    }
}
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='yamls/canonical.yaml')
    parser.add_argument('--source', type=str, default='canonical.json')
    parser.add_argument('--target', type=str, default='canonical.json')
    parser.add_argument('--load_model', type=str, default='standard_EqFalse_Epoch12.pt')
    parser.add_argument('--duplicate_proposal_num', type=int, default=1)
    args = parser.parse_args()  
    yaml_path = args.yaml_path
    source = args.source
    target = args.target
    load_model_name = args.load_model
    duplicate_proposal_num = args.duplicate_proposal_num
    with open(yaml_path, encoding='utf-8') as f:
        config = yaml.load(f, FullLoader)
    with open(config['annotation_path'], encoding='utf-8') as f:
        meta = json.load(f)
    with open(source, encoding='utf-8') as f:
        source = json.load(f)
    classes = meta['classes']
    classifier = Classifier(config)
    classifier.load_state_dict(torch.load(os.path.join(config['classifier_model_path'], load_model_name)))
    classifier.eval()
    out_dict = {'results': dict()}
    for filename in tqdm(source['results']):
        segment_list = []
        meta_duration = meta['database'][filename]["duration"]
        feature = np.load(os.path.join(config['feature_path'], f'{filename}.npy'))
        shrink_ratio = meta_duration/len(feature)
        for segment in source['results'][filename]:
            st, ed = segment['segment']
            if st == 0 and ed == 0:
                segment_list.append({'label': list(classes.keys())[0], 'segment': [st, ed], 'score': 0})
                continue
            proposal_feature = feature[st:ed]
            if len(proposal_feature) < config['unit_length']:
                #padding
                proposal_feature_len = len(proposal_feature)
                padding_size = config['unit_length'] - len(proposal_feature)
                padding = np.zeros([padding_size, proposal_feature.shape[1]])
                proposal_feature = np.concatenate([proposal_feature, padding], axis=0)
            elif len(proposal_feature) > config['unit_length']:
                #linear interpolate
                proposal_feature_len = config['unit_length']
                proposal_feature = resize_feature(proposal_feature, config['unit_length'])
            else:
                #pittari!
                proposal_feature_len = config['unit_length']
            mask = np.array([False for _ in range(config['unit_length'])])
            mask[proposal_feature_len:] = True
            proposal_feature = torch.from_numpy(proposal_feature.astype(np.float32)).to(config['device']).unsqueeze(0)
            mask = torch.from_numpy(mask).to(config['device']).unsqueeze(0)
            with torch.no_grad():
                score = classifier(proposal_feature, mask)
                score = torch.softmax(score, dim=1).squeeze().cpu().numpy()[1:]
            if duplicate_proposal_num == 1:
                #standard assignment      
                class_idx, confidence = get_idx_and_confidence(score, [float(st*shrink_ratio), float(ed*shrink_ratio)], segment_list, classes)
                for key in classes.keys():
                    idx = classes[key]
                    if idx == class_idx:
                        class_name = key
                        break
                _st, _ed = float(st*shrink_ratio), float(ed*shrink_ratio)
                segment_list.append({'label': class_name, 'segment': [_st, _ed], 'score': confidence})
            elif duplicate_proposal_num > 1:
                #duplicated assignment
                class_idxes_and_confidences = get_idx_and_confidences(score, n=duplicate_proposal_num)
                for i_c in class_idxes_and_confidences:
                    class_idx = i_c[0]
                    confidence = i_c[1]
                    for key in classes.keys():
                        idx = classes[key]
                        if idx == class_idx:
                            class_name = key
                            break
                    _st, _ed = float(st*shrink_ratio), float(ed*shrink_ratio)
                    segment_list.append({'label': class_name, 'segment': [_st, _ed], 'score': confidence})
            else:
                raise NotImplementedError()
        out_dict['results'][filename] = segment_list
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f)


        
    

    

    
    