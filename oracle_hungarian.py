import json
import argparse
from utils import get_hungarian_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='thumos14_v2.json')
    parser.add_argument('--pred', type=str, default='canonical.json')
    parser.add_argument('--tiou', type=float, default=0.5)
    args = parser.parse_args()
    gt = args.gt
    pred = args.pred
    tiou = args.tiou
    with open(gt, encoding='utf-8') as f:
        gt = json.load(f)['database']
    with open(pred, encoding='utf-8') as f:
        pred = json.load(f)['results']
    tp_cum = 0
    p_cum = 0
    a_cum = 0
    for vidname in pred:
        oracle_proposals = []
        annotations = gt[vidname]['annotations']
        for anno in annotations:
            oracle_proposals.append(anno['segment'])
        pred_proposals = []
        for anno in pred[vidname]:
            pred_proposals.append(anno['segment'])
        hungarian_results = get_hungarian_score(oracle_proposals, pred_proposals, tiou)
        tp_cum += hungarian_results['tp']
        p_cum += hungarian_results['p']
        a_cum += hungarian_results['a']
    precision = tp_cum/(p_cum+1e-8)
    recall = tp_cum/(a_cum+1e-8)
    f1 = (2*precision*recall)/(precision+recall+1e-8)
    print(f'tiou: {tiou}')
    print(f'p_cum: {p_cum}')
    print(f'a_cum: {a_cum}')
    print(f'hungarian f1, precision, recall: {f1}, {precision}, {recall}')