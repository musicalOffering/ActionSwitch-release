import numpy as np
import urllib
import json
from random import uniform
from scipy import optimize


def get_2state_proposals(action_list):
    action_list = np.array(action_list)
    if np.any(action_list > 1):
        print(action_list)
        raise Exception('invalid action_list')
    action_list = action_list.tolist()
    cur_state = 0
    st = 0
    ed = 0
    proposals = []
    for i, state in enumerate(action_list):
        if cur_state == 0 and state == 0:
            cur_state = 0
        elif cur_state == 0 and state == 1:
            st = i
            cur_state = 1
        elif cur_state == 1 and state == 0:
            ed = i
            proposals.append([st, ed])
            cur_state = 0
        else:
            cur_state = 1
    return proposals

def get_4state_proposals(action_list):
    action_list = np.array(action_list)
    action_list = np.where(action_list == 4, 0, action_list)
    if np.any(action_list > 3):
        print(action_list)
        raise Exception('invalid action_list')
    action_list = action_list.tolist()
    button1 = []
    for i in action_list:
        if i == 2:
            i = 0
        elif i == 3:
            i = 1
        button1.append(i)
    button1_proposals = get_2state_proposals(button1)
    button2 = []
    for i in action_list:
        if i == 1:
            i = 0
        elif i == 3:
            i = 2
        button2.append(i)
    button2_proposals = get_2state_proposals([1 if i == 2 else 0 for i in button2])
    button1_proposals.extend(button2_proposals)
    button1_proposals.sort(key=lambda x: x[1])
    return button1_proposals

def calculate_iou(prediction:list, answer:list):
    intersection = -1
    s1 = prediction[0]
    e1 = prediction[1]
    s2 = answer[0]
    e2 = answer[1]
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
    iou = intersection/((l1 + l2 - intersection) + 1e-8)
    return iou

def get_moving_average(l, moving_average_range):
    if len(l) <= moving_average_range:
        return l
    ret = []
    for i in range(moving_average_range, len(l)):
        ret.append(np.mean(l[i-moving_average_range:i]))
    return ret

def get_idx_and_confidence(score, target_segment, segment_list, class_names, iou_threshold=0.5):
    max_iou = -1
    registered_segment = None
    for segment in segment_list:
        iou = calculate_iou(target_segment, segment['segment'])
        if iou > max_iou:
            max_iou = iou
            registered_segment = segment
    if max_iou < iou_threshold:
        registered_segment = None
    if registered_segment is None:
        class_idx = np.argmax(score)
        confidence = float(score[class_idx])
    else:
        i = class_names[registered_segment['label']]
        score[i] = 0
        class_idx = np.argmax(score)
        confidence = float(score[class_idx])
    return class_idx, confidence

def get_idx_and_confidences(score, n):
    class_idxes_and_confidences = []
    for _ in range(n):
        class_idx = np.argmax(score)
        confidence = float(score[class_idx])
        class_idxes_and_confidences.append([class_idx, confidence])
        score[class_idx] = 0
    return class_idxes_and_confidences

def get_hungarian_score(answer:list, prediction:list, iou_threshold=0.5):
    #IN: answer[[st,ed], [st,ed]...], prediction[[st,ed],[st,ed]...]
    #OUT: dict: tp(True positive), p(Positive), a(Answer)
    no_answer_flag = False
    no_pred_flag = False
    if len(answer) == 0:
        no_answer_flag = True
        answer.append([0,0])
    if len(prediction) == 0:
        no_pred_flag = True
        prediction.append([0,0])
    answer = np.array(answer)
    prediction = np.array(prediction)
    profit = np.zeros((len(answer), len(prediction)))
    for i in range(len(answer)):
        for j in range(len(prediction)):
            profit[i][j] = calculate_iou(answer[i], prediction[j])
    r, c = optimize.linear_sum_assignment(profit, maximize=True)
    tp = np.sum(np.where(profit[r, c] >= iou_threshold, 1, 0))
    a = answer.shape[0]
    p = prediction.shape[0]
    if no_answer_flag:
        a = 0
    if no_pred_flag:
        p = 0
    return {'tp':tp, 'p':p, 'a':a}

def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def get_aux_loss(logits):
    '''
    logits: [B,L,n_state]
    return penalty loss
    -- loss which penalizes context change --
    '''
    import torch
    max_idx = torch.argmax(logits, dim=-1)
    #[B,L]
    dummy = torch.full((logits.size(0),1), fill_value=-1).to(logits.device)
    front = torch.cat([dummy, max_idx], dim=1)
    back = torch.cat([max_idx, dummy], dim=1)
    boolidx = (front != back)[:,:-1]
    boolidx[:, 0] = False
    front = front[:, :-1]
    target = front[boolidx]
    logits = logits[boolidx]
    max_idx = torch.argmax(logits, dim=-1)
    is_target_not4 = target != 4
    is_logits_not4 =  max_idx != 4
    exclusive_mask = torch.logical_and(is_target_not4, is_logits_not4)
    target = target[exclusive_mask]
    logits = logits[exclusive_mask]
    loss = torch.nn.functional.cross_entropy(logits, target)
    return loss

def g(base_value, start_warmup_value, final_value, warmup_iters, final_iter):
    import math
    def f(i):
        if i <= warmup_iters:
            value = (base_value - start_warmup_value)*(i/warmup_iters) + start_warmup_value
        elif i < final_iter:
            value = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * (i-warmup_iters) / (final_iter - warmup_iters)))
        else:
            value = final_value
        return value
    return f

def get_blocked_videos(api='http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/challenge19/api.py'):
    api_url = '{}?action=get_blocked'.format(api)
    response = urllib.request.urlopen(api_url)
    return json.loads(response.read())

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
  
    return tIoU

def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap
