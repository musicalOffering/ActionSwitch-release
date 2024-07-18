import argparse
import numpy as np
import json
import pandas as pd
from joblib import Parallel, delayed
from utils import compute_average_precision_detection, get_blocked_videos

#python compute.py thumos14_v2.json canonical.json

parser = argparse.ArgumentParser(description='This script allows you to evaluate the ActivityNet '
                                        'detection task which is intended to evaluate the ability '
                                        'of  algorithms to temporally localize activities in '
                                        'untrimmed video sequences.')
parser.add_argument('ground_truth_filename',
            help='Full path to json file containing the ground truth.')
parser.add_argument('prediction_filename',
            help='Full path to json file containing the predictions.')
parser.add_argument('--dataset', default='Thumos', 
            help=('Dataset that wants to get map values'))
parser.add_argument('--subset', default='test', 
            help='String indicating subset to evaluate: ') 
parser.add_argument('--tiou_thresholds', type=float, default=0.7,
            help='Temporal intersection over union threshold.')
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--check_status', type=bool, default=True)

def main():
    args = parser.parse_args()
    ground_truth_filename = args.ground_truth_filename
    prediction_filename = args.prediction_filename
    dataset = args.dataset
    subset = args.subset
    tiou_thresholds = args.tiou_thresholds
    verbose = args.verbose
    check_status = args.check_status

    if dataset == 'Thumos':
        detection = THUMOSdetection(ground_truth_filename, prediction_filename, 
                                    subset=subset, tiou_thresholds=tiou_thresholds,
                                    verbose=verbose, check_status=False)

    detection.evaluate()

class THUMOSdetection(object):
    GROUND_TRUTH_FIELDS = ['database']
    PREDICTION_FIELDS = ['results']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 subset="test", verbose=False, 
                 check_status=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = [tiou_thresholds]
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()
        self.ground_truth, self.activity_index = self._import_ground_truth(ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print ('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print ('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print ('\tNumber of predictions: {}'.format(nr_pred))
            print ('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.
        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.
        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r', encoding='utf-8') as fobj:
            data = json.load(fobj)

        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for vidname, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            if vidname in ['video_test_0000270', 'video_test_0001496']:
                continue

            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(vidname)
                #t_start_lst.append(float(ann['segment_frame'][0] ))  # bbdb / 6 | THUMOS / 5 | HACS / 2
                #t_end_lst.append(float(ann['segment_frame'][1] ))
                t_start_lst.append(ann['segment'][0])
                t_end_lst.append(ann['segment'][1])
                label_lst.append(activity_index[ann['label']])
        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        print(activity_index)
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.
        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.
        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)

        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data['results'].items():

            if videoid in ['video_test_0000270', 'video_test_0001496']:
                continue

            if videoid in self.blocked_videos:
                continue
            for result in v:
                try:
                    label = self.activity_index[result['label']]
                except KeyError:
                    label = 0
                
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label. 
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            print ('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')
        n_jobs = min(30, len(self.activity_index))
        results = Parallel(n_jobs=n_jobs)(
                    delayed(compute_average_precision_detection)(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                        prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                        tiou_thresholds=self.tiou_thresholds,
                    ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()
        formated_ap = [np.around((elem), 3) * 100  for elem in self.ap]

        if self.verbose:
            print ('[RESULTS] Performance on THUMOS detection task.')
            for key in self.activity_index:
                print('class:   {} - ap:    {}'.format(key, self.ap[0][self.activity_index[key]] * 100 ))
            print ('\tAverage-mAP: {}'.format(self.average_mAP))

if __name__ == '__main__':
    main()
