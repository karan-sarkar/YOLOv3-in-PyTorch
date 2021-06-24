from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

__all__ = ['COCOEvaluator']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str)
    parser.add_argument('--dt', type=str)
    _options = parser.parse_args()
    return _options

class COCOEvaluator(object):

    def __init__(self, anno_gt_file, anno_dt_file):
        self.coco_gt = COCO(anno_gt_file)
        self.coco_dt = COCO(anno_dt_file)
        self._hack_coco_dt()

    def _hack_coco_dt(self):
        for ann in self.coco_dt.dataset['annotations']:
            ann['score'] = 1.0

    def evaluate(self, iou_type='bbox'):
        coco_eval = COCOeval(self.coco_gt, self.coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval
        
args = parse_args()

eval = COCOEvaluator(args.gt, args.dt)
print(eval.evaluate())