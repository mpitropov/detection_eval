# Detection Eval

## I. Requirements
* numpy
* scipy [optional, for scipy distance metric]
* torch, mmcv [optional, for IoU metric]
* tqdm [optional, for displaying evaluation progress]
```
pip install numpy scipy tqdm torch mmcv-full
```

## II. Basic Usage

To perform basic evaluation, simply instantiate a `DetectionEval` instance and call it
```python
detection_evaluator = DetectionEval( ... )
results = detection_evaluator( gts, preds )
```
or 
```python
results = DetectionEval.evaluate( gts, preds, ... )
```

The arguments are:
  * `gts`: a list of ground truth annotations. If not in `(labels, boxes)` format, then `gt_processor` must be specified.
  * `preds`: a list of predictions. If not in `(labels, scores, boxes)` format, then `pred_processor` must be specified.
  * `thresholds`: a list or dict that maps a label to a matching threshold.
  * `criterion` [optional]: matching criterion, can be one of \
  `['iou', iou_2d', iou_bev', 'iou_3d', 'iof', 'iof_2d', 'iof_bev', 'iof_3d', 'euclidean', 'euclidean_3d']`\
  Defaults to `iou`.
  * `epsilon` [optional]: minimum threshold for matching. Defaults to 0.1.
  * `filters` [optional]: a list of callables that specify boxes that should be discarded or ignored. Defaults to None.
  * `n_positions` [optional]: number of recall positions to use in the AP evaluation. Defaults to 100.
  * `metrics` [optional]: a list of metrics that will be in the returned results. Defaults to ['ap'].
  * `gt_processor` [optional]: a callable that transforms each item in `gts` into `(labels, boxes)` format.
  * `pred_processor` [optional]: a callable that transforms each item in `preds` into `(labels, scores, boxes)` format.
  * `verbose` [optional]: whether to display evaluation progress. Defaults to False.

## III. Evaluation Pipeline

For each specified filter, the detection evaluation consists of three steps
  * For each sample/frame, box matching is performed to compute TP/FP/FN breakdowns
  * The TP/FP/FN information from all samples are gathered and combined
  * Statsitics (e.g. AP) are computed

### 1. Evaluating One Sample
To manually evaluate one sample and get the detailed TP/FP/FN breakdown, 
you can use
```python
gt_list, pred_list = DetectionEval.evaluate_one_sample( gt, pred, ... )
```

The parameters are similar to `DetectionEval.evaluate(...)`:
  * `gt`: ground truth annotation. If not in `(labels, boxes)` format, then `gt_processor` must be specified.
  * `pred`: predictions. If not in `(labels, scores, boxes)` format, then `pred_processor` must be specified.
  * `thresholds`: a list or dict that maps a label to a matching threshold.
  * `criterion` [optional]: matching criterion, can be one of \
  `['iou', iou_2d', iou_bev', 'iou_3d', 'iof', 'iof_2d', 'iof_bev', 'iof_3d', 'euclidean', 'euclidean_3d']`\
  Defaults to `iou`.
  * `epsilon` [optional]: minimum threshold for matching. Defaults to 0.1.
  * `filta` [optional]: a callables that specify boxes that should be discarded or ignored. Defaults to None.
  * `ctable` [optional]: pairwise distance table between the gt and pred boxes. Defaults to None.
  * `gt_processor` [optional]: a callable that transforms each item in `gts` into `(labels, boxes)` format.
  * `pred_processor` [optional]: a callable that transforms each item in `preds` into `(labels, scores, boxes)` format.

The return value is a tuple of two `BoxList` objects: `gt_list` and `pred_list`.

### 2. BoxList - TP/FP/FN Data Structure
The return value from the evaluation are two `BoxList` type objects that store the necessary TP/FP/FN information. 

`BoxList` objects has the following attributes:
  * `ignored`: a binary mask indicating if a box is ignored during the evaluation
  * `bg`: a binary mask indicating if a box does not match with any GT/pred box (e.g. IoU < epsilon)
  * `localized`: a binary mask indicating if a box is correctly localized (e.g. best_iou > threshold)
  * `loc_scores`: an array that records the localization scores (e.g. IoU, distance, etc.)
  * `classified`: a binary mask indicating if a box is correctly classified, ignoring localization
  * `gt_labels`: (matched) ground truth labels for the boxes
  * `pred_labels`: (matched) predicted labels for the boxes
  * `pred_scores`: (matched) predicted scores for the boxes
  * `matched_idx`: an array that stores indices of the matching GT/pred boxes (for TPs)
  * `data`: extra payload that can be attached by the user during evaluation

For each sample, the evaluation will always generate two `BoxList` objects, one for GT boxes and one for predicted boxes. To correctly perform aggregation and combine `matched_idx` attribute when combining `BoxList` objects, the two list can (and should) be paired. This is done automatically, but to manually pair two `BoxList`, do

```python
gt_list.pair_with(pred_list)
```

### 3. Evaluate Multiple Samples
To evaluate multiple samples, you can call `DetectionEval.evaluate_one_sample` for each sample and aggregate the results.
Alternatively, `DetectionEval` also provides a function that simplifies this process.
```python
gt_list, pred_list = DetectionEval.evaluate_all_samples( ... )
```
The parameters are the same as `DetectionEval.evaluate_one_sample`, with the following exceptions:
  * `gts`: instead of a single `gt` sample, `evaluate_all_samples` takes a list of `gt` samples.
  * `preds`: similarly, it takes a list of `preds` samples.
  * `pbar` [optional]: a `tqdm` progress bar object to show the evaluation progress. 
  `pbar.update()` will be called after each sample evaluation.
  * `callback` [optional]: a callable that will be called after evaluating each sample.
  It can be used to attach additional `data` to the result `BoxList` objects.

    The `callback` will be provided with the following positional arguments:

    `callback(sample_idx, gt, pred, gt_list, pred_list)` 

```python
# Progress bar can be tqdm, tqdm.notebook or other compatible objects
from tqdm import tqdm
pbar = tqdm(total=len(gts))

# A callback that attaches extra data to the `BoxList`
def attach_data( sample_idx, gt, pred, gt_list, pred_list ):
    gt_list.data = gt['data']
    pred_list.data = pred['data']

gt_list, pred_list = DetectionEval.evaluate_all_samples(
    preds, gts, ..., pbar=pbar, callback=attach_data
)