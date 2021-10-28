import pickle, math
import numpy as np
from pandas.core import frame
from . import DetectionFilter

class KITTIFilter(DetectionFilter):
    DIFFICULTY = ['easy', 'moderate', 'hard']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]

    def __init__(self, class_name='car', class_label=1, difficulty='moderate', infos=None, info_path=None, *args, **kwargs):
        """ Detection filter for KITTI dataset.
        Results will be filtered by class and difficulty levels
        """
        if type(difficulty) is int and 0 <= difficulty < 3:
            pass
        elif type(difficulty) is str:
            difficulty = self.DIFFICULTY.index(difficulty)
        else:
            raise ValueError(f'invalid KITTI difficulty level: {difficulty}')

        class_name = class_name.lower()
        name = f'kitti_{class_name}_{self.DIFFICULTY[difficulty]}'
        super().__init__(name, *args, **kwargs)
        
        self.class_name = class_name
        self.class_label = class_label
        self.difficulty = difficulty

        if (infos is None and info_path is None) or (infos is not None and info_path is not None):
            raise ValueError('must provide either infos or info_path')
        elif infos is not None:        
            self.infos = infos
        else:
            self.infos = self.build_kitti_infos(info_path)
    
    @staticmethod
    def build_kitti_infos(info_path):
        with open(info_path, 'rb') as f:
            raw_infos = pickle.load(f)
        
        # Build metadata database for annotations
        # This will be used to calculate difficulty levels later
        kitti_infos = {}
        for info in raw_infos:
            frame_id = info['point_cloud']['lidar_idx']
            kitti_infos[frame_id] = []
            annos = info['annos']
            names = annos['name']
            boxes = annos['gt_boxes_lidar']
            occluded = annos['occluded']
            truncated = annos['truncated']
            bboxes = annos['bbox']
            heights = bboxes[:,3] - bboxes[:,1]
            num_points = annos['num_points_in_gt']
            bbox_occlusion = annos['occlusion_level'] if 'occlusion_level' in annos else np.full(len(names), np.nan)
            for name, box, o, t, h, n, bo in zip(names, boxes, occluded, truncated, heights, num_points, bbox_occlusion):
                kitti_infos[frame_id].append({
                    'name': name,
                    'loc': box[:2],
                    'occluded': o,
                    'truncated': t,
                    'height': h,
                    'num_points_in_gt': n,
                    'bbox_occlusion': bo
                })
        return kitti_infos

    @classmethod
    def check_difficulty(cls, difficulty, occluded, truncated, height):
        return (occluded <= cls.MAX_OCCLUSION[difficulty] and
                truncated <= cls.MAX_TRUNCATION[difficulty] and
                height > cls.MIN_HEIGHT[difficulty])

    def get_ignored_gt(self, gt):
        # Ignored boxes will not be counted towards FP if detected or FN if not detected
        # frame_id = gt['frame_id']
        # names = gt['gt_names']
        frame_id = gt['image']['image_idx']
        names = gt['annos']['name']

        ignored = np.zeros(len(names), dtype=bool)
        for idx, name in enumerate(names):
            name = name.lower()
            if name == 'dontcare':
                ignored[idx] = True
                continue
            annos = self.infos[frame_id][idx]
            # Ignore similar classes
            if self.class_name == 'pedestrian' and name == 'person_sitting':
                ignored[idx] = True
            elif self.class_name == 'car' and name == 'van':
                ignored[idx] = True
            # If label is a different class, don't ignore
            elif self.class_name != name:
                ignored[idx] = False
            # Ignore same class but different difficulty
            elif not self.check_difficulty(self.difficulty, annos['occluded'], annos['truncated'], annos['height']):
                ignored[idx] = True
            else:
                ignored[idx] = False
        return ignored
    
    def get_discarded_gt(self, gt):
        # Discarded boxes will not be counted towards FN if not detected,
        # but will be counted as FP if detected
        # labels = gt['gt_labels']
        classes = dict(
            Car=1, Pedestrian=2, Cyclist=3,
            Truck=-1, Misc=-1, Van=-1, Tram=-1, Person_sitting=-1,
            DontCare=-1
        )
        labels = []
        for name in gt['annos']['name']:
            if name == 'DontCare':
                continue
            labels.append(classes[name])
        # print('gt', gt)
        # print("gt['annos']['name']", gt['annos']['name'])
        # print('self.class_label', self.class_label)
        # print('labels', labels)
        # exit()
        ret = []
        for label in labels:
            ret.append(label != self.class_label)
        return ret
        # return (labels != self.class_label)

    def get_discarded_pred(self, pred):
        # Discarded prediction will not be regarded as positive
        # i.e. will not be counted as either TP or FP
        class_names = ['Car', 'Pedestrian', 'Cyclist']
        labels = np.array([class_names.index(n)+1 for n in pred['name']])
        return labels != self.class_label


def build_kitti_filters(info_path, class_names=['Car', 'Pedestrian', 'Cyclist'], class_labels=[1, 2, 3], difficulty_levels=['easy', 'moderate', 'hard'], *args, **kwargs):
    kitti_infos = KITTIFilter.build_kitti_infos(info_path)
    filters = []
    for name, label in zip(class_names, class_labels):
        for difficulty in difficulty_levels:
            filters.append( KITTIFilter(name, label, difficulty, kitti_infos, *args, **kwargs) )
    return filters

class CADCFilter(DetectionFilter):
    DIFFICULTY = ['easy', 'moderate', 'hard']
    MIN_HEIGHT = [-1, -1, -1] # Do not limit by height since we want 360 deg test
    MAX_OCCLUSION = [0, 1, 2] # lidar occlusion is used as occlusion level
    MAX_TRUNCATION = [1.0, 1.0, 1.0] # Do not limit by height since we want 360 deg test

    def __init__(self, class_name='car', class_label=1, difficulty='moderate', infos=None, info_path=None, *args, **kwargs):
        """ Detection filter for CADC dataset.
        Results will be filtered by class and difficulty levels
        """
        if type(difficulty) is int and 0 <= difficulty < 3:
            pass
        elif type(difficulty) is str:
            difficulty = self.DIFFICULTY.index(difficulty)
        else:
            raise ValueError(f'invalid CADC difficulty level: {difficulty}')

        class_name = class_name.lower()
        name = f'cadc_{class_name}_{self.DIFFICULTY[difficulty]}'
        super().__init__(name, *args, **kwargs)
        
        self.class_name = class_name
        self.class_label = class_label
        self.difficulty = difficulty

        if (infos is None and info_path is None) or (infos is not None and info_path is not None):
            raise ValueError('must provide either infos or info_path')
        elif infos is not None:        
            self.infos = infos
        else:
            self.infos = self.build_cadc_infos(info_path)
    
    @staticmethod
    def build_cadc_infos(info_path):
        with open(info_path, 'rb') as f:
            raw_infos = pickle.load(f)
        
        # Build metadata database for annotations
        # This will be used to calculate difficulty levels later
        cadc_infos = {}
        for info in raw_infos:
            date, seq, frame = info['point_cloud']['lidar_idx']
            frame_id = date + '_' + seq + '_' + frame
            cadc_infos[frame_id] = []
            annos = info['annos']
            names = annos['name']
            boxes = annos['gt_boxes_lidar']
            occluded = annos['occluded']
            truncated = annos['truncated']
            bboxes = annos['bbox']
            heights = bboxes[:,3] - bboxes[:,1] # This is invalid due to be calculated for front camera
            num_points = annos['num_points_in_gt']
            bbox_occlusion = annos['occlusion_level'] if 'occlusion_level' in annos else np.full(len(names), np.nan)
            for name, box, o, t, h, n, bo in zip(names, boxes, occluded, truncated, heights, num_points, bbox_occlusion):
                cadc_infos[frame_id].append({
                    'name': name,
                    'loc': box[:2],
                    'occluded': o,
                    'truncated': t,
                    'height': h,
                    'num_points_in_gt': n,
                    'bbox_occlusion': bo
                })
        return cadc_infos

    @classmethod
    def check_difficulty(cls, difficulty, occluded, truncated, height):
        return (occluded <= cls.MAX_OCCLUSION[difficulty] and
                truncated <= cls.MAX_TRUNCATION[difficulty] and
                height > cls.MIN_HEIGHT[difficulty])

    def get_ignored_gt(self, gt):
        # Ignored boxes will not be counted towards FP if detected or FN if not detected
        date, seq, frame = gt['image']['image_idx']
        frame_id = date + '_' + seq + '_' + frame
        names = gt['annos']['name']

        ignored = np.zeros(len(names), dtype=bool)
        for idx, name in enumerate(names):
            name = name.lower()
            if name == 'dontcare':
                ignored[idx] = True
                continue
            annos = self.infos[frame_id][idx]
            # Ignore similar classes
            if self.class_name == 'pedestrian' and name == 'person_sitting':
                ignored[idx] = True
            elif self.class_name == 'car' and name == 'van':
                ignored[idx] = True
            # If label is a different class, don't ignore
            elif self.class_name != name:
                ignored[idx] = False
            # Ignore same class but different difficulty
            elif not self.check_difficulty(self.difficulty, annos['occluded'], annos['truncated'], annos['height']):
                ignored[idx] = True
            else:
                ignored[idx] = False
        return ignored

    def get_discarded_gt(self, gt):
        # Discarded boxes will not be counted towards FN if not detected,
        # but will be counted as FP if detected
        classes = dict(
            Car=1, Pedestrian=2, Pickup_Truck=3,
            Cyclist=-1, Misc=-1, Van=-1, Tram=-1, Person_sitting=-1,
            DontCare=-1
        )
        labels = []
        for name in gt['annos']['name']:
            if name == 'DontCare':
                continue
            labels.append(classes[name])
        ret = []
        for label in labels:
            ret.append(label != self.class_label)
        return ret

    def get_discarded_pred(self, pred):
        # Discarded prediction will not be regarded as positive
        # i.e. will not be counted as either TP or FP
        class_names = ['Car', 'Pedestrian', 'Pickup_Truck']
        labels = np.array([class_names.index(n)+1 for n in pred['name']])
        return labels != self.class_label


def build_cadc_filters(info_path, class_names=['Car', 'Pedestrian', 'Pickup_Truck'], class_labels=[1, 2, 3], difficulty_levels=['easy', 'moderate', 'hard'], *args, **kwargs):
    cadc_infos = CADCFilter.build_cadc_infos(info_path)
    filters = []
    for name, label in zip(class_names, class_labels):
        for difficulty in difficulty_levels:
            filters.append( CADCFilter(name, label, difficulty, cadc_infos, *args, **kwargs) )
    return filters

class NuScenesFilter(DetectionFilter):
    DIFFICULTY = ['default']
    CLS_DETECTION_RANGE = {
        'car':50, 'truck':50, 'construction_vehicle':50, 'bus':50, 'trailer':50, \
        'barrier':30, 'motorcycle':40, 'bicycle':40, 'pedestrian':40, 'traffic_cone':30
    }

    def __init__(self, class_name='car', class_label=1, difficulty='default', infos=None, info_path=None, *args, **kwargs):
        """ Detection filter for NuScenes dataset.
        Results will be filtered by class and difficulty levels
        """
        if type(difficulty) is int and 0 <= difficulty < 3:
            pass
        elif type(difficulty) is str:
            difficulty = self.DIFFICULTY.index(difficulty)
        else:
            raise ValueError(f'invalid NuScenes difficulty level: {difficulty}')

        class_name = class_name.lower()
        name = f'nuscenes_{class_name}_{self.DIFFICULTY[difficulty]}'
        super().__init__(name, *args, **kwargs)
        
        self.class_name = class_name
        self.class_label = class_label
        self.difficulty = difficulty

        if (infos is None and info_path is None) or (infos is not None and info_path is not None):
            raise ValueError('must provide either infos or info_path')
        elif infos is not None:        
            self.infos = infos
        else:
            self.infos = self.build_nuscenes_infos(info_path)
    
    @staticmethod
    def build_nuscenes_infos(info_path):
        with open(info_path, 'rb') as f:
            raw_infos = pickle.load(f)
        
        # Build metadata database for annotations
        # This will be used to calculate difficulty levels later
        nuscenes_infos = {}
        for info in raw_infos:
            frame_id = info['token']
            nuscenes_infos[frame_id] = []
            annos = info # Only to keep consistent with other datasets
            names = annos['gt_names']
            boxes = annos['gt_boxes']
            num_points = annos['num_lidar_pts']
            num_radar_points = annos['num_radar_pts']
            for name, box, n, nr in zip(names, boxes, num_points, num_radar_points):
                nuscenes_infos[frame_id].append({
                    'name': name,
                    'loc': box[:2],
                    'num_points_in_gt': n,
                    'num_radar_points_in_gt': nr
                })
        return nuscenes_infos

    # Using filtering logic from NuScenes
    # https://github.com/nutonomy/nuscenes-devkit/blob/05b601f363a96deadb60123103b9011110a1ca82/python-sdk/nuscenes/eval/common/loaders.py#L207
    @classmethod
    def check_ignored(self, cls, box, lidar_pts, radar_pts):
        # ignored classes are already handled

        # If the object is outside of class detection range
        if math.sqrt(box[0]**2 + box[1]**2) > self.CLS_DETECTION_RANGE[cls]:
            return 1

        # If the object has no lidar or radar points
        if lidar_pts == 0 and radar_pts == 0:
            return 1
        
        # The GT is valid
        return 0

    def get_ignored_gt(self, gt):
        # Ignored boxes will not be counted towards FP if detected or FN if not detected
        frame_id = gt['token']
        names = gt['gt_names']
        boxes = gt['gt_boxes']
        num_points = gt['num_lidar_pts']
        num_radar_points = gt['num_radar_pts']

        classes = dict(
            car=1, truck=2, construction_vehicle=3, bus=4, trailer=5, \
            barrier=6, motorcycle=7, bicycle=8, pedestrian=9, traffic_cone=10
        )

        # No boxes are ignored in the NuScenes dataset
        ignored = np.zeros(len(names), dtype=bool)
        for idx, name in enumerate(names):
            name = name.lower()
            # Ignored classes like bikes in a bike rack are discarded
            if name == 'ignore':
                ignored[idx] = True
                continue

            if self.check_ignored(name, boxes[idx], num_points[idx], num_radar_points[idx]):
                ignored[idx] = True
            elif classes[name] != self.class_label:
                ignored[idx] = True
            else:
                ignored[idx] = False
        return ignored

    def get_discarded_gt(self, gt):
        # Discarded boxes will not be counted towards FN if not detected,
        # but will be counted as FP if detected
        names = gt['gt_names']
        classes = dict(
            car=1, truck=2, construction_vehicle=3, bus=4, trailer=5, \
            barrier=6, motorcycle=7, bicycle=8, pedestrian=9, traffic_cone=10, \
            ignore = -1
        )
        labels = []
        for name in names:
            labels.append(classes[name])
        ret = []
        for label in labels:
            if label == -1: # ignore is always discarded
                ret.append(True)
            else:
                ret.append(label != self.class_label)
        return ret


    def get_discarded_pred(self, pred):
        # Discarded prediction will not be regarded as positive
        # i.e. will not be counted as either TP or FP
        class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', \
                        'bicycle', 'pedestrian', 'traffic_cone']
        labels = np.array([class_names.index(n)+1 for n in pred['name']])
        return labels != self.class_label


def build_nuscenes_filters(info_path,\
    class_names=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', \
        'bicycle', 'pedestrian', 'traffic_cone'],\
        class_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], difficulty_levels=['default'], *args, **kwargs):
    nuscenes_infos = NuScenesFilter.build_nuscenes_infos(info_path)
    filters = []
    for name, label in zip(class_names, class_labels):
        for difficulty in difficulty_levels:
            filters.append( NuScenesFilter(name, label, difficulty, nuscenes_infos, *args, **kwargs) )
    return filters
