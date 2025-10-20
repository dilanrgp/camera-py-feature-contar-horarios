import numpy as np
from data_association import associate_detections_to_trackers
from kalman_tracker import KalmanBoxTracker


class Sort:

    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, img_size, predict_num):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE:as in practical realtime MOT, the detector doesn't run on every single frame
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t][0].predict()  # kalman predict ,very fast ,<1ms
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if dets != []:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk[0].update(dets[d, :][0])
                    trk[1] = dets[d, 4][0]

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :])
                self.trackers.append([trk, dets[i, 4]])

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            #if dets == []:
            #    trk[0].update([])
            d = trk[0].get_state()
            if (trk[0].time_since_update < 1) and (trk[0].hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk[0].id + 1], [trk[1]])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet            
            if trk[0].time_since_update >= self.max_age or trk[0].predict_num >= 100 or d[2] < 0 or d[3] < 0 or d[0] > img_size[1] or d[1] > img_size[0]:
                print('==========================')
                print(trk[0].time_since_update, self.max_age)
                print(trk[0].predict_num, predict_num)
                
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
