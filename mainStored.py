import numpy as np
import cv2
import video
from common import anorm2, draw_str
import time

lk_params = dict( winSize  = (51,51),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.1,
                       minDistance = 20,
                       blockSize = 10)


class LKProcessing:
    def __init__(self, video_src,usePiCamera=False):
        self.track_len = 10
        self.detect_interval = 3
        self.tracks = []
        self.cam = video.create_capture(video_src,usePiCamera=usePiCamera)
        self.frame_idx = 0

    def run(self):
        counter = 0
        startTime = time.time()
        oldTime = startTime

        while True:
            ret,frame = self.cam.read()
            if not ret:
                break
            frame = cv2.resize(frame,(640,480))
            vis = frame.copy()
            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            if len(self.tracks) > 0:
                img0, img1 = cv2.GaussianBlur(self.prev_gray,(7,7),7), cv2.GaussianBlur(frame_gray,(7,7),7)

                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1

                new_tracks = []

                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks

                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            cv2.imwrite('./Results/Movie1/{}.png'.format(counter),vis)
            newTime = time.time()-startTime
            print('Saving image {} on time: {}, from previous: {}, fps = {}'.format(counter,newTime,newTime-oldTime,1./(newTime-oldTime)))
            oldTime = newTime
            counter += 1
            if (cv2.waitKey(1)&0xFF) == ord('q'):
                break


if __name__ == '__main__':
    LK = LKProcessing('Artificial.mov',usePiCamera = False)
    LK.run()
    cv2.destroyAllWindows()
