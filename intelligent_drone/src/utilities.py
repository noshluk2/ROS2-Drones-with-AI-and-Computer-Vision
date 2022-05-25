import cv2

def get_centroid(cnt):
    M = cv2.moments(cnt)
    if (M['m00']==0):
        pt_a = cnt[0][0]
        return (pt_a[0],pt_a[1])
    else:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx,cy)