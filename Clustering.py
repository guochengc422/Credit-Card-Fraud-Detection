import numpy as np


def Clustering(acts):
    centerdict = {}
    tmp = acts[0]
    D = 1
    epsino = 1.5
    centerdict[D] = [tmp]
    acts = acts[1:]
    for a in acts:
        dist = 999
        for item in centerdict.items():
            key = item[0]
            points = item[1]
            center = sum(points)/1.0/len(points)
            if abs(center - a) < dist:
                dist = abs(center - a)
                tmpkey = key
        if dist <= epsino:
            centerdict[tmpkey].append(a)
        else:
            D += 1
            centerdict[D] = [a]

    centerlist = []

    # print centerdict
    for item in centerdict.items():
            points = item[1]
            center = float(sum(points)/1.0/len(points))
            # print center
            # print type(center)
            centerlist.append(center)

    print centerlist
    return centerlist


x = np.random.randn(1000, 1)
Clustering(x)
