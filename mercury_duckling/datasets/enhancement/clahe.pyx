import numpy as np

def np_interpolate(subBin, LU, RU, LB, RB, subX, subY):
    """
    Image interpolation function, need to be applied after CLAHE
    """
    subImage = np.zeros(subBin.shape)
    num = subX * subY
    for i in range(subX):
        inverseI = subX - i
        for j in range(subY):
            inverseJ = subY - j
            val = subBin[i, j].astype(int)
            subImage[i, j] = np.floor(
                (inverseI * (inverseJ * LU[val] + j * RU[val]) + i * (inverseJ * LB[val] + j * RB[val])) / float(num))
    return subImage


def np_clahe(img, clipLimit, nrBins=128, nrX=0, nrY=0):
    """
    CLAHE algorithm implementation

    :param img: Input image
    :param clipLimit: Normalized clipLimit. Higher value gives more contrast
    :param nrBins: Number of gray level bins for histogram("dynamic range")
    :param nrX: Number of contextual regions in X direction
    :param nrY: Number of contextual regions in Y direction
    """
    h, w = img.shape
    if clipLimit == 1:
        return
    nrBins = max(nrBins, 128)
    if nrX == 0:
        # Taking dimensions of each contextial region to be a square of 32X32
        xsz = 32
        ysz = 32
        nrX = np.ceil(h / xsz)  # 240
        # Excess number of pixels to get an integer value of nrX and nrY
        excX = int(xsz * (nrX - h / xsz))
        nrY = np.ceil(w / ysz)  # 320
        excY = int(ysz * (nrY - w / ysz))
        # Pad that number of pixels to the image
        if excX != 0:
            img = np.append(img, np.zeros((excX, img.shape[1])).astype(int), axis=0)
        if excY != 0:
            img = np.append(img, np.zeros((img.shape[0], excY)).astype(int), axis=1)
    else:
        xsz = np.round(h / nrX)
        ysz = np.round(w / nrY)
        excX = int(xsz * (nrX - h / xsz))
        excY = int(ysz * (nrY - w / ysz))
        if excX != 0:
            img = np.append(img, np.zeros((excX, img.shape[1])).astype(int), axis=0)
        if excY != 0:
            img = np.append(img, np.zeros((img.shape[0], excY)).astype(int), axis=1)

    nrPixels = xsz * ysz
    claheimg = np.zeros(img.shape)

    if clipLimit > 0:
        clipLimit = max(1, clipLimit * xsz * ysz / nrBins)
    else:
        clipLimit = 50

    # Making LUT
    minVal = 0  # np.min(img)
    maxVal = 255  # np.max(img)

    binSz = np.floor(1 + (maxVal - minVal) / float(nrBins))
    LUT = np.floor((np.arange(minVal, maxVal + 1) - minVal) / float(binSz))

    # Creating bins from LUT with image
    bins = LUT[img]

    # Making Histogram
    nrX = int(nrX)
    nrY = int(nrY)
    xsz = int(xsz)
    ysz = int(ysz)
    hist = np.zeros((nrX, nrY, nrBins))
    for i in range(nrX):
        for j in range(nrY):
            bin_ = bins[i * xsz:(i + 1) * xsz, j * ysz:(j + 1) * ysz].astype(int)
            for i1 in range(xsz):
                for j1 in range(ysz):
                    hist[i, j, bin_[i1, j1]] += 1

    # Clipping Histogram
    if clipLimit > 0:
        for i in range(nrX):
            for j in range(nrY):
                nrExcess = 0
                for nr in range(nrBins):
                    excess = hist[i, j, nr] - clipLimit
                    if excess > 0:
                        nrExcess += excess

                binIncr = nrExcess / nrBins
                upper = clipLimit - binIncr
                for nr in range(nrBins):
                    if hist[i, j, nr] > clipLimit:
                        hist[i, j, nr] = clipLimit
                    else:
                        if hist[i, j, nr] > upper:
                            nrExcess += upper - hist[i, j, nr]
                            hist[i, j, nr] = clipLimit
                        else:
                            nrExcess -= binIncr
                            hist[i, j, nr] += binIncr

                if nrExcess > 0:
                    stepSz = max(1, np.floor(1 + nrExcess / nrBins))
                    for nr in range(nrBins):
                        nrExcess -= stepSz
                        hist[i, j, nr] += stepSz
                        if nrExcess < 1:
                            break

    # Mapping Histogram
    map_ = np.zeros((nrX, nrY, nrBins))
    # print(map_.shape)
    scale = (maxVal - minVal) / float(nrPixels)
    for i in range(nrX):
        for j in range(nrY):
            sum_ = 0
            for nr in range(nrBins):
                sum_ += hist[i, j, nr]
                map_[i, j, nr] = np.floor(min(minVal + sum_ * scale, maxVal))

    # Interpolation
    xI = 0
    for i in range(nrX + 1):
        if i == 0:
            subX = int(xsz / 2)
            xU = 0
            xB = 0
        elif i == nrX:
            subX = int(xsz / 2)
            xU = nrX - 1
            xB = nrX - 1
        else:
            subX = xsz
            xU = i - 1
            xB = i

        yI = 0
        for j in range(nrY + 1):
            if j == 0:
                subY = int(ysz / 2)
                yL = 0
                yR = 0
            elif j == nrY:
                subY = int(ysz / 2)
                yL = nrY - 1
                yR = nrY - 1
            else:
                subY = ysz
                yL = j - 1
                yR = j
            UL = map_[xU, yL, :]
            UR = map_[xU, yR, :]
            BL = map_[xB, yL, :]
            BR = map_[xB, yR, :]

            subBin = bins[xI:xI + subX, yI:yI + subY]

            subImage = np_interpolate(subBin, UL, UR, BL, BR, subX, subY)
            claheimg[xI:xI + subX, yI:yI + subY] = subImage
            yI += subY
        xI += subX

    if excX == 0 and excY != 0:
        return claheimg[:, :-excY]
    elif excX != 0 and excY == 0:
        return claheimg[:-excX, :]
    elif excX != 0 and excY != 0:
        return claheimg[:-excX, :-excY]
    else:
        return claheimg