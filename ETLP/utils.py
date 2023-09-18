from collections import Counter


def prediction_mostcommon(outputs):
    maxs = outputs.argmax(axis=-1)
    res = []
    for i in range(maxs.shape[0]):
        res.append(Counter(maxs[i]).most_common(1)[0][0])
    return res
