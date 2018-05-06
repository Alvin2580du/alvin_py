from scipy.spatial import distance


def cosine(emb_1, emb_2):
    return distance.cosine(emb_1, emb_2)


if __name__ == '__main__':
    res = cosine([1, 2, 3], [3, 4, 5])
    print(res)
