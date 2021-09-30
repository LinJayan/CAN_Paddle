
import numpy



def prepare_data(inputs, target, maxlen = None, return_neg = False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in inputs]
    seqs_mid = [inp[3] for inp in inputs]
    seqs_cat = [inp[4] for inp in inputs]
    noclk_seqs_mid = [inp[5] for inp in inputs]
    noclk_seqs_cat = [inp[6] for inp in inputs]
    seqs_item_carte = [inp[7][0] for inp in inputs]
    seqs_cate_carte = [inp[7][1] for inp in inputs]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        new_seqs_item_carte = []
        new_seqs_cate_carte = []
        for l_x, inp in zip(lengths_x, inputs):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_seqs_item_carte.append(inp[7][0][l_x - maxlen:])
                new_seqs_cate_carte.append(inp[7][1][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_seqs_item_carte.append(inp[7][0])
                new_seqs_cate_carte.append(inp[7][1])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat
        seqs_item_carte = new_seqs_item_carte
        seqs_cate_carte = new_seqs_cate_carte

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    item_carte = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cate_carte = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, [s_x, s_y, no_sx, no_sy, i_c, c_c] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat, seqs_item_carte, seqs_cate_carte)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy
        item_carte[idx, :lengths_x[idx]] = i_c
        cate_carte[idx, :lengths_x[idx]] = c_c

    uids = numpy.array([inp[0] for inp in inputs])
    mids = numpy.array([inp[1] for inp in inputs])
    cats = numpy.array([inp[2] for inp in inputs])

    carte = numpy.stack([item_carte, cate_carte], axis=1)

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), noclk_mid_his, noclk_cat_his, carte

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), carte