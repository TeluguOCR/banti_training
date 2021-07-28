#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import ast
import bz2
import json
import pickle
import socket
import traceback
import datetime

import numpy as np
import theano as th
import theanet.neuralnet as nn

######################################################################### Common Helper Functions For Training


def parse_sys_args():
    if len(sys.argv) < 3:
        print('Usage:', sys.argv[0],
              ''' <x.bz2> <y.bz2> [auxillary.bz2] <params_file(s)> [redirect=0]
        .bz2 files contain the samples and the output classes as generated
            by the gen_cnn_data.py script (or the like).
        params_file(s) :
            Parameters for the NeuralNet
            - params_file.py  : contains the initialization code
            - params_file.pkl : pickled file from a previous run (has wts too).
        redirect:
            1 - redirect stdout to a <SEED>.txt file
        ''')
        sys.exit()
    imgs_fname = sys.argv[1]
    lbls_fname = sys.argv[2]
    aux_fname = None
    prms_fnames = []
    write2file = sys.argv[-1] == '1'
    if len(sys.argv) > 3:
        if sys.argv[3].endswith('.bz2'):
            aux_fname = sys.argv[3]
            prms_fnames = sys.argv[4:]
        else:
            prms_fnames = sys.argv[3:]
    if len(prms_fnames) > 0 and prms_fnames[-1] == '1':
        prms_fnames.pop()

    return imgs_fname, lbls_fname, aux_fname, prms_fnames, write2file


def read_json_bz2(path2data, dtype=None):
    if path2data.endswith('.bz2'):
        bz2_fp = bz2.BZ2File(path2data, 'r')
        data = np.array(json.loads(bz2_fp.read().decode('utf-8')), dtype=dtype)
        bz2_fp.close()
    elif path2data.endswith('.json'):
        with open(path2data, "r") as jsonfp:
            data = np.array(json.load(jsonfp), dtype=dtype)
    return data


def read_prms_file(prms_file_name):
    if prms_file_name.endswith('.pkl'):
        with open(prms_file_name, 'rb') as f:
            params = pickle.load(f)
    else:
        with open(prms_file_name, 'r') as f:
            params = ast.literal_eval(f.read())
    return params


def share(data, dtype=th.config.floatX, borrow=True):
    return th.shared(np.asarray(data, dtype), borrow=borrow)


def get_shared_data(n_tr, datax, datay, dataaux=None):
    trin_x = share(datax[:n_tr, ])
    test_x = share(datax[n_tr:, ])
    trin_y = share(datay[:n_tr, ], 'int32')
    test_y = share(datay[n_tr:, ], 'int32')
    if dataaux is not None:
        trin_aux = share(dataaux[:n_tr, ])
        test_aux = share(dataaux[n_tr:, ])
    else:
        trin_aux, test_aux = None, None

    return trin_x, test_x, trin_y, test_y, trin_aux, test_aux


class WrapOut:
    def __init__(self, use_file, name=''):
        self.name = name
        self.use_file = use_file
        if use_file:
            self.stream = open(name, 'w', 1)
        else:
            self.stream = sys.stdout

    def write(self, data):
        self.stream.write(data)

    def forceflush(self):
        if self.use_file:
            self.stream.close()
            self.stream = open(self.name, 'a', 1)

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def test_wrapper(nylist):
    sym_err, bit_err, n = 0., 0., 0
    for symdiff, bitdiff in nylist:
        sym_err += symdiff
        bit_err += bitdiff
        n += 1
    return sym_err/n, bit_err/n


#####################################################################  Process parameters and read data
imgs_file_name, lbls_file_name, aux_file_name, prms_file_names, write_to_file = parse_sys_args()

print("\nLoading the data ...")
data_y = read_json_bz2(lbls_file_name)
data_x = read_json_bz2(imgs_file_name, dtype=th.config.floatX)
if data_x.ndim == 3:
    data_x = np.expand_dims(data_x, axis=1)
data_aux = read_json_bz2(aux_file_name) if aux_file_name else None

corpus_sz, _, imgsz, _ = data_x.shape
stdout_orig = sys.stdout


def train(prms_file_name, out_to_file=True):
    ##########################################  Import Parameters
    params = read_prms_file(prms_file_name)
    layers = params['layers']
    layers[0][1]['img_sz'] = imgsz
    tr_prms = params['training_params']
    try:
        allwts = params['allwts']
    except KeyError:
        allwts = None

    ## Init SEED
    if ('SEED' not in tr_prms) or (tr_prms['SEED'] is None):
        tr_prms['SEED'] = np.random.randint(0, 1e6)

    out_file_head = os.path.basename(prms_file_name, ).replace(
        os.path.splitext(prms_file_name)[1], f"_{tr_prms['SEED']:06d}")

    if out_to_file:
        print(f"Printing output to {out_file_head}.txt", file=sys.stderr)
        sys.stdout = WrapOut(True, out_file_head + '.txt')
    else:
        sys.stdout = WrapOut(False)

    ##########################################  Print Parameters
    print('Batch Command: ', ' '.join(sys.argv))
    print('Processing: ', prms_file_name)
    print('Time   :' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f'Device : {th.config.device} ({th.config.floatX})')
    print('Host   :', socket.gethostname())

    print(nn.get_layers_info(layers))
    print(nn.get_training_params_info(tr_prms))
    print(f"X (samples, dimensions): {data_x.shape} {data_x.nbytes // 1000}KB\n"
          f"X (min, max) : {data_x.min()} {data_x.max()}\n"
          f"Y (samples, dimensions): {data_y.shape} {data_y.nbytes // 1000}KB\n"
          f"Y (min, max) : {data_y.min()} {data_y.max()}")

    n_train = int(corpus_sz * tr_prms['TRAIN_ON_FRACTION'])
    trin_x, test_x, trin_y, test_y, trin_aux, test_aux = get_shared_data(n_train, data_x, data_y, data_aux)

    ##########################################  Build Network
    print("\nInitializing the net ... ")
    net = nn.NeuralNet(layers, tr_prms, allwts)
    print(net)
    print(net.get_wts_info(detailed=True).replace("\n\t", ""))

    print("\nCompiling ... ")
    training_fn = net.get_trin_model(trin_x, trin_y, trin_aux)
    test_fn_tr = net.get_test_model(trin_x, trin_y, trin_aux)
    test_fn_te = net.get_test_model(test_x, test_y, test_aux)

    ##########################################  Prepare to train
    batch_sz = tr_prms['BATCH_SZ']
    tr_corpus_sz = n_train
    te_corpus_sz = corpus_sz - n_train
    nEpochs = tr_prms['NUM_EPOCHS']
    nTrBatches = tr_corpus_sz // batch_sz
    nTeBatches = te_corpus_sz // batch_sz
    err2_name = 'P(MLE)'

    def get_test_indices(tot_samps, bth_samps=tr_prms['TEST_SAMP_SZ']):
        n_bths_each = int(bth_samps / batch_sz)
        n_bths_all = int(tot_samps / batch_sz)
        cur = 0
        while True:
            yield [i % n_bths_all for i in range(cur, cur + n_bths_each)]
            cur = (cur + n_bths_each) % n_bths_all

    test_indices = get_test_indices(te_corpus_sz)
    trin_indices = get_test_indices(tr_corpus_sz)
    pickle_file_name = out_file_head + '_{:02.0f}.pkl'
    saved_file_name = None

    def do_test():
        nonlocal saved_file_name
        test_err, test_err2 = test_wrapper(test_fn_te(i) for i in next(test_indices))
        trin_err, trin_err2 = test_wrapper(test_fn_tr(i) for i in next(trin_indices))
        print(f"{trin_err:6.2%}  ({trin_err2:6.2%})      {test_err:6.2%}  ({test_err2:6.2%})")
        if out_to_file:
            print(f"({test_err:.0%})", end='.. ', file=stdout_orig)
        sys.stdout.forceflush()

        if saved_file_name is not None:
            os.remove(saved_file_name)

        saved_file_name = pickle_file_name.format(100*test_err)
        with open(saved_file_name, 'wb') as pkl_file:
            pickle.dump(net.get_init_params(), pkl_file, 3)

    ############################################ Training Loop
    print("Training ...")
    print(f"Epoch   Cost  Tr_Error Tr_{err2_name}    Te_Error Te_{err2_name}")
    initial_cost_estimate = training_fn(0)[0] * nTrBatches
    print(f"{-1:3d} {initial_cost_estimate:>8.2f}", end='    ')
    if out_to_file:
        print(-1, end='', file=stdout_orig)
    do_test()

    try:
        for epoch in range(nEpochs):
            total_cost = 0

            for ibatch in range(nTrBatches):
                output = training_fn(ibatch)
                total_cost += output[0]

                if np.isnan(total_cost):
                    print(f"Epoch:{epoch} Iteration:{ibatch}")
                    print(net.get_wts_info(detailed=True))
                    raise ZeroDivisionError(f"Nan cost at Epoch:{epoch} Iteration:{ibatch}")

            if epoch % tr_prms['EPOCHS_TO_TEST'] == 0:
                print(f"{net.get_epoch():3d} {total_cost:>8.2f}", end='    ')
                if out_to_file:
                    print(epoch, end='', file=stdout_orig)
                do_test()
                if total_cost > 1e6:
                    print("Cost too high! Reduce init learning rate! Quitting!")
                    break

            net.inc_epoch_set_rate()

    except KeyboardInterrupt:
        print("Interupted at epoch", net.get_epoch(), file=stdout_orig)

    ########################################## Final Error Rates
    teerr, teerr2 = test_wrapper(test_fn_te(i) for i in range(nTeBatches))
    trerr, trerr2 = test_wrapper(test_fn_tr(i) for i in range(nTrBatches))
    print(f"{net.get_epoch():3d} {0:>8.2f}    {trerr:6.2%}  ({trerr2:6.2%})      {teerr:6.2%}  ({teerr2:6.2%})")
    sys.stdout = stdout_orig


def train_all(file_patterns):
    for file_name in file_patterns:
        print("################################  Processing: ", file_name)
        try:
            train(file_name, out_to_file=write_to_file)
            print("Succesfully Done.")
        except:
            sys.stdout = stdout_orig
            print("Unexpected error while processing ", file_name)
            traceback.print_exc()


train_all(prms_file_names)
