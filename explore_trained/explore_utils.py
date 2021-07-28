import pickle
import numpy as np


def loadnet(fname):
    with open(fname, 'rb') as f:
        d = pickle.load(f)
    return d["layers"], d["allwts"], d["training_params"]


def savenet(fname, specs, wts, tps):
    d = {"layers":specs, "allwts":wts, "training_params":tps}
    with open(fname, "wb") as f:
        pickle.dump(d, f, 4)
    print("Saved ", fname)


def tofloat32(fname):
    S, W, T = loadnet(fname)
    for i, wb in enumerate(W):
        for j, w in enumerate(wb):
            W[i][j] = w.astype('float32')
    outfn = fname[:-4] + '_f32.pkl'
    savenet(outfn, S, W, T)


def norms(w):
    if w.ndim == 2:
        return np.sqrt(np.sum(w**2, axis=0))
    elif w.ndim == 4:
        return np.sqrt(np.sum(w**2, axis=(1, 2, 3)))


def printspecs(specs, wts):
    for i in range(len(specs)):
        name, params = specs[i]
        wb = wts[i]
        print(f"Layer: {i}: {name}\n\t{params}")
        for w in wb:
            print(f"\t{w.shape} {w.max():.4f} {w.min():.4f} {norms(w)}")

def savedense(a, m, w, h):
    nin, nout = a.shape
    assert m*w*h == nin, f"{m*w*h} vs {nin}"

    img = np.zeros((nin, nout, 3), dtype='uint8')
    apos = np.clip(a, 0, 1)
    img[:,:,0] += (255*apos/apos.max()).astype('uint8')
    aneg = np.clip(-a, 0, 1)
    img[:,:,1] += (255*aneg/aneg.max()).astype('uint8')
    
    b = img.reshape((m, w, h, nout, 3))
    b = b.swapaxes(0, 1).swapaxes(1, 2)
    print(b.shape)
    
    b[:,:,:,-1,:] = 255
    b[:,:,-1,:,:] = 255
    c = np.vstack([
            np.hstack([b[i][j] 
                for j in range(w)])
                    for i in range(h)])

    
    print(c.shape)
    Image.fromarray(c).save("densearray.png")

