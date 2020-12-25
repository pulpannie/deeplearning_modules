import numpy as np
from skimage.util.shape import view_as_windows

#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        ## If necessary, you can define additional class variables here
        self.filter_width = filter_width
        self.filter_height = filter_height
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def convolve(self, a, b, stride=1):
        (batch_size, _, in_width, in_height) = a.shape
        (num_filters, in_ch_size, filter_width, filter_height) = b.shape
        output_w, output_h = in_width - filter_width + 1, in_height - filter_height + 1
        result = np.array([])
        print('****************************************')
        print('just W', b.shape)
        print("reshaped W", b.reshape(b.shape[0], -1).T.shape, b.reshape(b.shape[0], b.shape[1], -1).T.shape)
        if a.shape[1] == b.shape[1]:
            print("if")
            print(a.shape[0])
            for i in range(a.shape[0]):
                tmp = np.array([])
                for j in range(b.shape[0]):
                    W = b[j].reshape(-1)
                    print("reshaped W", W.shape)
                    y = view_as_windows(a[i], (a.shape[1], b.shape[2], b.shape[3]))
                    print('////////////////////////////////////////')
                    print("windowed y", y.shape)
                    y = y.reshape((y.shape[0], y.shape[1], y.shape[2], -1))
                    print("reshaped y", y.shape)
                    result_ = y.dot(W.T)
                    print("result shape", result_.shape)
                    result_ = result_.reshape(result_.shape[0], result_.shape[1], result_.shape[2])
                    tmp = np.append(tmp, result_)
                result = np.append(result, tmp)
            print("final result shape", result.shape)
            result = result.reshape(a.shape[0], b.shape[0],result_.shape[2],result_.shape[2])
        else:
            print("else")
            print(a.shape[0])
            for i in range(a.shape[0]):
                W = b[i].reshape(b.shape[1],-1)
                print("reshaped W", W.shape)
                y = view_as_windows(a[i], (a.shape[1], b.shape[2], b.shape[3]))
                print('////////////////////////////////////////')
                print("windowed y", y.shape)
                y = y.reshape((y.shape[0], y.shape[1], y.shape[2], y.shape[3], -1))
                print("reshaped y", y.shape)
                result_ = y.dot(W.T)
                print("result shape", result_.shape)
                result_ = result_.reshape(result_.shape[0], -1, result_.shape[-1])
                print("reshaped result_ shape", result_.shape)
                result = np.append(result, result_)
            print("final result shape", result.shape)
            # result = result.reshape(a.shape[0], a.shape[0], a.shape[1], a.shape[1], a.shape[1])
            # result = np.sum(result, axis=0)
            # result = result.reshape(a.shape[0], a.shape[1],a.shape[1],a.shape[1])
            result = result.reshape(num_filters, in_ch_size, a.shape[1], output_w, output_h)
            result = np.sum(result, axis=0)
            result = result.reshape(in_ch_size,a.shape[1],output_w,output_h)
        return result

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        print("x", x.shape)
        print("W", self.W.shape)
        result = self.convolve(x,self.W)
        result += self.b
        return result

    #######
    # Q2. Complete this method
    #######
    def backprop(self, x, dLdy):
        print('///////////CONV BACKPROP/////////////////')
        print("X",x.shape)
        print("DLDY", dLdy.shape)
        dLdW = self.convolve(x, dLdy)
        #dLdx = np.dot(self.W)

        print("b shape", self.b.shape)
        dLdb = np.sum(dLdy, axis=(0,2,3))
        dLdb= dLdb.reshape(self.b.shape)
        print("dLdb", dLdb.shape)
        tmp = int(x.shape[2] - dLdy.shape[2])
        pad_dLdy = np.pad(dLdy, ((0,0),(0,0),(tmp,tmp),(tmp,tmp)))
        W = np.transpose(self.W, [1, 0, 2, 3])[:, :, ::-1, ::-1]
        print("rotated w", W.shape)
        print("padded dLdy", pad_dLdy.shape)
        dLdx = self.convolve(pad_dLdy, W)
        print("dLdx", dLdx.shape)
        return dLdx, dLdW, dLdb

    #######
    ## If necessary, you can define additional class methods here
    #######


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
        print("maxpool forward x", x.shape, x[0,0,0])
        print("maxpool forward x", x.shape, x[0, 0, 1])
        y = np.zeros(
            (int(x.shape[0]), int(x.shape[1]), int(x.shape[2] / 2), int(x.shape[3] / 2), self.stride, self.stride))
        print("maxpool x", x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y[i, j] = view_as_windows(x[i, j, :, :], (self.pool_size, self.pool_size), self.stride)
        print("maxpool y", y[0,0,0])
        print('maxpool////////////////////////////////////////')
        y = y.reshape((y.shape[0], y.shape[1], y.shape[2], y.shape[3], -1))
        out = np.max(y, axis=4)
        print("forward out shape", out.shape, out[0,0])
        return out

    #######
    # Q4. Complete this method
    #######
    def backprop(self, x, dLdy):
        print('maxpool////////////////////////////////////////')
        print("dLdy.shape", dLdy.shape, x.shape)
        dLdy_w, dLdy_h = dLdy.shape[2], dLdy.shape[3]

        y_ = np.zeros((int(x.shape[0]), int(x.shape[1]), int(x.shape[2]/2), int(x.shape[3]/2), self.stride, self.stride))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y_[i,j] = view_as_windows(x[i,j,:,:],(self.pool_size, self.pool_size),self.stride)
        y = y_.reshape((y_.shape[0],y_.shape[1],y_.shape[2],y_.shape[3],-1))

        out_index = np.argmax(y, axis=4)

        local = np.zeros((x.shape[0] * x.shape[1] * dLdy_w * dLdy_h, self.pool_size * self.pool_size))
        local[np.arange(len(local)), out_index.flatten()] = 1
        print(local)
        mask = local.reshape(x.shape[0], x.shape[1], y_.shape[2], y_.shape[2],self.pool_size, self.pool_size)

        dLdy = dLdy.repeat(self.pool_size, axis=2).repeat(self.pool_size, axis=3)


        y_ = np.zeros((int(x.shape[0]), int(x.shape[1]), int(x.shape[2]/2), int(x.shape[3]/2), self.stride, self.stride))
        for i in range(dLdy.shape[0]):
            for j in range(dLdy.shape[1]):
                y_[i,j] = view_as_windows(dLdy[i,j,:,:],(self.pool_size, self.pool_size),self.stride)

        dLdx = mask * y_

        tmp = np.zeros((x.shape[0],x.shape[1],x.shape[2], x.shape[3]))

        for x in range(dLdy_w):
            for y in range(dLdy_h):
                x_ = x * self.stride
                y_ = y * self.stride
                tmp[:, :, x_:x_ + self.pool_size, y_:y_ + self.pool_size] += dLdx[:, :, x, y]
        dLdx = tmp
        return dLdx

    #######
    ## If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    #x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    x = np.ones((batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')