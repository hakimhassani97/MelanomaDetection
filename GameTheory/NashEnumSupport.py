import numpy as np

class NashEnumSupport(object):
    l, ll, lll = list(), list(), list()
    A, B = np.array([]), np.array([])
    T = int()
    W = tuple()
    def __init__(self,A,B):
        try:
            if A.shape != B.shape: exit("A.shape != B.shape")
            self.l.clear()
            self.A = A
            self.B = B
            self.T = self.min(self.A.shape)
            self.W = A.shape
        except AttributeError:
            print("type de A,B doit etre 'numpy.ndarray' et meme dimension")

    def min(self,w):
        if w[0] >= w[1]:
            return 1
        else:
            return 0

    def p(self,x, X=list(), t=int):
        if t > 0:
            if x > t:
                Wt = X.copy()
                Wt.append(1)
                self.p(x - 1, Wt, t - 1)
                X.append(0)
                self.p(x - 1, X, t)
            else:
                X.extend(np.ones((1, t), dtype=int)[0])
                self.l.append(X)

        else:
            if x > 0:
                X.extend(np.zeros((1, x), dtype=int)[0])
                self.l.append(X)

                if x >= self.W[1 - self.T] - self.W[self.T]:
                    self.ll.append(X[:self.W[self.T]])

    def supp(self,x=list()):

        arr = np.zeros((len(self.l),), dtype=[('var1', list), ('var2', list)])

        arr[:] = (0, x.copy())

        arr['var1'] = self.l.copy()

        arr = list(map(self.lineair,arr))



        list(map(self.test, arr))

        arr = np.array([])

    def lineair(self,xy):
        x = np.array(xy[1 - self.T], dtype=np.float64)
        y = np.array(xy[self.T], dtype=np.float64)
        A1 = self.A.transpose()
        A1 = A1[y == 1]
        A1 = A1.transpose()
        A1 = A1[x == 1]
        z = y.copy()
        y[y == 1] = np.linalg.lstsq(A1, y[y == 1], rcond=None)[0]
        if y.sum() != 0:
            y = y / y.sum()

        B1 = self.B[x == 1]
        B1 = B1.transpose()
        B1 = B1[z == 1]
        x[x == 1] = np.linalg.lstsq(B1, x[x == 1], rcond=None)[0]
        if x.sum() != 0:
            x = x / x.sum()
        return (x, y)


    def test(self,xy):

        if (np.float16(xy[0].min()) >= np.float16(-0.0)) and (np.float16(xy[1].min()) >= np.float16(-0.0)) and (np.float16(xy[0].max()) > np.float16(0.0)) and (np.float16(xy[1].max()) > np.float16(0.0)):

            U = np.dot(self.A, xy[1])
            V = np.dot(xy[0], self.B)
            if (np.float16(U[xy[0] > np.float16(-0.)].min()) >= np.float16(U.max())) and (np.float16(V[xy[1] > np.float16(-0.)].min()) >= np.float16(V.max())):

                self.lll.extend([(xy[0].tolist(),xy[1].tolist())])


    def enum_support(self):

        for t in range(1, self.W[self.T]+ 1, 1):
            self.p(self.W[1 - self.T], list(), t)

            if self.W[0] == self.W[1]: self.ll = self.l.copy()

            list(map(self.supp, self.ll))
            self.ll.clear()
            self.l.clear()
        return self.lll


    def EQ(self):
        return self.enum_support()
