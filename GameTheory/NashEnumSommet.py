import numpy as np

class NashEnumSommet(object):
    A, B, l, T, tp = np.array([]), np.array([]), tuple(([], [])), 0, tuple()
    U1, V1 = list(), list()
    X=[]
    Y=[]

    def __init__(self, A,B):
        try:

            if A.shape != B.shape: exit("A.shape != B.shape")
            self.A = A
            self.B = B
            self.tp = A.shape
            if self.tp[0] >= self.tp[1]:
                self.T = 1

            self.X = []
            self.Y = []
            self.X = np.eye(self.tp[0],dtype=np.int8).tolist()
            self.Y = np.eye(self.tp[1],dtype=np.int8).tolist()

            self.U1 = []
            self.V1 = []
            self.l[0].clear()
            self.l[1].clear()


            self.U1.extend(np.zeros((1, self.tp[0]), dtype=np.int8)[0].tolist())
            self.U1.extend(np.ones((1, self.tp[1]), dtype=np.int8)[0].tolist())
            self.U1 = np.array(self.U1)

            self.V1.extend(np.ones((1, self.tp[0]), dtype=np.int8)[0].tolist())
            self.V1.extend(np.zeros((1, self.tp[1]), dtype=np.int8)[0].tolist())
            self.V1 = np.array(self.V1)

        except AttributeError:
            print("type de A,B doit etre 'numpy.ndarray' et meme dimension")

    def p(self, x, X=list(), t=int, T=int):
        if t > 0:
            if x > t:
                Wt = X.copy()
                Wt.append(1)
                self.p(x - 1, Wt, t - 1, T)
                X.append(0)
                self.p(x - 1, X, t, T)
            else:
                X.extend(np.ones((1, t), dtype=int)[0])
                self.l[T].append(X)

        else:
            if x > 0:
                X.extend(np.zeros((1, x), dtype=int)[0])
                self.l[T].append(X)

    def resolut(self):
        global A1, B1

        A1, B1, = list(), list()

        A1.extend(self.X)
        A1.extend(self.B.transpose().tolist())
        A1 = np.array(A1).reshape(self.tp[0] + self.tp[1], self.tp[0])

        B1.extend(self.A)
        B1.extend(self.Y)
        B1 = np.array(B1).reshape(self.tp[0] + self.tp[1], self.tp[1])

        self.p(self.tp[0] + self.tp[1], list(), max(self.tp), 1 - self.T)


        self.l[self.T].extend((np.array(self.l[1-self.T],dtype=int).__neg__()+1).tolist())


        ol = list(zip(self.l[0],self.l[1]))
        nash = np.array(list(map(self.systems_xy, ol)))

        self.l[0].clear()
        self.l[1].clear()

        return nash[nash != None]

    def systems_xy(self, s):
        aa = np.array(A1[np.array(s[0]) == 1])
        bb = np.array(self.U1[np.array(s[0]) == 1])
        x = np.linalg.lstsq(aa, bb, rcond=None)[0]

        a = not (False in (np.less_equal(np.dot(A1[self.U1 == 1], x) - 0.9e-8 ,self.U1[self.U1 == 1])))
        b = not (False in (np.array(x, dtype=np.float16) >= np.float16(-0.9e-8)))

        c = not (False in (np.absolute(np.dot(aa, x) - bb) <= np.float16(9.0e-4)))

        if a and b and c:

            aa = np.array(B1[np.array(s[1]) == 1])
            bb = np.array(self.V1[np.array(s[1]) == 1])
            y = np.linalg.lstsq(aa, bb, rcond=None)[0]

            a = not (False in (np.less_equal(np.dot(B1[self.V1 == 1], y)-0.9e-8,self.V1[self.V1 == 1])))
            b = not (False in (np.array(y, dtype=np.float16) >= np.float16(-0.9e-8)))
            c = not (False in (np.absolute(np.dot(aa,y) - bb) <= np.float16(9.0e-4)))

            if a and b and c:
                if x.sum() != 0:
                    return x/x.sum(), y/y.sum()

    def EQ(self):
        return self.resolut()