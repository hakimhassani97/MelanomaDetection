import numpy as np

class NashLemkHowson(object):
    A, B = np.array([]), np.array([])
    lxy = tuple()
    U1, V1 = list(), list()
    tp = tuple()
    nash, K_initial = tuple(), np.int8()

    def __init__(self, A, B):
        try:
            if A.shape != B.shape or A.shape == ():
                exit("A.shape != B.shape == ()")

            self.K_initial = 0
            self.tp = A.shape
            self.lxy = (np.array([]), np.array([]))
            self.A = A
            self.B = B
            self.U1 = []
            self.V1 = []

            self.U1.extend(
                np.zeros((1, self.tp[0]), dtype=np.int8)[0].tolist())
            self.U1.extend(np.ones((1, self.tp[1]), dtype=np.int8)[0].tolist())
            self.U1 = np.array(self.U1)

            self.V1.extend(np.ones((1, self.tp[0]), dtype=np.int8)[0].tolist())
            self.V1.extend(
                np.zeros((1, self.tp[1]), dtype=np.int8)[0].tolist())
            self.V1 = np.array(self.V1)

            self.lxy = (self.V1.copy(), self.U1.copy())
        except AttributeError:
            print("type de A,B doit etre de type 'numpy.ndarray'  et meme dimension")
    def resolut(self, l, t):
        global A1, B1
        A1, B1 = list(), list()

        if t == 0:
            A1.extend(np.eye(self.tp[0]).tolist())
            A1.extend(self.B.transpose().tolist())
            A1 = np.array(A1).reshape(self.tp[0] + self.tp[1], self.tp[0])
            return self.systems_x(l)
        if t == 1:
            B1.extend(self.A)
            B1.extend(np.eye(self.tp[1]).tolist())
            B1 = np.array(B1).reshape(self.tp[0] + self.tp[1], self.tp[1])
            return self.systems_y(l)
    def systems_x(self, s=list()):
        aa = np.array(A1[s == 1])
        bb = np.array(self.U1[s == 1])
        x = np.linalg.lstsq(aa, bb, rcond=None)[0]

        a = not (False in (np.less_equal(
            np.dot(A1[self.U1 == 1], x) - 0.9e-8, self.U1[self.U1 == 1])))
        b = not (False in (np.array(x, dtype=np.float16) >= np.float16(-0.9e-8)))
        c = not (False in (np.absolute(np.dot(aa, x) - bb) <= np.float16(9.0e-4)))

        if a and b and c:
            return (True, x)
        else:
            return False
    def systems_y(self, s=list()):

        aa = np.array(B1[s == 1])
        bb = np.array(self.V1[s == 1])
        y = np.linalg.lstsq(aa, bb, rcond=None)[0]

        a = not (False in (np.less_equal(
            np.dot(B1[self.V1 == 1], y) - 0.9e-8, self.V1[self.V1 == 1])))
        b = not (False in (np.array(y, dtype=np.float16) >= np.float16(-0.9e-8)))
        c = not (False in (np.absolute(np.dot(aa, y) - bb) <= np.float16(9.0e-4)))

        if a and b and c:
            return (True, y)
        else:
            return False

    ################################################################################################
    def string(self, l):
        return str.join('', list(str(l[_]) for _ in range(len(l))))
    ################################################################

    def lemke_howson_(self, D=0):
        V1, V2 = np.array(self.V1.copy()), np.array(self.U1.copy())
        r1, r2, t = False, False, D
        b = True

        MMr = [(self.string(np.zeros((1, self.tp[0]), dtype=np.float16)[0].tolist(
        )), self.string(np.zeros((1, self.tp[1]), dtype=np.float16)[0].tolist()))]
        ########
        while b:
            vi = ((-1)*V1+1).nonzero()[0].tolist()
            V1[t] = 0
            for i in vi:
                V1[i] = 1
                r1 = self.resolut(V1.copy(), 0)

                if r1 != False:
                    t = i
                    break
                V1[i] = 0

            if r1 == False:
                return False
            ############################
            if t == D:
                if r2 == False:
                    return False
                if r1[1].sum() == 0 or r2[1].sum() == 0:
                    return False
                return np.float16(r1[1]/r1[1].sum()), np.float16(r2[1]/r2[1].sum())
            #######################################
            vj = ((-1) * V2 + 1).nonzero()[0].tolist()
            V2[t] = 0
            for j in vj:
                V2[j] = 1
                r2 = self.resolut(V2.copy(), 1)
                if r2 != False:
                    t = j
                    break
                V2[j] = 0

            if r2 == False:
                return False
            ############################
            if t == D:
                if r1[1].sum() == 0 or r2[1].sum() == 0:
                    return False
                return np.float16(r1[1] / r1[1].sum()), np.float16(r2[1] / r2[1].sum())

            x = self.string(r1[1].tolist())
            y = self.string(r2[1].tolist())
            if MMr.__contains__((x, y)):
                return False
            MMr.append((x, y))
            ########
    #####################################################################################################

    def EQ(self):
        j = list(range(self.tp[0]))
        eq = False
        while (eq == False) and len(j) != 0:
            eq = self.lemke_howson_(j.pop(j.index(np.random.choice(j))))
        if eq != False:
            return eq
        else:
            return np.zeros((1, self.tp[0]), dtype=np.float16)[0], np.zeros((1, self.tp[1]), dtype=np.float16)[0]

    #################################################
    def EQs(self):
        l = []
        for i_ in range(0, self.tp[0]):
            l.append(self.lemke_howson_(i_))
        return l
