from lab_cheat import *
import numpy as np
import unittest


class TestDigitalAccuracy(unittest.TestCase):
    def setUp(self):
        self.var = var

    def test_1(self):
        self.assertEqual(self.var.digital_accuracy(127.8, 8.56), 0)

    def test_2(self):
        self.assertEqual(self.var.digital_accuracy(127.8, 2.56), -1)

    def test_3(self):
        self.assertEqual(self.var.digital_accuracy(1027.8, 250.6), 1)

    def test_4(self):
        self.assertEqual(self.var.digital_accuracy(127.8, 0.256), -2)


class TestDigitalNormalizeStr(unittest.TestCase):
    def setUp(self):
        self.var = var

    def test_1(self):
        self.assertEqual(self.var.digital_normalize_str(Var(127.8, 8.56)), r'128 \pm 9')

    def test_2(self):
        self.assertEqual(self.var.digital_normalize_str(Var(127.8, 2.56)), r'127.8 \pm 2.6')

    def test_3(self):
        self.assertEqual(self.var.digital_normalize_str(Var(1027.8, 250.6)), r'1030 \pm 250')

    def test_4(self):
        self.assertEqual(self.var.digital_normalize_str(Var(127.8, 0.256)), r'127.80 \pm 0.26')


class TestDigitalNormalizeTuple(unittest.TestCase):
    def setUp(self):
        self.var = var

    def test_1(self):
        self.assertEqual(self.var.digital_normalize_tuple(Var(127.8, 8.56)), (128, 9))

    def test_2(self):
        self.assertEqual(self.var.digital_normalize_tuple(Var(127.8, 2.56)), (127.8, 2.6))

    def test_3(self):
        self.assertEqual(self.var.digital_normalize_tuple(Var(1027.8, 250.6)), (1030, 250))

    def test_4(self):
        self.assertEqual(self.var.digital_normalize_tuple(Var(127.8, 0.256)), (127.80, 0.26))


if __name__ == '__main__':
    unittest.main()

m1 = GroupVar([Var(0.5139, 0.0001), Var(0.5061, 0.0001), Var(0.5039, 0.0001), Var(0.5073, 0.0001)])
m2 = GroupVar([Var(0.5049, 0.0001), Var(0.5136, 0.0001), Var(0.5011, 0.0001), Var(0.5043, 0.0001)])
x = [Var(0.135, 0.001), Var(0.138, 0.001), Var(0.135, 0.001), Var(0.135, 0.001)]
x = GroupVar(x)
M1 = 0.7296
M2 = 0.7299
g = 9.8
L = [Var(2.23, 0.01)]*4
t1 = [Var(19.59, 0.03)]*4
t2 = [Var(14.68, 0.03)]*4
d = [Var(0.48, 0.01)]*4
r = [Var(0.25, 0.01)]*4
R = [Var(0.34, 0.01)]*4
L = GroupVar(L)
t1 = GroupVar(t1)
t2 = GroupVar(t2)
d = GroupVar(d)
r = GroupVar(r)
R = GroupVar(R)
tim = GroupVar([Var(43.5, 0.1), Var(43.9, 0.1), Var(43.7, 0.1), Var(44.3, 0.1), Var(44.8, 0.1)])
kI = 2*np.pi*(M1+M2)*R**2*t1/(t1**2 - t2**2)
u = x*kI*1000/(2*d*m2*r)*0.4
t = TexTable()
t.add(u, r'u, \frac{м}{с}', show_err=True)
t.show(table=True, floatrow=True)

