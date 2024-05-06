import unittest

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

from graphax import jacve
from graphax.examples import (Simple, Helmholtz, Perceptron, Encoder, RoeFlux_1d,
                                f, g, RoeFlux_3d, RobotArm_6DOF, EncoderDecoder,
                                PropaneCombustion, HumanHeartDipole)
from graphax.examples.deep_learning import encoder_block
from alphagrad.vertexgame import (cross_country, forward, reverse, 
                                make_graph, minimal_markowitz)


def test_function(f, *xs):    
    jaxpr = jax.make_jaxpr(f)(*xs)
    print(jaxpr)
    
    graph = make_graph(f, *xs)
    _, fwd_fmas = jax.jit(forward)(graph)
    _, rev_fmas = jax.jit(reverse)(graph)
    
    order = jax.jit(minimal_markowitz, static_argnums=1)(graph, int(graph[0, 0, 1]))
    order = [int(o) for o in order]
    _, cc_fmas = jax.jit(cross_country)(order, graph)
    
    argnums = list(range(len(xs)))
    _, _fmas = jax.jit(jacve(f, order="fwd", argnums=argnums, count_ops=True))(*xs)
    gx_fwd_fmas = _fmas["num_muls"]
    _, _fmas = jax.jit(jacve(f, order="rev", argnums=argnums, count_ops=True))(*xs)
    gx_rev_fmas = _fmas["num_muls"]
    _, _fmas = jax.jit(jacve(f, order=order, argnums=argnums, count_ops=True))(*xs)
    gx_cc_fmas = _fmas["num_muls"]
    
    print("###")
    print(fwd_fmas, "graphax result:", gx_fwd_fmas)
    print(rev_fmas, "graphax result:", gx_rev_fmas)
    print(cc_fmas, "graphax result:", gx_cc_fmas)
    print("###")
    # 
    if fwd_fmas == gx_fwd_fmas and rev_fmas == gx_rev_fmas and cc_fmas == gx_cc_fmas:
        return True
    else:
        return False


# 76:0
# 84:0
# 4:0
# 5:0
# 9:1
# 10:1
# 12:3
# 14:3
# 22:3
# 23:3
# 30:3
# 31:3
# 37:1
# 39:1
# 45:1
# 51:1
# 70:1
# 73:1
# 75:0
# 77:0
# 78:0
# 83:0
# 85:0
# 86:0
# 119:1
# 122:1
# 125:1
# 3:0
# 6:2
# 16:6
# 24:4
# 27:2
# 32:4
# 35:2
# 38:2
# 40:2
# 46:2
# 52:2
# 54:6
# 57:2
# 59:4
# 60:4
# 62:4
# 66:2
# 69:2
# 72:2
# 79:6
# 81:2
# 87:6
# 89:2
# 91:6
# 92:6
# 93:2
# 94:2
# 95:2
# 97:2
# 98:2
# 100:2
# 102:2
# 103:2
# 105:2
# 107:2
# 109:2
# 110:2
# 112:2
# 114:2
# 116:2
# 118:2
# 120:2
# 123:2
# 126:2
# 128:2
# 130:2
# 21:3
# 25:5
# 28:3
# 29:3
# 33:5
# 36:3
# 43:3
# 44:9
# 53:5
# 56:3
# 63:5
# 67:5
# 82:5
# 90:5
# 101:0
# 104:3
# 108:0
# 111:3
# 115:0
# 117:3
# 124:3
# 129:3
# 133:7
# 135:9
# 11:8
# 13:6
# 15:6
# 19:0
# 20:0
# 47:8
# 50:4
# 55:8
# 65:4
# 96:4
# 99:12
# 106:18
# 113:18
# 127:6
# 80:9
# 88:9
# 41:6
# 49:10
# 61:28
# 121:8
# 137:11
# 18:0
# 64:28
# 68:16
# 131:12
# 26:25
# 34:25
# 1:14
# 48:28
# 71:50
# 74:50
# 132:30
# 7:18
# 8:18
# 58:55
# 2:30
# 42:50
# 17:60


class GraphaxAlignmentTest(unittest.TestCase):
    # # Scalar function tests
    # def test_Simple(self):
    #     result = test_function(Simple, 1., 2.)
    #     self.assertTrue(result)
        
    # def test_RoeFlux_1d(self):
    #     xs = [.01, .02, .02, .01, .03, .03]
    #     result = test_function(RoeFlux_1d, *xs)
    #     self.assertTrue(result)
        
    # def test_RobotArm_6DOF(self):
    #     xs = [.01, .02, .02, .01, .03, .03]
    #     result = test_function(RobotArm_6DOF, *xs)
    #     self.assertTrue(result)
        
    # def test_g(self):
    #     xs = [jnp.array([1.])]*15
    #     result = test_function(g, *xs)
    #     self.assertTrue(result)
    
    # def test_HumanHeartDipole(self):
    #     xs = [.15]*8
    #     result = test_function(HumanHeartDipole, *xs)
    #     self.assertTrue(result)


    # def test_PropaneCombustion(self):
    #     xs = [.15]*11
    #     result = test_function(PropaneCombustion, *xs)
    #     self.assertTrue(result)
        
    # # Vector function tests
    # def test_Helmholtz(self):
    #     xs = jnp.array([.1, .1, .2, .2])
    #     result = test_function(Helmholtz, xs)
    #     self.assertTrue(result)
        
    def test_Perceptron(self):
        key = jrand.PRNGKey(1234)

        x = jnp.ones(4)
        y = jrand.normal(key, (4,))

        w1key, b1key, key = jrand.split(key, 3)
        W1 = jrand.normal(w1key, (8, 4))
        b1 = jrand.normal(b1key, (8,))

        w2key, b2key, key = jrand.split(key, 3)
        W2 = jrand.normal(w2key, (4, 8))
        b2 = jrand.normal(b2key, (4,))

        xs = (x, y, W1, b1, W2, b2, 0., 1.)
        result = test_function(Perceptron, *xs)
        self.assertTrue(result)
        
    # def test_attention(self):
    #     key = jrand.PRNGKey(250197)
    #     x = jnp.ones((4, 4))

    #     wqkey, wkkey, wvkey, key = jrand.split(key, 4)
    #     WQ1 = jrand.normal(wqkey, (4, 4))
    #     WK1 = jrand.normal(wkkey, (4, 4))
    #     WV1 = jrand.normal(wvkey, (4, 4))
            
    #     xs = (x, WQ1, WK1, WV1)

    #     def attn_fn(x, Wq, Wk, Wv):
    #         q = Wq @ x
    #         k = Wk @ x
    #         v = Wv @ x
    #         a = jnn.softmax(q.T @ k, axis=1)
    #         return a @ v
        
    #     result = test_function(attn_fn, *xs)
    #     self.assertTrue(result)
        
    # def test_encoder_block(self):
    #     key = jrand.PRNGKey(250197)
    #     x = jnp.ones((4, 4))

    #     wqkey, wkkey, wvkey, key = jrand.split(key, 4)
    #     WQ1 = jrand.normal(wqkey, (4, 4))
    #     WK1 = jrand.normal(wkkey, (4, 4))
    #     WV1 = jrand.normal(wvkey, (4, 4))
        
    #     wkey, bkey = jrand.split(key, 2)
    #     W = jrand.normal(wkey, (4, 4))
    #     b = jrand.normal(bkey, (4, 1))
            
    #     xs = (x, WQ1, WK1, WV1, W, b, jnp.array([[1.]]), jnp.array([[0.]]))     
    #     result = test_function(encoder_block, *xs)
    #     self.assertTrue(result)
        
    def test_Encoder(self):
        key = jrand.PRNGKey(250197)
        x = jnp.ones((4, 4))
        y = jrand.normal(key, (2, 4))

        wq1key, wk1key, wv1key, key = jrand.split(key, 4)
        WQ1 = jrand.normal(wq1key, (4, 4))
        WK1 = jrand.normal(wk1key, (4, 4))
        WV1 = jrand.normal(wv1key, (4, 4))

        wq2key, wk2key, wv2key, key = jrand.split(key, 4)
        WQ2 = jrand.normal(wq2key, (4, 4))
        WK2 = jrand.normal(wk2key, (4, 4))
        WV2 = jrand.normal(wv2key, (4, 4))

        w1key, w2key, b1key, b2key = jrand.split(key, 4)
        W1 = jrand.normal(w1key, (4, 4))
        b1 = jrand.normal(b1key, (4,))

        W2 = jrand.normal(w2key, (2, 4))
        b2 = jrand.normal(b2key, (2, 1))
        
        xs = (x, y, WQ1, WQ2, WK1, WK2, WV1, WV2, W1, W2, b1, b2, 0., 1., 0., 1.)
        result = test_function(Encoder, *xs)
        self.assertTrue(result)
        
    # def test_EncoderDecoder(self):
    #     key = jrand.PRNGKey(250197)
    #     x = jnp.ones((4, 4))
    #     y = jrand.normal(key, (2, 4))

    #     wq1key, wk1key, wv1key, key = jrand.split(key, 4)
    #     WQ1 = jrand.normal(wq1key, (4, 4))
    #     WK1 = jrand.normal(wk1key, (4, 4))
    #     WV1 = jrand.normal(wv1key, (4, 4))

    #     wq2key, wk2key, wv2key, key = jrand.split(key, 4)
    #     WQ2 = jrand.normal(wq2key, (4, 4))
    #     WK2 = jrand.normal(wk2key, (4, 4))
    #     WV2 = jrand.normal(wv2key, (4, 4))

    #     w1key, w2key, b1key, b2key = jrand.split(key, 4)
    #     W1 = jrand.normal(w1key, (4, 4))
    #     b1 = jrand.normal(b1key, (4,))

    #     W2 = jrand.normal(w2key, (2, 4))
    #     b2 = jrand.normal(b2key, (2, 1))
        
    #     xs = (x, y, WQ1, WQ2, WK1, WK2, WV1, WV2, W1, W2, b1, b2, 0., 1., 0., 1.)
    #     jaxpr = jax.make_jaxpr(Encoder)(*xs)
    #     print(jaxpr)
    #     result = test_function(Encoder, *xs)
    #     self.assertTrue(result)
    
    # def test_f(self):
    #     key = jrand.PRNGKey(250197)
    #     a = jrand.uniform(key, (4,))
    #     b = jrand.uniform(key, (2, 3))
    #     c = jrand.uniform(key, (4, 4))
    #     d = jrand.uniform(key, (4, 1))
    #     xs = (a, b, c, d)
        
    #     jaxpr = jax.make_jaxpr(f)(*xs)
    #     print(jaxpr)
    #     result = test_function(f, *xs)
    #     self.assertTrue(result)
        
    # def test_RoeFlux_3d(self):
    #     ul0 = jnp.array([.1])
    #     ul = jnp.array([.1, .2, .3])
    #     ul4 = jnp.array([.5])
    #     ur0 = jnp.array([.2])
    #     ur = jnp.array([.2, .2, .4])
    #     ur4 = jnp.array([.6])
        
    #     xs = (ul0, ul, ul4, ur0, ur, ur4)
    #     result = test_function(RoeFlux_3d, *xs)
    #     print(result)
    #     self.assertTrue(result)
                
    # vmap and batching tests
        

if __name__ == '__main__':
    unittest.main()

