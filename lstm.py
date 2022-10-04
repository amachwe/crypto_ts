import numpy as np





wh = np.array([0.1,0.2,0.3])
wx = np.array([0.3,.2,.3])
b = np.zeros(3)

h = np.array([1,2,3])
x = np.array([2,3,2])

class LSTMCell(object):

    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        shape = (1,num_neurons)

        self.Wxo = np.random.random(shape)
        self.Who = np.random.random(shape)
        self.bo = np.random.random(1)

        self.Wxi = np.random.random(shape)
        self.Whi = np.random.random(shape)
        self.bi = np.random.random(1)

        self.Wxf = np.random.random(shape)
        self.Whf = np.random.random(shape)
        self.bf = np.random.random(1)

        self.Wxg = np.random.random(shape)
        self.Whg = np.random.random(shape)
        self.bg = np.random.random(1)
    @staticmethod
    def sig_path(wh,wx,h,x,b):
        return 1/(1+np.exp(-(np.matmul(wh.transpose(),h)+np.matmul(wx.transpose(),x)+b)))

    @staticmethod
    def tanh_path(wh,wx,h,x,b):
        return np.tanh(-(np.matmul(wh.transpose(),h)+np.matmul(wx.transpose(),x)+b))

    def activation(self,X,hp,cp):
        i = LSTMCell.sig_path(self.Whi,self.Wxi,hp,X,self.bi)
        f = LSTMCell.sig_path(self.Whf,self.Wxf,hp,X,self.bf)
        o = LSTMCell.sig_path(self.Who,self.Wxo,hp,X,self.bo)
        g = LSTMCell.tanh_path(self.Whg,self.Wxg,hp,X,self.bg)

        c = (f*cp)+(i*g)
        y = h = o*np.tanh(c)

        return y,h,c


if __name__ == "__main__":

    cell = LSTMCell(6)


  

    X = np.array([
        [1,1,0],
        [0,0,1],
        [1,0,1],
        [0,0,0],
        [1,1,1],
        [0,1,0]
    ])

    h = np.zeros((1,3))
    c = np.zeros((1,3))
    for x in X:
        y,_,_ = cell.activation([x],h,c)
    
        print(x," - ",y)

    print("Y",y)
    print("H",h)
    print("C",c)

    
