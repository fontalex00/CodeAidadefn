
import z03aidasource as z03
from   z98aidaglobar import *
import z99aidaglobas as z

def Unsquand(at,whts,pos):
   bt = at.unsqueeze_(pos)
   bt = bt.expand_as(whts)
   return bt

def Funksin (x):
   return x

def Damper (x,tp):
   bc = (abs(x-tp)/abs(1-tp))**2
   bc = torch.clamp(bc,0,1)
   return bc

def Sigmost (xt,tp,sl):
   zt = torch.sigmoid (sl*(xt-tp))
   rt = torch.sigmoid (20*(xt-tp))
   bc = Damper (xt,tp)
   yt = bc*rt +(1-bc)*zt
   return yt

def Tanhost (xt,tp,sl):
   zt = torch.tanh (sl*(xt-tp))
   rt = torch.tanh (20*(xt-tp))
   bc = Damper (xt,tp)
   yt = bc*rt +(1-bc)*zt
   return yt

def Asfunco(x,lr,b):
   hd = torch.tensor(HD)
   if lr <=-1: hu = 1.00+lr/5
   if lr == 1: hu = hd[lr]+0.50-b/10
   if lr >= 2: hu = hd[lr]+0.50-b/10
   p = Sigmost(x,hu,10)
   q = Sigmost(x,hu,10)
   y = torch.where(x<=hu,p,q)
   return y

def Funkoux (x,lr,net,nd=-1):
   if abs(lr) == 0:
      if SBOL == 0: return Sigmost(x,0.5,6)
      if SBOL ==-1: return Tanhost(x,0.0,3)
   exec(closed)
   if lr ==lr: b = 0
   if lr >= 1: b = net.nl[lr].bias.clone()
   if lr >= 1 and nd !=-1: b = b[nd]
   f = 0.03*(x+1.1)**1
   #ifabs(lr) == 1: y = Pcewise(x,lr,b)
   if abs(lr) >= 1: y = Asfunco(x,lr,b)
   y = torch.where(y<=f,f,y)
   y = torch.clamp(y,0,1)
   if SBOL ==-1: y = (y-0.5)*2
   return y

def Funkous (x,lr,net,nd=-1):
   b = net.nl[lr].bias.clone()
   bt= Unsquand(b,x,0)
   #y = torch.relu(x)
   #y = Sigmost(x,0.5,6)
   y = x#-HD+bt/10
   return y

def Funkasd (x,ts,hp,up):
   at = torch.ones_like(x)*ts
   d = 0.05*(x+0.0)
   h = torch.sigmoid(hp*(x-ts))
   u = torch.sigmoid(up*(x-ts))
   d = torch.where(d < h,1*h,d)
   d = torch.where(x >at,1*u,d)
   d = torch.where(d < 0,0*d,d)
   return d

def Funkbot (x,ts,hp,up):
   at = torch.ones_like(x)*0.5
   h = Funkasd(2*0.5-x,ts,hp,up)
   u = Funkasd(0*0.5+x,ts,hp,up)
   z = torch.where(x <at,h,u)
   return z

def Whtconcallo(u,lr):
   uhtar = u.data.clone()
   res =torch.ones(z.lrsize[lr])
   for no in range(z.lrsize[lr]):
      uht = uhtar[no]
      uht [uht < 0] = 0
      uht = uht / BUDL
      ubzcnc = (0.10 **2)*10
      uhtcnc = torch.sum(uht**2,0)
      res[no]= torch.tanh(2*(uhtcnc/ubzcnc))
   exec(closed)
   return res

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Boardsatu(oldar,oldas,net,lr):
   len0 = list(oldar.size())[0]
   #len1= list(oldar.size())[1]
   board = torch.Tensor(17,len0)
   board.fill_(1)
   hd = torch.tensor(HD)

   for ii in range(2):
      if ii == 0: dc = oldar
      if ii == 1: dc = oldas
      b = net.nl[lr].bias.clone() / 10
      bc= dc -hd[lr]+Unsquand(b,dc,-1)
      coer = 0.75-lr*0.00
      coes = 0.70+lr*0.05
      if ii ==ii: f2 = 1*dc
      if ii ==ii: f3 = Funkbot (0*0.50+bc,0.60,20,28)
      if ii == 0: f4 = Funkasd (0*0.50+bc,coer,20,28)
      if ii == 1: f4 = Funkasd (0*0.50+bc,coes,20,28)
      if ii == 0: f5 = Funkasd (2*0.50-bc,coer,20,28)
      if ii == 1: f5 = Funkasd (2*0.50-bc,coes,20,28)
      f3 = z03.Topbot(f3,90,False)[0]
      board[2+ii*8] = torch.std (f2,1)
      board[3+ii*8] = torch.mean(f3,1)
      board[4+ii*8] = torch.mean(f4,1)
      board[5+ii*8] = torch.mean(f5,1)
      board[6+ii*8] = Whtconcallo(net.nl[lr].weight,lr)
   exec(closed)

   r3 = 1*board[3] +0*board[11]
   r4 = 1*board[4] +0*board[12]
   s4 = 0*board[4] +1*board[12]
   r5 = 1*board[5] +0*board[13]
   s5 = 0*board[5] +1*board[13]
   wt = 1*board[6] +0*board[14]
   r3 = torch.tanh(3 *r3)
   if lr == 4: s4 = 0
   if lr ==lr: s5 = 0
   h4 = torch.tanh(14*(r4-s4))
   h5 = torch.tanh(14*(r5-s5))
   #zs = Funkasd(1-zs,0.90,20,28)
   board[1] = 0.60*h4*h5 +0.40*r3*h4*h5 +0.00*wt
   # spread r, polarisation, spread s (andness)

   # fitness sharing
   uht  = net.nl[lr].weight.clone()
   uht[uht < 0] = 0
   dist = torch.cdist(uht.data, uht.data)
   dist = torch.tanh(0.3*dist)
   onet = torch.ones(len0)
   shar = onet-torch.mean(dist,1)
   #ubo  = shar.max()
   #lbo  = shar.min()
   #shur = (shar-lbo)/(ubo-lbo)
   board[6] = (1+1*shar)

   if (lr <= 3):
      u1 = net.nl[lr+1].weight
      at = z.grold[lr]
      board[0] = abs(at).mean(1) /abs(at).mean()
      at = u1.data.transpose(1,0)
      board[5] = abs(at).mean(1) /abs(at).mean()
   exec(closed)
   #cr = board[0].sum()

   board[torch.isnan(board)] = 0
   return board

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Gradconc(lr,net,opt):
   u = net.nl[lr].weight; pr = lr-1
   rndat = torch.rand(u.grad.size())
   rndat = rndat.to(z.device)
   #if lr == 1: arnd = 1
   orgrad = u.grad.clone()
   cos1   = abs(u.grad).sum(1)
   ubou   = abs(u.grad).max(1)[0]
   u.grad/= Unsquand(ubou,u.grad,-1)

   # Achtung: conc rules_beg
   # orness avoidance is guaranteed by *= abs(u.data)
   # avoid that abs(u.grad) high and abs(u.data) low
   # but only if of = sign
   #ad = torch.mean(concod, 1)
   #ad = Unsquand (ad,concod,-1)
   if lr == 1:
      concoe = abs(u.data)** 1.00
      concoe*= abs( rndat)** 0.00
      concoe*= abs(u.grad)** 0.00
   exec(closed)
   if lr >= 2:
      concoe = abs(u.data)**(1.05-lr*0.05)
      concoe*= abs( rndat)**(0.00-lr*0.00)
      concoe*= abs(u.grad)**(0.00-lr*0.00)
   exec(closed)
   # grad intrinsic conc_end
   #xx = z03.Topbot(abs(u.grad),50,True)[0]
   #xc = torch.min(xx, 1)[0]
   #xc = Unsquand(xc,u.grad,-1)
   #u.grad [abs(u.grad) < xc] = 0
   # grad intrinsic conc_end
   u.grad*= concoe
   # Achtung: conc rules_end

   cos2   = abs(u.grad).sum(1)
   cosf   = cos1 / cos2
   u.grad*= Unsquand(cosf,u.grad,-1)
   br = 0 #br = random.randrange(20)/100
   u.grad = orgrad*br + u.grad*(1-br)
   u.grad [torch.isnan(u.grad)] = 0

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Uhtnormal(whts, lr):
   for ii in range(2):
      xuht = whts.clone()
      sind = whts * ((-1) ** ii)
      xuht[sind < 0] = 0
      xsu = torch.sum(abs(xuht), dim=1)
      if ii == 0: xs0 = xsu
   exec(closed)
   oar = torch.ones_like(xsu)
   npr = torch.where(xs0 != 0.0, xsu/xs0, oar)
   coe = torch.where(npr >= 0.8, 0.8/npr, oar)
   coe = coe.unsqueeze_(-1)
   xuht *= coe.expand_as(whts)
   whts[sind > 0] = xuht[sind > 0]
   # L1 normalisation
   xuht = whts.clone()
   if SBOL == 0: sind = whts
   if SBOL ==-1: sind = abs(whts)
   xuht[sind < 0] = 0
   xsu = torch.sum (abs(xuht), dim=1)
   coe = abs(BUDL/xsu)
   coe = coe.unsqueeze_(-1)
   whts *= coe.expand_as(whts)

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# class
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Net(nn.Module):

   def __init__(self, x):
      super(Net, self).__init__()   ; z.lrsize[0]=784
      self.fc1 = nn.Linear(784, 512); z.lrsize[1]=512
      self.fc2 = nn.Linear(512, 384); z.lrsize[2]=384
      self.fc3 = nn.Linear(384, 256); z.lrsize[3]=256
      self.fc4 = nn.Linear(256,  10); z.lrsize[4]= 10

      self.nx = np.empty(5, dtype=np.object)
      self.ns = np.empty(5, dtype=np.object)
      self.nl = np.empty(5, dtype=np.object)
      self.nl[1] = self.fc1
      self.nl[2] = self.fc2
      self.nl[3] = self.fc3
      self.nl[4] = self.fc4
      self.forward(x)

   def forward(self, x):
      self.nx[0] = x.view(-1, 784)
      self.nx[0] = Funkoux(self.nx[0],0,self)
      for lr in range(1, 5):
         bias = self.nl[lr].bias.data
         whts = self.nl[lr].weight.data
         x = self.nx[lr-1]
         #z03.Histodraw(x,lr)
         x = Funksin (x)
         #z03.Histodraw(x,lr)
         x  = self.nl[lr](x)
         output = x
         x = x - bias
         self.nx[lr] = Funkoux (x/BUDL,lr,self)
         self.ns[lr] = Funkous (x/BUDL,lr,self)
      exec(closed)
      return output

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Auxilo_Scrogen(net,sot,sit):
   # scrambled /creates sit
   for lr in range(0, 5):
      for bi in range(0,z.butchsze):
         for ri in range(int(z.lrsize[lr]/10)):
            idx = torch.randperm(int(z.lrsize[lr]))[:10]
            bn = random.randrange(z.butchsze)
            sit.nx[lr][bi][idx] = copy.deepcopy \
               (net.nx[lr][bn][idx].detach())
         exec(closed)
      exec(closed)
   exec(closed)

   # scrambled /updates sot
   sot.nx[0] = sit.nx[0]
   sot.nx[0] = Funkoux (sot.nx[0],0,net)
   for lr in range(1, 5):
      bias = net.nl[lr].bias.data
      x = sit.nx[lr-1]
      y = sot.nx[lr-1]
      #if lr != 1: x = y
      x = Funksin(x)
      x = net.nl[lr](x)
      bias = bias.expand_as(x)
      x = x - bias
      sot.nx[lr] = Funkoux (x/BUDL,lr,net)
      sot.ns[lr] = Funkous (x/BUDL,lr,net)
   exec(closed)

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def PruneUeights(net):
   lastlr = 3
   sn = z.nord[1][16][0]
   if 1 == 1 and sn >= 1: # prune copies
      for lr in range(1, lastlr):
         nord5 = z.nord[lr][5][:z.lrsize[lr]]/sn
         nord1 = z.nord[lr][1][:z.lrsize[lr]]/sn
         u = net.nl[lr].weight
         puht = u.data.clone()
         puht [puht < 0] = 0
         dist = torch.cdist(puht, puht)
         dist = torch.tanh(0.3*dist)
         copc = torch.ones(z.lrsize[lr])
         dust = torch.ones(z.lrsize[lr])
         for no in range(z.lrsize[lr]):
            distno = dist[no]
            distno [nord1 < nord1[no]] = 1
            distno [no] = 1 # excludes itself
            copc[no] = torch.sort(distno)[1][0]
            dust[no] = torch.sort(distno)[0][0]
         exec(closed)
         at = nord5 #*dust
         htul = torch.sort(at)[1]
         for hu in range(5):
            nu = htul[hu]; #cu = copc[nu]
            zn = z.lrsize[lr-1]
            u.data[nu] = torch.rand((zn,))
            u.data[nu] =(u.data[nu]-0.5)/2
         exec(closed)
      exec(closed)
   exec(closed)

