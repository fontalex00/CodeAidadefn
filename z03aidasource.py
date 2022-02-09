
import z02aidasource as z02
from   z98aidaglobar import *
import z99aidaglobas as z

def Bildschau(bild):
   bild = bild / 2 + 0.5  # unnormalize
   npbild = bild.numpy()
   plt.imshow(np.transpose(npbild, (1,2,0)))
   plt.show()

def Bgaussian(x, mu, si):
   return np.exp(-np.power(x - mu, 2.) / (2 * np.power(si, 2.)))

def Bsinvoid(x):
   x [x <= 0.01] = 0.01
   x [x >= 0.99] = 0.99
   z = -torch.log(1/x-1)*0.1+0.5
   return z

def Histoplot(actor_,lr,opn):
   actor = actor_.clone().detach()
   #sze0 = list(actor.size())[0]
   cbins = [-1.0]
   for ii in range(1,41): cbins.append(-1.0+ii*0.05)
   htar = torch.sort(-abs(z.nord[lr][1]))[1]
   for ti in range(1):
      an = min(20, z.lrsize[lr])
      for ho in range(an):
         if ti == 0: no = htar[ho]
         if ti == 1: no = htar[z.lrsize[lr]-1-ho]
         if no < z.lrsize[lr]:
            cn = np.histogram(actor[no], bins=cbins)[0]
            if no == 109 or no == 190: exec(closed)
            astr = 'Outdir/0' +str(lr)
            if opn == 1: bstr = astr +'besto_histor.png'
            if opn == 1: wstr = astr +'worst_histor.png'
            if opn == 2: bstr = astr +'besto_histos.png'
            if opn == 2: wstr = astr +'worst_histos.png'
            plt.hist(cbins[:-1], cbins, weights=cn)
            plt.ylim(None,200)
            if ti == 0: plt.savefig(bstr)
            if ti == 1: plt.savefig(wstr)
         exec(closed)
      exec(closed)
      plt.close()
   exec(closed)

def Topbot(at, an, lopt):
   return torch.topk(at, an, largest=lopt)

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def DsamplInsel(net):

   sn = z.nord[1][16][0]
   insar = np.zeros(8)
   for lr in range(1,5):
      insar= np.zeros_like(insar)
      uht  = net.nl[lr].weight.detach().clone()
      puht = uht.clone(); puht [puht < 0] = 0
      nuht = uht.clone(); nuht [nuht > 0] = 0
      uconc = z02.Whtconcallo(uht,lr)
      losco = z.nord[lr][0][:z.lrsize[lr]]/sn; #losco = 1
      losco = losco.detach().clone() /z.lrsize[lr]
      insar[1] += np.dot(losco, uconc)
      insar[2] += np.dot(losco, abs(puht.sum(1)))
      insar[3] += np.dot(losco, abs(nuht.sum(1)))
      insar[4] += np.dot(losco, net.nl[lr].bias.data)
      for bi in range(10):
         bt = net.nx[lr][bi].detach().clone()
         bt = torch.sigmoid(20*(bt-0.5))
         insar[5] += np.dot(losco, bt)/10
      exec(closed)
      for ii in range(1,7):
         z.insel[lr,ii] += insar[ii]
      z.insel[lr,0] += 1
   exec(closed)

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def DsamplPindo():

   sn = z.nord[1][16][0]
   qx = torch.Tensor(6)
   for lr in range(1,5):
      sze0 = z.lrsize[lr]
      nord = z.nord[lr].clone()
      sortal = torch.sort(nord[1], descending=True)
      for ii in range(0,16): nord[ii] = nord[ii][sortal[1]]
      for ii in range(1,16): nord[ii]*= nord[0]/sn
      for ti in range(2):
         if (ti == 0): nu = int(sze0*0.0)
         if (ti == 1): nu = int(sze0*0.0)
         #if (ti == 1): nu = int(sze0*0.9)
         qx[0] = nord[ 2][nu:sze0].mean()/sn
         qx[1] = nord[10][nu:sze0].mean()/sn
         qx[2] = nord[ 3][nu:sze0].mean()/sn
         qx[3] = nord[11][nu:sze0].mean()/sn
         qx[4] = nord[ 4][nu:sze0].mean()/sn
         qx[5] = nord[12][nu:sze0].mean()/sn
         for ii in range(6):
            z.dpindo[ii+ti*6][lr][0] = 1
            z.dpindo[ii+ti*6][lr][1] = qx[ii]
         exec(closed)
      exec(closed)
   exec(closed)

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def PrintoUhtval(net,sot):

   fn = open(WHTSFILE, "w+")
   fn.write("\n Layer: {:3d}\n".format(0))
   for no in range(z.lrsize[0]):
      if no % 100 == 0: fn.write("\n ner ")
      fn.write("({:3d}){:6.3f} ".format(no,net.nx[0][0][no]))
   exec(closed)

   fn.write("\n")
   for lr in range(1, 5):
      bias = net.nl[lr].bias.data
      u = net.nl[lr].weight
      fn.write("\n Layer: {:3d}".format(lr))
      bt = abs(z.nord[lr][1])
      bt =bt[:z.lrsize[lr]]
      htro = torch.sort(-bt)[1]
      for ho in range(10):
         no = htro[ho]
         totp = u.data[no][u.data[no] > 0].sum()
         totn = u.data[no][u.data[no] < 0].sum()
         fn.write("\n\n nd  ({:3d})".format(no))
         fn.write(" bias{:6.2f}".format(bias[no]))
         fn.write(" totp{:6.2f}"  .format(totp))
         fn.write(" totn{:6.2f}\n".format(totn))
         fn.write("cnar {:5.3f} " .format(net.ns[lr][0][no]))
         fn.write("cnas {:5.3f} " .format(sot.ns[lr][0][no]))
         htar= np.argsort(-abs(u.data[no]))
         #ztar= np.sort(-abs(whts[no]))
         for hi in range(10):
            if hi%5 == 0: fn.write("\n wht ")
            ni = htar[hi]
            if ni < z.lrsize[lr-1]:
               astr = "({:3d}){:6.3f} "
               fn.write(astr.format(ni,u.data[no][ni]))
            exec(closed)
         exec(closed)
      fn.write("\n")
   fn.close()

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def PrintoUhtshp(net,sot):

   sn = z.nord[1][16][0]; #sn = 7
   lr = 1 # lr 1 only
   at = z.nord[1][0]/sn
   bt = z.nord[1][1]/sn
   lsq = torch.sort(-at*(1- 0))[1]
   rsq = torch.sort(-at*(1-bt))[1]
   rsq = rsq [bt[rsq] <= 0.5]
   if len(rsq) == 0: rsq = [0]
   fn = open(WHT1FILE, "w+")
   whts = net.nl[1].weight.data

   for ho in range(int(z.lrsize[1]/2)):
      h1 = min(ho, len(rsq)-1)
      astr = "nd:{:3d}|{:3d}" + ' '*79
      bstr = "xx:{:5.2f}" + ' '*69
      bstr+= "xx:{:5.2f}" + ' '*69 + "xx:{:5.2f}"
      fn.write("\n")
      for ti in range(3):
         for ni in range(2):
            if ni == 0: no = lsq[ho]
            if ni == 1: no = rsq[h1]
            if ti == 0: b1 = z.nord[lr][16][ 0]
            if ti == 0: b2 = z.nord[lr][ 0][no]/sn
            if ti == 0: b3 = z.nord[lr][ 1][no]/sn
            if ti == 1: b1 = z.nord[lr][ 2][no]/sn
            if ti == 1: b2 = z.nord[lr][ 3][no]/sn
            if ti == 1: b3 = z.nord[lr][ 4][no]/sn
            if ti == 2: b1 = z.nord[lr][10][no]/sn
            if ti == 2: b2 = z.nord[lr][11][no]/sn
            if ti == 2: b3 = z.nord[lr][12][no]/sn
            if ni == 0: fn.write(astr.format(ho,no))
            if ni == 1: fn.write(astr.format(h1,no))
            fn.write(bstr.format(b1,b2,b3))
            fn.write(" "*62)
         exec(closed)
         fn.write("\n")
      exec(closed)

      for yi in range(1, 28):
         for xi in range(28):
            pxl = whts[lsq[ho]][yi*28+xi]
            fn.write("{:10.7f} ".format(pxl))
         exec(closed)
         fn.write(" _ _ ")
         for xi in range(28):
            pxl = whts[rsq[h1]][yi*28+xi]
            fn.write("{:10.7f} ".format(pxl))
         exec(closed)
         fn.write("\n")
      exec(closed)

      # histograms _
      oldar = net.ns[1].transpose(1,0).detach()
      oldas = sot.ns[1].transpose(1,0).detach()
      fn.write("\n\n\n\n\n\n\n\n\n")
      cbins = [-1.0]
      for ii in range(1,41): cbins.append(-1.0+ii*0.05)
      cn = np.histogram(oldar[lsq[ho]], bins=cbins)[0]
      dn = np.histogram(oldas[lsq[ho]], bins=cbins)[0]
      for yi in range(23,0,-1):
         fn.write(' '*0)
         for xi in range(12,40):
            if cn[xi] >=yi: rs = 1
            if cn[xi] < yi: rs = 0
            fn.write("{:10.7f} ".format(rs))
         exec(closed)
         fn.write(' '*5)
         for xi in range(12,40):
            if dn[xi] >=yi: rs =-1
            if dn[xi] < yi: rs = 0
            fn.write("{:10.7f} ".format(rs))
         exec(closed)
         fn.write("\n")
      exec(closed)
      fn.write("\n\n\n\n\n")
   exec(closed)
   fn.close()

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def PrintoBoard ():

   boardfn = open(BOARDFN, "w")
   sn = z.nord[1][16][0]; lord = z.nord.clone()
   for lr in range(1, 5):
      sortal = torch.sort(lord[lr][0], descending=True)
      lord[lr][:] = lord[lr][:,sortal[1]]
      for no in range(10):
         boardfn.write("({:2d},{:3d})".format(lr,no))
         for ii in range(6):
            ar = lord[lr][ii][no]/sn
            boardfn.write("{:6.2f}".format(ar))
         exec(closed)
         boardfn.write("\n")
      exec(closed)
   exec(closed)
   boardfn.close()

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Printo_Consoleb (dtime,lastlr,output):

   sn = z.nord[1][16][0]
   content = ""
   if os.path.isfile(CONSFILE):
      consfn = open(CONSFILE, "r")
      content = consfn.read()
      content = content[:20000] # crops
      consfn.close();
      os.remove(CONSFILE)
   consfn = open(CONSFILE, "w")
   an = z.butch+1
   consfn.write("\n Ep: {:3d} {:d}".format(z.epoch, an))
   consfn.write(" dloss: {:5.3f}" .format(z.dloss  /an))
   consfn.write(" dloss2:{:6.3f}" .format(z.dloss2 /an))
   consfn.write(" daccur:{:6.3f}" .format(z.daccur /an))
   consfn.write("\nlast lr: ")
   for ci in range(10):
      consfn.write("{:6.2f}".format(lastlr[0][ci]))
   exec(closed)
   consfn.write("\n output: ")
   for ci in range(10):
      consfn.write("{:6.2f}".format(output[0][ci]))
   exec(closed)

   # insel
   for pi in range(-2,9):
      if pi ==-2: consfn.write("\n gdmean: ")
      if pi ==-1: consfn.write("\n gxmean: ")
      if pi == 1: consfn.write("\n puconc: ")
      if pi == 2: consfn.write("\n  whtsp: ")
      if pi == 3: consfn.write("\n  whtsn: ")
      if pi == 4: consfn.write("\n   bias: ")
      if pi == 5: consfn.write("\n  hones: ")
      if pi == 6: consfn.write("\n  hvals: ")
      if pi == 7: consfn.write("\n  nord1: ")
      if pi == 8: consfn.write("\nalnord1: ")
      for lr in range(1, 5):
         if pi ==-2: ar = z.gdmean[lr]
         if pi ==-1: ar = z.gxmean[lr]
         br = z.insel[lr][pi] /z.insel [lr][0]
         ct = z.nord[lr][1] /sn;
         dr = ct[:z.lrsize[lr]].mean()
         ct*= z.nord[lr][0] /sn
         cr = ct[:z.lrsize[lr]].mean()
         if pi >=-2: astr = " ({:7.5f})"
         if pi >= 1: astr = " ({:7.2f})"
         if pi >= 4: astr = " ({:7.3f})"
         if pi >=-2: printedvar = ar
         if pi >= 1: printedvar = br
         if pi >= 7: printedvar = cr
         if pi >= 8: printedvar = dr
         if pi != 0: consfn.write(astr.format(printedvar))
      exec(closed)
   exec(closed)

   # dpindo
   for pi in range(0,12):
      if pi%6==0: consfn.write("\n")
      consfn.write("\n dpind" + str(pi) + ":")
      if pi <= 9: consfn.write(" ")
      for lr in range(1, 5):
         ar = z.dpindo[pi][lr][0]
         br = z.dpindo[pi][lr][1]
         if ar == 0: cr = 0
         else: cr = br / ar
         astr = " ({:d},{:5.0f})"
         consfn.write(astr.format(lr, cr*1000))
      exec(closed)
      if pi == 0: consfn.write(" xxxxx_o")
      if pi == 3: consfn.write(" xxxxx_o")
      if pi == 6: consfn.write(" xxxxx_o")
      if pi == 9: consfn.write(" xxxxx_o")
   exec(closed)
   # duration
   bstr = " durat (min): {:4.1f} ___\n\n\n\n"
   consfn.write(bstr.format(dtime))
   consfn.write(content)
   consfn.close()

