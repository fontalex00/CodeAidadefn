
from   z02aidasource import *
from   z03aidasource import *
from   z04aidasource import *
from   z98aidaglobar import *
import z99aidaglobas as z

# inits vars
time0 = time.clock(); dtime = 0

if __name__ == "__main__":

   print(sys.version)
   root = '../data'
   print(root)
   print(" CUDA Available: ", torch.cuda.is_available())
   if (torch.cuda.is_available()):
      z.device = torch.device("cuda")
   z.device = torch.device("cpu")
   epochbase = 0
   if os.path.isfile(CONSFILE):
      #os.remove(CONSFILE)
      consfn = open(CONSFILE, "r")
      inhalt = consfn.read()[:18]
      epochbase = [int(s) for s
         in inhalt.split() if s.isdigit()][0]
      consfn.close()
   exec(closed)

   # Test neu functions
   aa = torch.randn(3, 4); print(" ", aa)
   bb = torch.argsort(aa, dim=1); print(" ", bb)
   #cc = Ofunction(torch.tensor([-0.42]),1)
   #dd = Ofunction(torch.tensor([ 0.42]),1)
   xr = torch.tensor(0.7); print(z.ubias)
   #quit("Terminated by Alex")
   # Test neu functions_End

   if SBOL ==-1.0: ar =  0.50; br = 0.50
   if SBOL == 0.0: ar =  0.00; br = 1.00
   if SBOL == 0.2: ar = -0.25; br = 1.25
   transform = transforms.Compose([
      #transforms.ToTensor()])
      transforms.ToTensor(),
      transforms.Normalize((ar,), (br,))])

   an = BUTCHSZA
   trainset = torchvision.datasets.MNIST(root=DATAPATH,
      train=True, download=True, transform=transform)
   trainloader = torch.utils.data.DataLoader(trainset,
      batch_size= an,shuffle=True, num_workers=2)
   checkloader = torch.utils.data.DataLoader(trainset,
      batch_size=400,shuffle=True, num_workers=2)

   testoset = torchvision.datasets.MNIST(root=DATAPATH,
      train=False, download=True, transform=transform)
   testoloader = torch.utils.data.DataLoader(testoset,
      batch_size=1, shuffle=True)

   inputs = next(iter(trainloader))[0]
   net = Net(inputs); print(" ",net)
   inputs = next(iter(checkloader))[0]
   sit = Net(inputs); sot = Net(inputs)
   for lr in range(5):
      sit.nx[lr] = sit.nx[lr].detach()
      sot.nx[lr] = sot.nx[lr].detach()
   exec(closed)
   #inputs2 = Funkout(net,inputs,0)
   #inputs3 = torch.cat((inputs, inputs2))
   #imshow(torchvision.utils.make_grid(inputs3))

   if 1 == 1:
      for lr in (0,-1,-2,-3,-4):
         x = np.linspace(-1.0,1.0,100)
         xt = torch.Tensor([x])
         foutb = torch.squeeze(torch.relu(xt))
         fout0 = torch.squeeze(Funkoux(xt,lr,net))
         func2 = torch.squeeze(Funkbot(xt,0.60,20,28))
         func3 = torch.squeeze(Funkasd(0*0.50+xt,0.80,20,28))
         func4 = torch.squeeze(Funkasd(2*0.50-xt,0.80,20,28))
         # figures_beg
         #fout0 = torch.squeeze(Funkoux(xt,-2,net))
         #func4 = torch.squeeze(Funkoux(xt,-4,net))
         #fout0 = torch.squeeze(Bgaussian(xt, 0.0,0.3))*0.8
         #func4 = torch.squeeze(Bgaussian(xt,-0.7,0.2))*1.0
         #func4+= torch.squeeze(Bgaussian(xt, 0.7,0.2))*0.6
         # figures_end
         fig = plt.figure()
         ax = fig.add_subplot(111)
         plt.axhline (0,color='black')
         plt.axvline (0,color='black')
         plt.rc ('xtick',labelsize=15)
         plt.rc ('ytick',labelsize=15)
         plt.xticks (np.arange(-1.0,1.1,0.5))
         plt.yticks (np.arange(-1.0,1.1,0.2))
         plt.tight_layout(pad=0)
         plt.subplots_adjust(left=0.1, right=0.9)
         ax.plot(x,foutb.numpy(),'#000000',linewidth=3)
         ax.plot(x,fout0.numpy(),'#8b0000',linewidth=3)
         ax.plot(x,func2.numpy(),'#008b00',linewidth=3)
         ax.plot(x,func3.numpy(),'#00008b',linewidth=3)
         ax.plot(x,func4.numpy(),'#00008b',linewidth=3)
         ax.set_aspect(aspect=0.80) # 1.30 for figures
         fig.canvas.manager.window.move (80,70)
         plt.show(block=False)
         plt.pause(4); plt.close()
      exec(closed)
   exec(closed)

   random.seed(0); np.random.seed(0); torch.manual_seed(0)
   if os.path.isfile(TNETFILE): net=torch.load(TNETFILE)
   #criterion = nn.NLLLoss()
   criterion = nn.CrossEntropyLoss()
   criterion2= nn.L1Loss()
   #criterion2= nn.MSELoss()
   optimizer = optim.SGD (net.parameters(),
      lr=BLRATE,momentum=0.9)

   for z.epoch in range(epochbase, EPOCHS):
      for z.butch, data in enumerate(trainloader, 0):
         butchblc = 60000/(BUTCHSZA*5)
         z.butchsze = BUTCHSZA
         if z.butch % 20 == 0:
            astr = ' epoch {} butch {}'
            print(astr.format(z.epoch, z.butch))
         exec(closed)
         if z.butch % butchblc == 0:
            data = next(iter(checkloader))
            z.butchsze = 400
         exec(closed)

         # Forward propagation
         ubias = torch.tensor(UBIAS)
         lbias = torch.tensor(LBIAS)
         for lr in range(1, 5):
            b = net.nl[lr].bias
            u = net.nl[lr].weight; Uhtnormal(u.data, lr)
            b.data [b.data < lbias[lr]] = lbias[lr]
            b.data [b.data > ubias[lr]] = ubias[lr]
         exec(closed)
         inputs, labels = data
         #net = net.to(z.device)
         #inputs = inputs.to(z.device)
         #labels = labels.to(z.device)
         output = net(inputs); lastlr = net.nx[4]
         softlr = torch.nn.Softmax(output)
         if z.butch % butchblc == 0 and 1 == 1:
            Auxilo_Scrogen(net,sot,sit)
            imagesn = net.nx[0].view(z.butchsze,1,28,28)
            imagess = sit.nx[0].view(z.butchsze,1,28,28)
            #imshow(torchvision.utils.make_grid(
              #torch.cat((imagesn, imagess), 0)))
         exec(closed)

         # Backward propagation and saturation loss
         optimizer.zero_grad()
         loss = criterion(output, labels)
         loss.backward(retain_graph=True)
         for lr in range(1, 5):
            z.grold[lr] = net.nl[lr].weight.grad.clone()
         exec(closed)
         optimizer.zero_grad()
         if z.butch % butchblc == 0:
            satloss = torch.Tensor([0])
            en = 5; tscoe = 0
            for lr in range(1, 5):
               at = net.ns[lr].transpose(1,0)
               bt = sot.ns[lr].transpose(1,0)
               sze0 = list(at.size())[0]
               board = Boardsatu(at,bt,net,lr)
               z.nord[lr][16] += 1
               z.nord[lr][:16,:z.lrsize[lr]] += board[:16,:]
               sortal = torch.sort(board[1],descending=True)
               board[:] = board[:,sortal[1]]
               nu = int(sze0*0.0)
               # Achtung: satloss calc _Beg
               ct = 1 - (torch.tanh(2.0*board[1])/board[6])
               dt = ct; et = ct * board[0]
               gr = (dt)[nu:sze0].mean()*0.5
               gr+= (et)[nu:sze0].mean()*0.5
               lrcoe = (1/lr)**2.0; tscoe += lrcoe
               if lr < en: satloss = satloss + gr*lrcoe
               # Achtung: satloss calc _End
            exec(closed)
            satloss = (satloss/tscoe)
            loss2 = criterion2(satloss,torch.Tensor([0]))
            loss2.backward(retain_graph=True)
            for lr in range(1, 5):
               u = net.nl[lr].weight
               if lr >= en: u.grad *= 0
               z.grad2[lr] = u.grad.clone()
            exec(closed)
         exec(closed)
         # saturation loss_end
         optimizer.zero_grad()
         loss = criterion(output,labels)
         loss.backward(retain_graph=True)

         # Gradient mix
         for lr in range(1, 5):
            u = net.nl[lr].weight
            z.grad1[lr] = u.grad.clone()
            for ii in range(2):
               at = abs(z.grad1[lr]).mean()
               ar = abs(z.grad1[lr]).mean()
               br = abs(z.grad2[lr]).mean()
               if torch.isnan(br): br = 0
               if ii == 0 and br != 0:
                  z.grad2[lr] *= ar/br * 0.20
               exec(closed)
            exec(closed)
            z.gxmean[lr] = br/(ar+br)
            u.grad = z.grad1[lr].clone()
            u.grad+= z.grad2[lr].clone()
         exec(closed)

         # Total grad adjustement
         for lr in range(1, 5):
            Gradconc(lr,net,2)
            for bu in range(2):
               if bu == 0: u = net.nl[lr].bias
               if bu == 1: u = net.nl[lr].weight
               if lr == 1: gradco1 = abs(u.grad).mean()
               if lr ==lr: gradcoe = abs(u.grad).mean()
               if lr == 4: u.grad *= (0.010/gradco1)
               if lr == 3: u.grad *= (0.010/gradco1)
               if lr == 2: u.grad *= (0.010/gradco1)
               if lr == 1: u.grad *= (0.010/gradco1)
               if bu == 0: u.grad *= 0.5 # weaker for bias
               if abs(u.grad).mean() >= 0.4:
                  u.grad *= (40*gradco1/gradcoe)
               z.gdmean[lr] = abs(u.grad).mean()
            exec(closed)
         exec(closed)
         optimizer.step()

         if z.butch == 0:
            for lr in range(4,0,-1):
               oldar = net.ns[lr].transpose(1,0).detach()
               oldas = sot.ns[lr].transpose(1,0).detach()
               Histoplot(oldar,lr,1)
               Histoplot(oldas,lr,2)
            exec(closed)
         exec(closed)
         if z.butch % butchblc == 0:
            DsamplPindo()
            DsamplInsel(net)
         exec(closed)
         if z.butch == BUTCHEND or z.butch+z.epoch == 0:
            PrintoBoard()
            PrintoUhtval(net,sot)
            PrintoUhtshp(net,sot)
            PruneUeights(net)
            output = net(inputs)
         exec(closed)

         z.dloss += loss .item()
         z.dloss2+= loss2.item(); output = output.float()
         bestind = torch.max(output, 1).indices
         bestval = torch.max(output, 1).values
         z.daccur+=torch.mean((bestind == labels).float())

         if z.butch == BUTCHEND or (z.epoch == epochbase and z.butch == 0):
            dtime = (time.perf_counter() - time0)/60
            Printo_Consoleb (dtime,lastlr,output)
            # reinits vars
            z.dloss = 0; z.dloss2 = 0; z.daccur = 0
            z.insel = np.zeros_like(z.insel)
            z.dpindo= np.zeros_like(z.dpindo)
            z.nord = torch.zeros_like(z.nord)
            time0 = time.perf_counter(); dtime = 0
            torch.save(net, TNETFILE)
            if z.epoch % 20 == 0:
               filepath = CNETPATH + 'trainednet_'
               filepath+= str(z.epoch) + '.pth'
               torch.save(net,filepath)
            exec(closed)
         exec(closed)
      gc.collect() # end epoch
   print('Finished Training'); net.eval()

   if 1 == 1:
      Atesto_Epsilon(net, "cpu", testoloader)
   exec(closed)
