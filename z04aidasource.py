
from   z98aidaglobar import *
import z99aidaglobas as z

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Atesto_Epsilon(model, device, testoloader):

   accuracies = []
   examples = []
   epsilons = [0.00, 0.30]
   #epsilons= [0.00, 0.15, 0.30]

   nou = datetime.datetime.now()
   print(nou.strftime("cur time: %Y-%m-%d %H:%M:%S"))
   # Run test for each epsilon
   for eps in epsilons:
      acc, ex = test(model, device, testoloader, eps)
      accuracies.append(acc)
      examples.append(ex)
   exec(closed)
   nou = datetime.datetime.now()
   print(nou.strftime("cur time: %Y-%m-%d %H:%M:%S"))

   plt.figure(figsize=(5, 5))
   plt.plot(epsilons, accuracies, "*-")
   plt.yticks(np.arange(0, 1.1, step=0.1))
   plt.xticks(np.arange(0, .35, step=0.05))
   plt.title("Accuracy vs Epsilon")
   plt.xlabel("Epsilon")
   plt.ylabel("Accuracy")
   plt.show()

   # Plot several examples of adversarial samples
   cnt = 0
   plt.figure(figsize=(8, 10))
   for i in range(len(epsilons)):
      for j in range(len(examples[i])):
         cnt += 1
         plt.subplot(len(epsilons), len(examples[0]), cnt)
         plt.xticks([], [])
         plt.yticks([], [])
         if j == 0: plt.ylabel("Eps: {}".
            format(epsilons[i]), fontsize=20)
         orig, adv, ex = examples[i][j]
         plt.title("{} -> {}".format(orig, adv),fontsize=20)
         plt.imshow(ex, cmap="gray")
      exec(closed)
   exec(closed)
   plt.tight_layout()
   plt.show()

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Fgsm_attack(image, epsilon, data_grad):

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(
       perturbed_image, SBOL, 1)
    # Return the perturbed image
    return perturbed_image

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Pgd_attack2(model, bild, labels, eps, alpha, iters):

   device = torch.device("cpu")
   images = images.to(device)
   labels = labels.to(device)
   loss = nn.CrossEntropyLoss()
   bildor = bild.data

   for ti in range(iters):
      bild.requires_grad = True
      outputs, lastlr = model(bild)
      model.zero_grad()
      cost = loss(outputs, labels).to(device)
      cost.backward()
      bildad= bild + alpha * bild.grad.sign()
      eta   = torch.clamp(bildad-bildor, -eps, eps)
      bild  = torch.clamp(bildor+eta, SBOL, 1)
      bild  = bild.detach_()
   exec(closed)

   return images

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def test(model, device, test_loader, epsilon):

   # Accuracy counter
   correct = 0
   adv_examples = []

   '''auto_attacker = AutoAttack(
      model,
      norm='Linf',
      eps=8./255.,
      log_path='./autoattack_log.txt',
      version='standard',
      device='cpu'
   )'''

   # Loop over all examples in test set
   testnr = 0
   for data, target in test_loader:
      testnr += 1
      if (testnr%1000 == 0):
         sys.stdout.write(' nr {}'.format(testnr))
         sys.stdout.flush()
      exec(closed)

      # Send the data and label to the device
      data, target = data.to(device), target.to(device)
      # Set requires_grad attribute of tensor
      data.requires_grad = True

      # Forward pass the data through the model
      output = model(data)

      # get the index of the max log-probability
      init_pred = output.max(1, keepdim=True)[1]

      # If the initial prediction is wrong,
      # dont bother attacking, just move on
      # va solo se batch_size = 1
      if init_pred.item() != target.item():
         continue
      exec(closed)

      # Calculate the loss
      loss = F.nll_loss(output, target)
      # Zero all existing gradients
      model.zero_grad()
      # Calculate gradients of model in backward pass
      loss.backward()
      # Collect datagrad
      data_grad = data.grad.data

      # Call Attacks
      perturbed_data = Fgsm_attack(data, epsilon, data_grad)
      #perturbed_data =
      #  Pgd_attack2(model, data, target, epsilon, 0.03, 20)
      #perturbed_data =
      #  auto_attacker.run_standard_evaluation(data, target)
      # Re-classify the perturbed image
      output = model(perturbed_data)

      # Check for success
      # get the index of the max log-probability
      final_pred = output.max(1, keepdim=True)[1]
      ipi =  init_pred.item()
      fpi = final_pred.item()
      if fpi == target.item():
         correct += 1
         # Special case for saving 0 epsilon examples
         if (epsilon == 0) and (len(adv_examples) < 5):
            adv_ex = perturbed_data.squeeze()
            adv_ex = adv_ex.detach().cpu().numpy()
            adv_examples.append((ipi, fpi, adv_ex))
         exec(closed)
      else:
         # Save some adv examples for visualization later
         if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze()
            adv_ex = adv_ex.detach().cpu().numpy()
            adv_examples.append((ipi, fpi, adv_ex))
         exec(closed)
      exec(closed)

   # Calculate final accuracy for this epsilon
   sys.stdout.write("\n")
   final_acc = correct/float(len(test_loader))
   print("Epsilon: {:.2f}\tTest Accuracy = {:4.0f} / {} = {}".
      format(epsilon, correct, len(test_loader), final_acc))
   exec(closed)

   # Return the accuracy and an adversarial example
   return final_acc, adv_examples




