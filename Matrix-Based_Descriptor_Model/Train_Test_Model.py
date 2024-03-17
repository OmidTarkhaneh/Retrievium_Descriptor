"""""
Train Your Own Neural Network Potential
=======================================

"""
from units import *
from DataLoading import *
from Utils import *
###############################################################################
# To begin with, let's first import the modules and setup devices we will use:

import torch
# import torchani
import os
import math
import torch.utils.tensorboard
import tqdm
import pickle
# from torchani.units import hartree2kcalmol


# helper function to convert energy unit from Hartree to kcal/mol
# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

species_order = ['H', 'C', 'N', 'O','S','Cl']
# species_order= ['H', 'C', 'N', 'O']
num_species = len(species_order)
energy_shifter = EnergyShifter(None)



###############################################################################


try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
# dspath = os.path.join('MyData_212testnew.h5')
# dspath=os.path.join('GDB28OrgCoulomb_TrainSort.h5')
dspath=os.path.join('GDB28Sorted_Train.h5')
batch_size = 256




training, validation = load(dspath)\
                                    .subtract_self_energies(energy_shifter, species_order)\
                                    .remove_outliers()\
                                    .shuffle()\
                                    .split(0.9, None)


training = training.collate(batch_size).cache()
validation = validation.collate(batch_size).cache()
print('Self atomic energies: ', energy_shifter.self_energies)




aev_dim = 29 # the dimension of the structure (each structure has the shape of 29*29)

class TinyModel(torch.nn.Module):

  def __init__(self):
      super(TinyModel, self).__init__()

      self.lin1 = torch.nn.Linear(aev_dim, 256)
      self.CELU1 = torch.nn.ELU(0.3)
      self.lin2 = torch.nn.Linear(256, 256)
      self.CELU2 = torch.nn.ELU(0.3)
      self.lin3 = torch.nn.Linear(256, 128)
      self.CELU3 = torch.nn.ELU(0.3)
      self.lin4 = torch.nn.Linear(128, 96)
      self.CELU4 = torch.nn.ELU(0.3)
      self.lin5 = torch.nn.Linear(224, 224)
      self.CELU5 = torch.nn.ELU(0.3)
      self.lin6 = torch.nn.Linear(224, 224)
      self.CELU6 = torch.nn.ELU(0.3)
      self.lin7 = torch.nn.Linear(224, 64)
      self.CELU7 = torch.nn.ELU(0.3)
      self.lin8 = torch.nn.Linear(64, 1)


  def __getitem__(self, key):
      return self.__dict__[key]

  def forward(self, x):
      x = self.lin1(x)
      x = self.CELU1(x)
      x= self.lin2(x)
      x = self.CELU2(x)
      x = self.lin3(x)
      x4 = self.CELU3(x)
      x = self.lin4(x4)
      x5 = self.CELU4(x)
      x6=torch.cat((x4,x5),dim=-1)
      x = self.lin5(x6)
      x=self.CELU5(x)
      xx=torch.multiply(x6,x)
      x=self.lin6(xx)
      x=self.CELU6(x)
      x=self.lin7(x)
      x=self.CELU7(x)
      x=self.lin8(x)


      return x




C_network = TinyModel()
H_network =TinyModel()
O_network =TinyModel()
N_network =TinyModel()
S_network =TinyModel()
Cl_network =TinyModel()


nn = ProposedModel([C_network, H_network, O_network, N_network,S_network,Cl_network])
print(nn)




###############################################################################
# Initialize the weights and biases.


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)


nn.apply(init_params)

###############################################################################

model = Sequential(nn).to(device)
AdamW = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)




###############################################################################
# Setting up a learning rate scheduler to do learning rate decay
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=50, threshold=0.0000001)

###############################################################################
# Train the model by minimizing the MSE loss, until validation RMSE no longer
# improves during a certain number of steps, decay the learning rate and repeat
# the same process, stop until the learning rate is smaller than a threshold.
#
# We first read the checkpoint files to restart training. We use `latest.pt`
# to store current training state.
latest_checkpoint = 'latest.pt'





# During training, we need to validate on validation set and if validation error
# is better than the best, then save the new best model to a checkpoint


def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    true_energies_1=[]
    predicted_energies_1=[]

    true_dftmain_energy=[]
    predicted_dftmain_energies=[]

    model.train(False)
    with torch.no_grad():
        for properties in validation:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device)
            _, predicted_energies = model((species, coordinates))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]


            # save predicted and true energy in list
            predicted_energies_1.append(predicted_energies.detach().cpu().numpy())
            true_energies_1.append(true_energies.detach().cpu().numpy())

    model.train(True)
    return hartree2kcalmol(math.sqrt(total_mse / count)), predicted_energies_1, true_energies_1


import copy


# Make a copy of the initial model parameters
initial_params = copy.deepcopy(model.state_dict())

##################################################################################
"""# Model Training"""
##################################################################################

# We will also use TensorBoard to visualize our training process
tensorboard = torch.utils.tensorboard.SummaryWriter()

###############################################################################
# Finally, we come to the training loop.
#
# In this tutorial, we are setting the maximum epoch to a very small number,
# only to make this demo terminate fast. For serious training, this should be
# set to a much larger value
mse = torch.nn.MSELoss(reduction='none')


print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 5
early_stopping_learning_rate = 1.0E-6
best_model_checkpoint = 'best.pt'

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):

    learning_rate = AdamW.param_groups[0]['lr']
    rmse, predicted_energies_1, true_energies_1 = validate()
    print('RMSE:', rmse, 'learning_rate:',learning_rate, 'at epoch', AdamW_scheduler.last_epoch + 1)


    if learning_rate < early_stopping_learning_rate:
        break

    # checkpoint
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse)
    # SGD_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):
        species = properties['species'].to(device)
        species=species.to(torch.float32)

        coordinates = properties['coordinates'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        num_atoms = (species > 0).sum(dim=1, dtype=true_energies.dtype)

        _, predicted_energies = model((species, coordinates))

        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

        AdamW.zero_grad()
        # SGD.zero_grad()
        loss.backward()
        AdamW.step()
        # SGD.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        # 'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        # 'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)




import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

true_energies_11= np.hstack( true_energies_1)
pred_energies_11= np.hstack( predicted_energies_1)

mae=np.sum(np.abs(true_energies_11-pred_energies_11))

mae=mae/(len(true_energies_11))

print('overall MAE(kcal/mol)=',(mae))


mse=mean_squared_error(true_energies_11,pred_energies_11)

rmse=np.sqrt(mse)

print('overall RMSE(kcal/mol)=',(rmse))

# device='cpu'
# model=model.to(device)


# flag=True

species_order = ['H', 'C', 'N', 'O','S','Cl']
# species_order= ['H', 'C', 'N', 'O']

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
dspath = os.path.join('GDB28Sorted_Test.h5')
batch_size = 256


training2, validation2 = load(dspath)\
                                    .subtract_self_energies(energy_shifter, species_order)\
                                    .remove_outliers()\
                                    .shuffle()\
                                    .split(0.0000001, None)

training2 = training2.collate(batch_size).cache()
validation2 = validation2.collate(batch_size).cache()
print('Self atomic energies: ', energy_shifter.self_energies)


def validate2():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    true_energies_1=[]
    predicted_energies_1=[]
    true_dft1main_energy=[]
    predicted_dft1main_energies=[]

    model.train(False)
    with torch.no_grad():
        for properties in validation2:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device)
            _, predicted_energies = model((species, coordinates))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]

            energy_shift = energy_shifter.sae(species)
            true_dft1_energy = true_energies + energy_shift.to(device)
            predicted_dft1_energies= predicted_energies + energy_shift.to(device)

            # save predicted and true energy in list
            predicted_energies_1.append(predicted_energies.detach().cpu().numpy())
            true_energies_1.append(true_energies.detach().cpu().numpy())

            true_dft1main_energy.append(true_dft1_energy.detach().cpu().numpy())
            predicted_dft1main_energies.append(predicted_dft1_energies.detach().cpu().numpy())

            # print('true_dft1main_energy=',true_dft1main_energy)
            # print('true_energies=',true_energies)

    model.train(True)
    return hartree2kcalmol(math.sqrt(total_mse / count)), predicted_energies_1, true_energies_1, true_dft1main_energy, predicted_dft1main_energies



rmse_1, predicted_energies_111,true_energies_111, true_dft1_energy, predicted_dft1_energies = validate2()



from sklearn.metrics import mean_squared_error, r2_score
true_energies_22= np.hstack(true_energies_111)
pred_energies_22= np.hstack(predicted_energies_111)

mae=np.sum(np.abs(true_energies_22-pred_energies_22))
mae=mae/(len(true_energies_22))
print('overall MAE(kcal/mol)=',hartree2kcalmol(mae))
mse=mean_squared_error(true_energies_22,pred_energies_22)
rmse=np.sqrt(mse)
print('overall RMSE(kcal/mol)=',hartree2kcalmol(rmse))
print( 'R2score=',r2_score(true_energies_22,pred_energies_22))



true_dft1_energies_11= np.hstack( true_dft1_energy)
pred_dft1_energies_11= np.hstack( predicted_dft1_energies)



mae=np.sum(np.abs(true_dft1_energies_11-pred_dft1_energies_11))
mae=mae/(len(true_dft1_energies_11))
print('overall MAE(kcal/mol)=',(mae))
mse=mean_squared_error(true_dft1_energies_11,pred_dft1_energies_11)
rmse=np.sqrt(mse)
print('overall RMSE(kcal/mol)=',(rmse))


NonZero_NewModeldf=pd.DataFrame()
NonZero_NewModeldf['truelabel']=true_dft1_energies_11
NonZero_NewModeldf['pred']=pred_dft1_energies_11
NonZero_NewModeldf['val']=np.arange(0,len(true_dft1_energies_11))
NonZero_NewModeldf.to_csv('GDB29_Results.csv')