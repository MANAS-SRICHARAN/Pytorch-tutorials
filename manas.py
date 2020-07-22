#%%

#%%
print("hello")

# %%
#am gonna do my job to do the mushrooms dataset leanring

# %%
#%%
# First, we'll need to load up a dataset. Pandas is a great
# tool to use to load csv data you may find, which we
# will later turn into tensors. 
# Let's start with the Dataset

#%%

import torch 
import pandas as pd
import numpy
from torch.utils.data import Dataset,DataLoader

class mushrooms(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv("C:\\Users\\user\\Desktop\learning\\PyTorch-Deep-Learning-in-7-Days-master\\PyTorch-Deep-Learning-in-7-Days-master\\mushrooms.csv")

    def __len__(self):
        return(len(self.data))

    def __getitem__(self,idx):
        return(self.data.iloc[idx][0:1])

    def __length__(self):
        return(len(self.data))
#%%
sh = mushrooms()

# %%
# when takingDataset class as inheritence, it actually gives out some pretty formatted output


# %%
sh.__len__()

# %%
sh.__getitem__(23)

# %%
len(sh)

# %%
# __getitem__() can also be used directly by using indexing onthe class instance
# __len__() can be used directly by using len(sh)..on the class instance

# %%
length(sh)
# we get an error not defined...why is that __methidname__()...doesnt mean anything..these are just magical emthods ..no internal meanign


# %%
#https://dbader.org/blog/meaning-of-underscores-in-python

# %%

# now we want the class to out put input and output as tuples
class MushroomDataset(Dataset):

    def __init__(self):
        '''Load up the data.
        '''
        self.data = pd.read_csv("C:\\Users\\user\\Desktop\learning\\PyTorch-Deep-Learning-in-7-Days-master\\PyTorch-Deep-Learning-in-7-Days-master\\mushrooms.csv")

    def __len__(self):
        '''How much data do we have?
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''Grab one data sample
        
        Arguments:
            idx {int, tensor} -- data at this position.
        '''
        # handle being passed a tensor as an index
        if type(idx) is torch.Tensor:
            idx = idx.item()
        return self.data.iloc[idx][1:], self.data.iloc[idx][0:1]

# %%
shh = MushroomDataset()
print(shh[22])

# %%
shrooms = MushroomDataset()
number_for_testing = int(len(shrooms) * 0.05)
number_for_training = len(shrooms) - number_for_testing


# %%
train, test = torch.utils.data.random_split(shrooms,
    [number_for_training, number_for_testing])
len(test), len(train)


# %%
test[0]

# %%


#LETS LEARN ABOUT THE ONE HOT ENCODING

# %%
one_hots = torch.eye(3, 3)
one_hots

#%%
ordinals = {c: i for i, c in enumerate(['A', 'B', 'C'])}
ordinals

# %%
ordinals

# %%
# lets say i need one hot encoding of the letter "A"
one_hots[ordinals["A"]]

# %%
class OneHotEncoder():
    def __init__(self, series):
        '''Given a single pandas series, creaet an encoder
        that can turn values from that series into a one hot
        pytorch tensor.
        
        Arguments:
            series {pandas.Series} -- encode this
        '''
        unique_values = series.unique()
        self.ordinals = {
            val: i for i, val in enumerate(unique_values)
            }
        self.encoder = torch.eye(
            len(unique_values), len(unique_values)
            )

    def __getitem__(self, value):
        '''Turn a value into a tensor
        
        Arguments:
            value {} -- Value to encode, 
            anything that can be hashed but most likely a string
        
        Returns:
            [torch.Tensor] -- a one dimensional tensor
        '''

        return self.encoder[self.ordinals[value]]

# %%

# creating a class to load the dataset and then get the required encoders from that dtaset
class CategoricalCSV(Dataset):
    def __init__(self, datafile, output_series_name):
        '''Load the dataset and create needed encoders for
        each series.
        
        Arguments:
            datafile {string} -- path to data file
            output_series_name {string} -- series/column name
        '''
        self.dataset = pandas.read_csv(datafile)
        self.output_series_name = output_series_name
        self.encoders = {}
        for series_name, series in self.dataset.items():
            # create a per series encoder
            self.encoders[series_name] = OneHotEncoder(series)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        '''Return an (input, output) tensor tuple
        with all categories one hot encoded.
        
        Arguments:
            index {[type]} -- [description]
        '''
        if type(index) is torch.Tensor:
            index = index.item()
        sample = self.dataset.iloc[index]
        output = self.encoders[self.output_series_name][
            sample[self.output_series_name]
        ]
        input_components = []
        for name, value in sample.items():
            if name != self.output_series_name:
                input_components.append(
                    self.encoders[name][value]
                )
        input = torch.cat(input_components)
        return input, output
     

        
#%%

shrooms = CategoricalCSV('C:\\Users\\user\\Desktop\learning\\PyTorch-Deep-Learning-in-7-Days-master\\PyTorch-Deep-Learning-in-7-Days-master\\mushrooms.csv', 'class')

#%%
#print(shrooms[0])
shrooms.__getitem__(0)
        

# %%

# this is just for understanidnng abive code.. skip this 

data =pd.DataFrame({"name":np.array([1,2,3,4,5]),"sex":np.array([6,7,8,9,10])})

data.items()# this will returna n geenrator object
for series_name, series in data.items():
    print(series_name) # will out put name and sex
    print(series) # will return series of that columns

# %%
trail_val =data.iloc[0]
trail_val
trail_val["sex"]

# %%
vec =OneHotEncoder(trail_val)

# %%
vec

# %%
vec[trail_val]

# %%


# building the network
class Model(torch.nn.Module):

    def __init__(self, input_dimensions, 
        output_dimensions, size=128):
        '''
        The constructor is the place to set up each of the layers
        and activations.
        '''
        super().__init__()
        self.layer_one = torch.nn.Linear(input_dimensions, size)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size, size)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size, 
            output_dimensions)

    def forward(self, inputs):

        buffer = self.layer_one(inputs)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = self.shape_outputs(buffer)
        return torch.nn.functional.softmax(buffer, dim=-1)


# %%
model = Model(shrooms[0][0].shape[0], shrooms[0][1].shape[0])
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.BCELoss()


#%%
import sklearn
from sklearn.metrics import accuracy_score
number_for_testing = int(len(shrooms) * 0.05)
number_for_training = len(shrooms) - number_for_testing
train, test = torch.utils.data.random_split(shrooms,
    [number_for_training, number_for_testing])
training = torch.utils.data.DataLoader(train, 
    batch_size=16, shuffle=True)
for epoch in range(2):
    for inputs, outputs in training:
        optimizer.zero_grad()
        results = model(inputs)
        loss = loss_function(results, outputs)
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(loss))

testing = torch.utils.data.DataLoader(test, 
    batch_size=len(test), shuffle=False)
for inputs, outputs in testing:
    results = model(inputs).argmax(dim=1).numpy()
    print(results)
    actual = outputs.argmax(dim=1).numpy()
    accuracy = accuracy_score(actual, results)
    print("\n****\n")
    print(accuracy)

#%%
# now let's run a training loop, we'll go through the dataset
# multiple times -- a loop through the dataset is conventionally
# called an epoch, inside of each epoch, 
# we'll go through each batch
# %%
#In mathematics, the softmax function, also known as softargmax or normalized exponential function, is a function that takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers.


# %%
import torch
import pandas

# %%
class OneHotEncoder():
    def __init__(self, series):
        '''Given a single pandas series, create an encoder
        that can turn values from that series into a one hot
        pytorch tensor.
        
        Arguments:
            series {pandas.Series} -- encode this
        '''
        unique_values = series.unique()
        self.ordinals = {
            val: i for i, val in enumerate(unique_values)}
        self.encoder = torch.eye(
            len(unique_values), len(unique_values))

    def __getitem__(self, value):
        '''Turn a value into a tensor
        
        Arguments:
            value {} -- Value to encode
            but most likely a string
        
        Returns:
            [torch.Tensor] -- a one dimensional tensor 
        '''

        return self.encoder[self.ordinals[value]]


# %%
look = pandas.read_csv('C:\\Users\\user\\Desktop\\learning\\PyTorch-Deep-Learning-in-7-Days-master\\PyTorch-Deep-Learning-in-7-Days-master\\kc_house_data.csv')
look.iloc[0]
# %%

# we would need the date encoder for this 


#setting all the categorical values

categorical = [
    'waterfront',
    'view',
    'condition',
    'grade',
]
#%%
# discarding the id's
discard = [
    'id'
]

# %%
import dateutil

# %%
# all dates taken seperately
dates = ['date']

# %%
class DateEncoder():
    def __getitem__(self, datestring):
        '''Encode into a tensor [year, month, date]
        given an input date string.
        
        Arguments:
            datestring {string} -- date string, ISO format
        '''
        parsed = dateutil.parser.parse(datestring)
        return torch.Tensor(
            [parsed.year, parsed.month, parsed.day])


# %%
DateEncoder()['20141013T000000']

# %%
from torch.utils.data import Dataset,DataLoader

# %%
class MixedCSV(Dataset):
    def __init__(self, datafile, output_series_name,
        date_series_names, categorical_series_names,
        ignore_series_names):
        self.dataset = pandas.read_csv(datafile)
        self.output_series_name = output_series_name
        self.encoders = {}

        for series_name in date_series_names:
            self.encoders[series_name] = DateEncoder()
        for series_name in categorical_series_names:
            self.encoders[series_name] = OneHotEncoder(
                self.dataset[series_name]
            )
        self.ignore = ignore_series_names

    def __len__(self):
            return len(self.dataset)
        

    def __getitem__(self, index):
        '''Return an (input, output) tensor tuple
        with all categories one hot encoded.
        
        Arguments:
            index {[type]} -- [description]
        '''
        if type(index) is torch.Tensor:
            index = index.item()
        sample = self.dataset.iloc[index]

        output = torch.Tensor([sample[self.output_series_name]]) 

        input_components = []
        for name, value in sample.items():
            if name in self.ignore:
                continue
            elif name in self.encoders:
                input_components.append(
                    self.encoders[name][value]
                )
            else:
                input_components.append(torch.Tensor([value]))
        input = torch.cat(input_components)
        return input, output





# %%
houses = MixedCSV('C:\\Users\\user\\Desktop\\learning\\PyTorch-Deep-Learning-in-7-Days-master\\PyTorch-Deep-Learning-in-7-Days-master\\kc_house_data.csv',
    'price',
    dates,
    categorical,
    discard
    )
houses[0]

# %%
class Model(torch.nn.Module):

    def __init__(self, input_dimensions, size=128):
        '''
        The constructor is the place to set up each layer
        and activations.
        '''
        super().__init__()
        self.layer_one = torch.nn.Linear(input_dimensions, size)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size, size)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size, 1)

    def forward(self, inputs):

        buffer = self.layer_one(inputs)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = self.shape_outputs(buffer)
        return buffer

model = Model(houses[0][0].shape[0])
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.MSELoss()

# %%
number_for_testing = int(len(houses) * 0.05)
number_for_training = len(houses) - number_for_testing
train, test = torch.utils.data.random_split(houses,
    [number_for_training, number_for_testing])
training = torch.utils.data.DataLoader(
    train, batch_size=64, shuffle=True)
for epoch in range(3):
    for inputs, outputs in training:
        optimizer.zero_grad()
        results = model(inputs)
        loss = loss_function(results, outputs)
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(loss))

# %%
actual = test[0][1]
predicted = model(test[0][0])
actual, predicted
#%%

import sklearn.metrics
import torch.utils.data

testing = torch.utils.data.DataLoader(
    test, batch_size=len(test), shuffle=False)
for inputs, outputs in testing:
    predicted = model(inputs).detach().numpy()
    actual = outputs.numpy()
    print(sklearn.metrics.r2_score(actual, predicted))

# %%
