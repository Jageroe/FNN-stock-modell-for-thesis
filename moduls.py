import pandas as pd
import warnings
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None

class DataPipeline(Dataset):

    """
    This class create a complete datapipeline.
    - It reads the given pandas DataFrame
    - It seperates the test and the training datasets based on the given date
    - It drops the unnecessary variables
    - It encodes the target labels
    - It standardize the data and convert it into float32 type tensors


    Args:
        df: the input dataframe which contains both the features and the target variables
        label_col: the name of the column containing the target variables.
        date_col:
        batch_size: it's the size of the mini bacthes of the training data
        test_from: the test dataset will be seperated from this date
        cols_to_drop: If the input df containts any unwanted columns, 
                we can easily drop them by give a
                list that contains those column names


    Attributes:
        scaler
        y_encoder
        train_loader
        test_loader

    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        date_col: str,
        batch_size: int,
        test_from: str,
        cols_to_drop: list = []
    ) -> None:
        
        # Droping the unwanted cols
        for col in cols_to_drop:
            df = df.drop(col, axis=1, inplace= False)

        # Seperate our data into a test and a train subset based on the test_from argument
        test_df = df[pd.to_datetime(df[date_col]) >= test_from]
        train_df = df[pd.to_datetime(df[date_col]) < test_from]

        # Droping the date variable. After splitting the test and train data, 
        # we no longer need this.
        test_df.drop(date_col, axis=1, inplace= True)
        train_df.drop(date_col, axis=1, inplace= True)

        # Printing out some basic statistics
        print(f'The rate of the test data: {round(test_df.shape[0] / train_df.shape[0] * 100,2)} %')
        print(f'The size of the training dataset: {train_df.shape[0]}')
        print(f'The size of the test dataset: {test_df.shape[0]}')

        # Create a Label encoder object and fit it on the target label
        self.y_encoder = LabelEncoder()
        self.y_encoder.fit(df[label_col].values)

        # Seperate the target labels from the datasets, and encode the target labels
        X_train = train_df.loc[:, train_df.columns != label_col].values
        y_train = self.y_encoder.transform(train_df[label_col].values)

        X_test = test_df.loc[:, test_df.columns != label_col].values
        y_test = self.y_encoder.transform(test_df[label_col].values)

        # Create a new attribute to store the number of features
        self.feature_num = X_train.shape[1]

        # Feature scaling
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)

        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Converting our data to tensors
        X_train = torch.tensor(X_train.astype('float32'))
        y_train = torch.tensor(y_train.astype('float32'))

        X_test = torch.tensor(X_test.astype('float32'))
        y_test = torch.tensor(y_test.astype('float32'))

        # Create Dataset and the DataLoader objects
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        self.train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
            shuffle = True
        )

        self.test_loader = DataLoader(
            dataset = test_dataset,
            batch_size = len(test_dataset),
            shuffle = False
        )


class FeedFowardModel(nn.Module):

    """
    Class for creating Feedforward Neural Network objects. 

    Because it was created mainly for experimental reasons, it's possible to 
    initialise it with 1 and also with 2 hidden layer, based on the given inputs. 

    Args:
        input_size: Size of the input layer (number of features)
        hidden_size1: Number of the neurons in the first hidden layer
        num_classes:  Number of class to be predicted (size of the output layer)
        hidden_size2: Number of the neurons in the second hidden layer
        gpu: default=False. If it's True the modell tries to use cuda if possible.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        num_classes: int,
        hidden_size2: int = None,
        gpu: bool = False
    ):

        super(FeedFowardModel,self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_classes = num_classes

        # Setting the device
        if gpu == True:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                warnings.warn('GPU is not available. CPU will be used.')
                self.device = 'cpu'
        else:
            self.device = 'cpu'


        if self.hidden_size2:

            self.relu = torch.nn.ReLU()
            self.hidden1 = torch.nn.Linear(self.input_size, self.hidden_size1)
            self.hidden2 = torch.nn.Linear(self.hidden_size1, self.hidden_size2)
            self.out_layer = torch.nn.Linear(self.hidden_size2, self.num_classes)


        elif self.hidden_size1:

            self.relu = torch.nn.ReLU()
            self.hidden1 = torch.nn.Linear(self.input_size, self.hidden_size1)
            self.out_layer = torch.nn.Linear(self.hidden_size1, self.num_classes)


    def forward(self, x):

        """
        Simple function that propagates trough the data on the network
        """

        if self.hidden_size2:

            out = self.hidden1(x)
            out = self.relu(out)
            out = self.hidden2(out)
            out = self.out_layer(out)


        elif self.hidden_size1:

            out = self.hidden1(x)
            out = self.relu(out)
            out = self.out_layer(out)

        return out


    def fit(
        self,
        epochs: int,
        lr: float,
        train_loader: DataLoader,
        val_loader: DataLoader
    )-> None:

        """
        Method to train the modell.

        It prints the current training and validation loss in every epoch.
        After the last epoch it plots the training curve.

        Args:
            epochs: number of epochs
            lr: learning rate 
            train_loader: the DataLoader object of the train data
            val_loader: the DataLoader object of the validation data
  
        """

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr= lr,
            weight_decay= 1e-4
        )

        train_loss_vals= []
        val_loss_vals= []

        for epoch in range(epochs):

            train_epoch_loss= []
            val_epoch_loss= []

            for _, (inputs, targets) in enumerate(train_loader):

                # forward pass
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                out = self.forward(inputs)

                # loss
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(out, targets.long().ravel())

                # backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_epoch_loss.append(loss.item())

            train_loss_vals.append(sum(train_epoch_loss)/len(train_epoch_loss))

            # validation part
            if val_loader:
                for _, (inputs, targets) in enumerate(val_loader):

                    # forward pass
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    out = self.forward(inputs)

                    # loss
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(out, targets.long().ravel())

                    val_epoch_loss.append(loss.item())
                
                val_loss_vals.append(sum(val_epoch_loss)/len(val_epoch_loss))

            if (epoch+1) % 1 == 0:
                print(f'epoch: {epoch+1} / {epochs} | training loss = {sum(train_epoch_loss)/len(train_epoch_loss):.4f} | validation lost {sum(val_epoch_loss)/len(val_epoch_loss):.4f}')

        plt.plot(train_loss_vals, "cornflowerblue", label="Train loss")
        plt.plot(val_loss_vals, "orange", label="Validation loss")
        plt.legend(loc="upper right")


    def accuracy(self, test_loader: DataLoader)-> None:
        """
        Return the accuracy score of the modell on the given dataset
        """

        for inputs, targets in test_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            out = self.forward(inputs)
            _, predictions = torch.max(out, 1)
        accuracy = accuracy_score(targets.cpu(), predictions.cpu())
        return accuracy


    def evaluate(self,test_loader: DataLoader)-> None:

        """
        Return a confusion matrix and a classification report
        """
      
        for inputs, targets in test_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

        out = self.forward(inputs)
        _, predictions = torch.max(out, 1)

        cm = confusion_matrix(targets.cpu(), predictions.cpu())
        df_cm = pd.DataFrame(cm)

        ax = sns.heatmap(
            df_cm, 
            annot=True, 
            cmap='Blues', 
            fmt='g'
        )

        ax.set(xlabel='Predicted label', ylabel= 'True label')
        ax.set(title= 'Confusion Matrix')
        plt.show()
        print("=" * 80)
        print(classification_report(targets.cpu(),predictions.cpu()))
            
        print("=" * 80)
