import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

from itertools import product

np.random.seed(302)


"""
def load_and_prepare_data: data loading and preparation: loads california hosuing data and spits into
    into train/validation/test sets.Returns: Dictionary containing all datasets and scalers,
    for regression and classification task 
"""
def load_and_prepare_data(test_size=0.25, validation_size=0.2, random_state=302, task = "regression"):

    X, y = fetch_california_housing(return_X_y=True)
   
    # split into full_train and test
    X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
    ) 
    
    if task == "classification":
       y_full_train[y_full_train < 2], y_test[y_test < 2] = 0, 0
       y_full_train[y_full_train >= 2], y_test[y_test >= 2] = 1, 1

    
    # second split: train and validation
    X_training, X_validation, y_training, y_validation = train_test_split(
        X_full_train, y_full_train, test_size=validation_size, random_state=random_state
    )
    
    # feature normalization with StandardScaler from sklern, fit_transform() on training set, transform() on validation set 
    scaler = StandardScaler()
    X_training_scaled = scaler.fit_transform(X_training)
    X_validation_scaled = scaler.transform(X_validation)

    # fit on full training set, and transform on test set
    scaler_full = StandardScaler()
    X_full_train_scaled = scaler_full.fit_transform(X_full_train)
    X_test_scaled = scaler_full.transform(X_test)
    
    #from numpy to pytorch tensor
    #float because numpy has float 64 and pytorch tensor has float 32
    #view(-1,1) becasue numpy arrays with d = 1 , has a shape of (100,) and not (100,1) 
    X_training_tensor = torch.from_numpy(X_training_scaled).float()
    X_validation_tensor = torch.from_numpy(X_validation_scaled).float()
    X_full_train_tensor = torch.from_numpy(X_full_train_scaled).float()
    X_test_tensor = torch.from_numpy(X_test_scaled).float()

    y_training_tensor = torch.from_numpy(y_training).float().view(-1, 1)
    y_validation_tensor = torch.from_numpy(y_validation).float().view(-1, 1)
    y_full_train_tensor = torch.from_numpy(y_full_train).float().view(-1, 1)
    y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)
    
    # Create datasets
    train_ds = TensorDataset(X_training_tensor, y_training_tensor)
    full_train_ds = TensorDataset(X_full_train_tensor, y_full_train_tensor)
    
    # Dictionary with all data

    return {
        'X_training': X_training_tensor,
        'X_validation': X_validation_tensor,
        'y_training': y_training_tensor,
        'y_validation': y_validation_tensor,
        'train_ds': train_ds,
        'scaler': scaler,
        'scaler_full': scaler_full,
        'X_full_train': X_full_train_tensor,
        'y_full_train': y_full_train_tensor,
        'X_test': X_test_tensor,
        'y_test': y_test_tensor,
        'full_train_ds': full_train_ds,
        'task': task, 
    }


"""
------------------------------------------------------------------------------------------------------------------------------
feature distribution: function for plotting histogram, longitude vs latidude, geo house pricing, statistics, pearson correlation coefficients
"""
def plot_feature_distribution(X_train,y_train):

    input_feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                       'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    df_train = pd.DataFrame(X_train, columns=input_feature_names)


    def plot_histogram():  # plot feature hsitograms with 99th percintile clipped

        features_to_plot = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                            'Population', 'AveOccup']

        fig, axes = plt.subplots(3, 2, figsize=(12, 8))

        for i, feature in enumerate(features_to_plot):
            ax = axes[i // 2, i % 2]

            p99 = df_train[feature].quantile(0.99)
            clipped_data = df_train[feature].clip(upper=p99)

            clipped_data.hist(bins=50, ax=ax, edgecolor='black')
            ax.set_xlabel(feature)
            ax.set_ylabel('count')
            ax.grid(False)
            
           
        plt.tight_layout()
        
        plt.savefig(f"histogram.png",  
                    dpi=300,                   
                    bbox_inches='tight',     
                    facecolor='white',) 
        
        plt.show()


    def plot_lo_la(): #plot 'Latitude', 'Longitude' on histogram

        features_to_plot = ['Latitude', 'Longitude']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for i, feature in enumerate(features_to_plot):
            ax = axes[i]

            ax.hist(df_train[feature],bins=50, edgecolor='black')
            ax.set_xlabel(feature)
            ax.set_ylabel('count')

        plt.tight_layout()
        plt.grid(False)
        plt.savefig(f"lo_la.png",  
                    dpi=300,                   
                    bbox_inches='tight',     
                    facecolor='white',)  
        plt.show()


    def geo_house_price(): # geographical house pricing data on a scatter plot,x = longitude,y = latitude, more red = more expensive

        plt.figure(figsize=(12, 10))
        
        scatter = plt.scatter(df_train['Longitude'], df_train['Latitude'], 
                            c=y_train, 
                            cmap='turbo', 
                            alpha=0.7, 
                            s=25, 
                            edgecolors='white', 
                            linewidth=0.1)
        
        cbar = plt.colorbar(scatter, label='Median House Value ($100,000)')
        cbar.ax.tick_params(labelsize=10)
        
        plt.title("California Housing Prices by Location", 
                fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Longitude", fontsize=13, fontweight='bold')
        plt.ylabel("Latitude", fontsize=13, fontweight='bold')
        
        plt.gca().set_aspect('equal')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("geo house price.png",  
                        dpi=300,                   
                        bbox_inches='tight',       
                        facecolor='white',         
                        )
        plt.show()


    
        


    def pearson(): # Pearson correlaion with panda dataframe

        df_train = pd.DataFrame(X_train, columns=input_feature_names)
        pearson_correlation_matrix = df_train.corr()
    
        plt.figure(figsize=(12, 8))
        sns.heatmap(pearson_correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
        plt.title('Feature Pearson Correlation Matrix')

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.savefig("pearson.png",  
                        dpi=300,                   
                        bbox_inches='tight',       
                        facecolor='white',         
                        )
        plt.show()


    def print_stats(): # statistics of all features
        
        print(f"{'Feature':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10} {'99th%':>10}")
        print("-"*70)
        
        for feature in input_feature_names:
            min_val = df_train[feature].min()
            max_val = df_train[feature].max()
            mean_val = df_train[feature].mean()
            median_val = df_train[feature].median()
            p99_val = df_train[feature].quantile(0.99)
            
            print(f"{feature:<15} {min_val:>10.2f} {max_val:>10.2f} {mean_val:>10.2f} "
                f"{median_val:>10.2f} {p99_val:>10.2f}")


    plot_histogram()   
    plot_lo_la() 
    geo_house_price()
    print_stats()
    pearson()


"""
------------------------------------------------------------------------------------------------------------------------------
Neural network class that works with automated testing that works for regression and classification task 
"""
class NeuralNetwork(nn.Module):
    #def __init__ is the constructro that will be called upon if i write model = NeuralNetwork(...)
    def __init__(self, input_features=8, hidden_layers=1, hidden_units=32, output_size=1, task="regression"):
        super().__init__() # calls nn.Module /parent class constructor 
        
        self.task= task # "classification" or "linear "regression" is stored 
        
         #empty list for all layers:
        layers = []
        
        #starts with 8 features 
        current_size = input_features
        
        #construct hidden layers with decreasing size
        for number in range(hidden_layers): 
            next_size = hidden_units // (2 ** number)
            
            #min. of 4 neurons:
            if next_size < 4:
                next_size = 4
            
            layers.append(nn.Linear(current_size, next_size)) # correspoinding to: self.fc1=nn.Linear(input_features, h1)
            layers.append(nn.ReLU()) #activation 
            current_size = next_size #update 
        
        # #output without ReLu : no activation for regression
        layers.append(nn.Linear(current_size, output_size))
        
        #nn.Sequential: from PyTorch: container class, can execute multiple layers in sequence
        self.network = nn.Sequential(*layers) 
    
     #how does our data flow through the neural network: x = input data gets passed to all layers (self.network)
    #returns the predicion logits (house price)    
    def forward(self, x):
        logits = self.network(x)
        
        if self.task == "classification":
            return torch.sigmoid(logits) #converts to [0 ,1] for binary classification 
        else:
            return logits #continous output for linear regression 


"""
------------------------------------------------------------------------------------------------------------------------------
function to train the model with early stopping when no improvement on validation loss after paticence counter
 
"""
def train_regression_model(model, data_dict, learning_rate, num_epochs=90, batch_size=64, 
                patience=20, min_delta=0.001):

    
    #dataLoader for mini batch training
    dataset = data_dict['train_ds']
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) # shuffle = true, shuffeld the data

    task = data_dict.get('task', 'regression')
    
    if task == 'classification':
        criterion = nn.BCELoss()  # Binary Cross Entropy for classification
    else:
        criterion = nn.L1Loss()   #loss function: after testing we desided to use MAE as our loss function
    
    #SGD: stochastic gradient descent which updates weights during training
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train_losses = []
    validation_losses = []
    best_validation_loss = float('inf') # first loss, is worst loss = infinity
    training_loss = float('inf')
    best_epoch = -1  # start at -1 so , first round starts at 0
    epochs_no_improve = 0 

    for epoch in range(num_epochs):
        # training phase
        model.train() #set our model to training mode 
        epoch_train_loss = 0.0
        
        for batch_X, batch_y in train_loader: #looping through batches
            optimizer.zero_grad() #clears the gradients
            
            #forward pass
            predictions = model(batch_X) 
            loss = criterion(predictions, batch_y)
            
            #backward pass: calculates new gradients
            loss.backward()
            
            #update weights
            optimizer.step()
            
            #accumulate the loss: 
            epoch_train_loss += loss.item() * batch_X.size(0)
            
        #average loss 
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        
        

        # validation phase
        model.eval() #set our model to evaluation mode
        with torch.no_grad(): #without calculating gradients on our validation data
            validation_predictions = model(data_dict['X_validation'])
            validation_loss = criterion(validation_predictions, data_dict['y_validation']).item()
        
        validation_losses.append(validation_loss)
        
        
        # best model with patience counter 
        if validation_loss < (best_validation_loss - min_delta):
            best_validation_loss = validation_loss
            training_loss = epoch_train_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # and early stopping if no improvement over patience counter
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} - No improvement for {patience} epochs")
            break

    return training_loss, best_validation_loss, best_epoch, train_losses, validation_losses


"""
------------------------------------------------------------------------------------------------------------------------------
hyperparameter tuning

-> perform grid search over hyperparamaters and save results in a csv file 

"""
def hyperparameter_tuning(data_dict, param_grid, num_epochs=100, csv_filename='hyperparameter_results.csv'):

    results = []
    
    task = data_dict.get("task", "regression")
    
    for hidden_layers, hidden_units, learning_rate, batch_size in product(
        param_grid["hidden_layers"],
        param_grid["hidden_units"],
        param_grid["learning_rate"],
        param_grid["batch_size"],
    ):
        print(f"\nTraining: Layers={hidden_layers}, Units={hidden_units}, LR={learning_rate}, Batch={batch_size}")
        
        model = NeuralNetwork(input_features=8, hidden_layers=hidden_layers, hidden_units=hidden_units, task=task)
        
        training_loss, validation_loss, best_epoch, train_losses, validation_losses = train_regression_model(
            model, 
            data_dict,
            learning_rate=learning_rate, 
            num_epochs=num_epochs, 
            batch_size=batch_size,
        )

        results.append({
            "hidden_layers": hidden_layers,
            "hidden_units": hidden_units,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "training_loss": training_loss,
            "best_validation_loss": validation_loss,
            "best_epoch": best_epoch,  
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "model": model
        })
    
    # create DataFrame for CSV export (without model and loss lists)
    df_results = pd.DataFrame([{
        "hidden_layers": r['hidden_layers'],
        "hidden_units": r['hidden_units'],
        "learning_rate": r['learning_rate'],
        "batch_size": r['batch_size'],
        "training_loss": r['training_loss'],
        "best_validation_loss": r['best_validation_loss'],
        "best_epoch": r['best_epoch'],
        "overfitting_gap": r['best_validation_loss'] - r['training_loss']
    } for r in results])
    
    # sorted by validation loss 
    df_results = df_results.sort_values('best_validation_loss').reset_index(drop=True)
    
    # save as csv table
    df_results.to_csv(csv_filename, index=False)
    
    
    return results, df_results


"""
------------------------------------------------------------------------------------------------------------------------------
Creat and train model, plot training/validation loss
"""

def create_train_model_and_plot(data_dict, params, num_epochs=100,patience=20, min_delta = 0.0001):
    
    
    task = data_dict.get('task', 'regression')
    
    # create  model
    model = NeuralNetwork(
        input_features=8, 
        hidden_layers=params['hidden_layers'], 
        hidden_units=params['hidden_units'],
        task=task,

    )

    # train model   
    training_loss, validation_loss, best_epoch, train_losses, validation_losses = train_regression_model(
        model,
        data_dict,
        learning_rate=params['learning_rate'],
        num_epochs=num_epochs,
        batch_size=params['batch_size'],
        patience= patience,
        min_delta = min_delta
    )
    
    # plot
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, validation_losses, 'r-', label='Validation Loss')
    
    # highlight best epoch
    plt.axvline(x=best_epoch, color='g', linestyle='--', 
                label=f'Best Epoch: {best_epoch}', linewidth=1.5)
    
    plt.xlabel('Epoch')
    
    if task == 'classification':
        plt.ylabel('Loss (BCE)')

    else:
        plt.ylabel('Loss (MAE)')
        
    plt.title('Training vs Validation Loss')
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"Val_loss_vs_Train_loss_{task}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    print(f"\nFinal Training Loss: {training_loss:.4f}")
    print(f"Best Validation Loss: {validation_loss:.4f}")
    print(f"patience = {patience}")
    print(f"min_delta = {min_delta}")
    
    return model, training_loss, validation_loss, best_epoch


"""
------------------------------------------------------------------------------------------------------------------------------
final training on on whole training set (taks c-part 1)
"""
def finaltrain_on_full_training_set(data_dict, hidden_layers, hidden_units, learning_rate, batch_size, num_epochs=250):
    

    
    task = data_dict.get('task', 'regression')

    full_train_loader = DataLoader(data_dict['full_train_ds'], batch_size=batch_size, shuffle=True, num_workers=0)
    
    # start model with specified configuration
    final_model = NeuralNetwork(
        input_features=8, 
        hidden_layers=hidden_layers, 
        hidden_units=hidden_units,
        task = task
    )
    
    # train on full training set
    
    if task == 'classification':
        criterion = nn.BCELoss()
    else:
        criterion = nn.L1Loss()

    
    optimizer = optim.SGD(final_model.parameters(), lr=learning_rate)
    

    for epoch in range(num_epochs):
        final_model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in full_train_loader:
            optimizer.zero_grad()
            predictions = final_model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        
        epoch_loss /= len(full_train_loader.dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}")
    
    
    return final_model


"""
------------------------------------------------------------------------------------------------------------------------------
evaluate model on test set (taks c- part 2) 
"""
def evaluate_final_model(final_model, data_dict, task= "regression"):
 
    final_model.eval()
    with torch.no_grad():
        y_pred_tensor = final_model(data_dict['X_test'])

    if task == "classification":
        criterion = nn.BCELoss()
    else:
        criterion = nn.L1Loss()

    test_loss = criterion(y_pred_tensor, data_dict['y_test']).item()
    
    y_pred = y_pred_tensor.numpy().flatten()
    y_true = data_dict['y_test'].numpy().flatten() 
    

    if task == "classification":
        # convert to binary predictions (threshold = 0.5)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        y_true_int = y_true.astype(int)
        
        # calculate classification metrics
        accuracy = accuracy_score(y_true_int, y_pred_binary)
        precision = precision_score(y_true_int, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_int, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_int, y_pred_binary, zero_division=0)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")  
        print(f"Test Loss (BCE): {test_loss:.4f}")
    

        # print confusion matrix
        cm = confusion_matrix(y_true_int, y_pred_binary)
        print("\nConfusion Matrix:")
        print(f"                  Predicted")
        print(f"                  0      1")
        print(f"Actual    0    {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"          1    {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        
        return y_pred, y_pred_binary, y_true, accuracy, precision, recall, f1
    else:
        # regression metrics
        r2 = r2_score(y_true, y_pred)
        mse = np.mean((y_true - y_pred) ** 2) 
        rmse = np.sqrt(mse)  
        
        print(f"Test Loss (MAE): {test_loss:.4f}")
        print(f"R² Score:        {r2:.4f}")
        print(f"MSE:             {mse:.4f}")  
        print(f"RMSE:            {rmse:.4f}")  
        print("="*70)
        
        return y_pred, y_true, r2, test_loss, mse, rmse
        


"""
------------------------------------------------------------------------------------------------------------------------------
scatter plot of predictions vs the actual prices
"""
def plot_predictions_vs_actual(y_pred, y_true, metric_value, task="regression"):
 
    plt.figure(figsize=(10, 8))

    plt.scatter(y_pred, y_true, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
        # Perfect prediction line
    min_val = min(y_pred.min(), y_true.min())
    max_val = max(y_pred.max(), y_true.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Predicted House Price ($100k)')
    plt.ylabel('Actual House Price ($100k)')
    plt.title('Predictions vs Ground Truth')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("predictions_vs_actual_regression.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


"""
------------------execution--------part----------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":
   

    print("Task a: investigate data, statistics and data normalization")
# Task a 1: Load and prepare data
    data_dict = load_and_prepare_data()

# Task a 2: Plot feature distribution
    #plot_feature_distribution(data_dict['X_full_train'],data_dict['y_full_train'])

    print("Task b: neural network for regression")
# Task b 1:  Define hyperparameter grids
#Grid for hyperparameter search
    param_grid = {
        "hidden_layers": [4,3],
        "hidden_units": [128,64],
        "learning_rate": [0.05,0.01],
        "batch_size": [64,32],
    }
    
    # Best parameters 
    best_param_grid = {
        "hidden_layers": 4,
        "hidden_units": 128,
        "learning_rate": 0.05,
        "batch_size": 32,
    }
    
# Task b 2: Perform hyperparameter tuning with param_grid 
    #results, df_results = hyperparameter_tuning(data_dict, param_grid, num_epochs=300, csv_filename='hyperparameter_results.csv')
    
# Task b 3: Create and train a new model with custom parameters from best_param_grid and plot it
    model, train_loss, val_loss, best_epoch = create_train_model_and_plot(
        data_dict, 
        best_param_grid, 
        num_epochs=250,
        patience=20,
        min_delta = 0.0001
    )
    
    print("Task c: final training and final test loss")

# Task c 1: Retrain on full training set with best_param_grid
    final_model = finaltrain_on_full_training_set(
        data_dict, 
        hidden_layers=best_param_grid['hidden_layers'],
        hidden_units=best_param_grid['hidden_units'],
        learning_rate=best_param_grid['learning_rate'],
        batch_size=best_param_grid['batch_size'],
        num_epochs= best_epoch  # train the full train data, with the best epoch 
    )
    
# Task c 2: Evaluate final model
    y_pred, y_true, r2, test_loss, mse, rmse = evaluate_final_model(final_model, data_dict)
    
# Task c3:  Plot predictions vs actual
    plot_predictions_vs_actual(y_pred, y_true, metric_value=r2, task='regression')

# Task d 1: Classification task

    print("Task d: binary classification problem")
   

    # d.1: data loading for classification 

    data_dict = load_and_prepare_data(task="classification")

    # d.3: create and train classification model  and plot
    class_model, class_train_loss, class_val_loss, class_best_epoch = create_train_model_and_plot(
    data_dict, 
    best_param_grid, 
    num_epochs=best_epoch,
    patience=20,
    min_delta = 0.0001,
    )

    # d.4: retrain on full training_set
    final_class_model = finaltrain_on_full_training_set(
    data_dict, 
    hidden_layers=best_param_grid['hidden_layers'],
    hidden_units=best_param_grid['hidden_units'],
    learning_rate=best_param_grid['learning_rate'],
    batch_size=best_param_grid['batch_size'],
    num_epochs= class_best_epoch
    )
    
    # d.5: evaluation of classificatoin task
    y_pred_proba, y_pred_binary, y_true_class, accuracy, precision, recall,f1 = evaluate_final_model(
        final_class_model, 
        data_dict, 
        task="classification"
    )
    
    