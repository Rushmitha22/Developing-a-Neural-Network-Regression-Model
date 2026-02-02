# EXPERIMENT 1 - Developing a Neural Network Regression Model
## NAME : RUSHMITHA R
## REGISTRATION NUMBER : 212224040281

## AIM :
To develop a neural network regression model for the given dataset.

## THEORY :
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features.
The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Neural Network Model :
<img width="1820" height="1017" alt="Screenshot 2026-02-02 094607EXP1" src="https://github.com/user-attachments/assets/91177b10-6ef2-428c-b60f-6fbbf3c926d2" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM :

### Name: RUSHMITHA R

### Register Number: 212224040281

```
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()


        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```

## Dataset Information :
<img width="347" height="496" alt="EXP1 TABLE" src="https://github.com/user-attachments/assets/c85ae5b5-6e76-4bcc-8588-f16703df9b0b" />



## OUTPUT :
<img width="407" height="237" alt="EPACH EXP1" src="https://github.com/user-attachments/assets/6487d76a-a241-4045-a31a-6fdb78a53632" />

<img width="397" height="43" alt="TESTLOSS EXP1" src="https://github.com/user-attachments/assets/ff1b0bc4-a697-4ad5-8162-8f53231858c1" />



## Training Loss Vs Iteration Plot :

<img width="737" height="572" alt="GRAPH EXP 1" src="https://github.com/user-attachments/assets/363d932b-aa7d-4a97-93fb-caded65afba0" />


## New Sample Data Prediction :
<img width="583" height="47" alt="PREDICTION EXP1" src="https://github.com/user-attachments/assets/1cf8308e-ae40-4243-8292-cf889886b6b6" />


## RESULT : 
Thus, a neural network regression model was successfully developed and trained using PyTorch.



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
