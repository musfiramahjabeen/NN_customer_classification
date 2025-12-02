# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/5774cff6-8c6f-4da9-9fad-0c737504cd4b)


## DESIGN STEPS

# Model Training Steps

## Step 1: Data Preprocessing  
- Clean and normalize the dataset.  
- Split data into training, validation, and test sets.  

## Step 2: Model Architecture  
- **Input Layer:** Number of neurons equal to the number of features.  
- **Hidden Layers:** Two layers with ReLU activation.  
- **Output Layer:** Four neurons (Segments A, B, C, D) with softmax activation.  

## Step 3: Model Compilation  
- Use **categorical crossentropy** as the loss function.  
- Optimize using the **Adam optimizer**.  
- Track **accuracy** as a performance metric.  

## Step 4: Training  
- Train the model using **early stopping** to prevent overfitting.  
- Use a **batch size** of 32 and set an appropriate number of epochs.  

## Step 5: Model Evaluation  
- Compile the model using the same **loss function**, **optimizer**, and **accuracy metric**.  

## Step 6: Final Training  
- Train the model again with early stopping, a **batch size of 32**, and a suitable number of epochs.  


## PROGRAM

### Name: Abdur Rahman Basil A H
### Register Number: 212223040002

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x

```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```
```python
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```



## Dataset Information

![image](https://github.com/user-attachments/assets/43acf6aa-7651-431c-8fce-dd6003ed2c91)


## OUTPUT

### Confusion Matrix

![image](https://github.com/user-attachments/assets/3649ee61-fd95-40a4-a3b4-a4d8fe455c18)



### Classification Report

![image](https://github.com/user-attachments/assets/b7f30a65-f4f7-482b-97a3-89c3749ff142)



### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/7138cb36-5437-48ea-a2b5-b124f90de763)


## RESULT
Thus, we have developed a neural network classification model for the given dataset.
