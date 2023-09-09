import torch
import torch.nn as nn


class Perceptron:
    def __init__(self, X_train):
        self.w = np.random.uniform(-1, 1, size=(X_train.shape[1]))
        self.lr = 0.001

        self.error = []
        self.ws = []
        self.bs = []
        self.b = np.random.uniform(-1, 1, size=1)

    def loss(self, y, prediction):
        # Clippen, um log(0) nicht undefiniert zu haben
        prediction = np.clip(prediction, 0.0001, 0.9999)
        loss = -np.mean(y*(np.log(prediction)) + (1-y)*np.log(1-prediction))
        return loss

    def accuracy(self, y, prediction):
        return 1-np.sum((prediction-y)**2)/y.shape[0]

    def predict(self, X_train):
        return np.where(np.dot(X_train, self.w)+self.b >= 0, 1, 0)

    def iteration(self, x, y):
        errors = []

        # für jede Zeile in X
        for xi, yi in zip(X_train, y):
            prediction = self.predict(xi)
            errors.append(self.loss(yi, prediction))

            # Gewichte nach Lernregel anpassen
            self.w = self.w+self.lr*(yi-prediction)*xi
            self.b = self.b+self.lr*(yi-prediction)

        # Fehler speichern für Visualisierung
        self.error.append(np.mean(errors))
        self.ws.append(self.w)
        self.bs.append(self.b)















# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#         # Anpassen der Größe abhängig von den Conv-Schichten,
#         self.fc1 = nn.Linear(32 * 8 * 8, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 8 * 8)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # "class Net(nn.Module):\n",
# # "  def __init__(self):\n",
# # "    super(Net, self).__init__()\n",
# # "    self.conv1 = nn.Conv2d(3,6,5)\n",
# # "    #self.conv1_bn = nn.BatchNorm2d(6)\n",
# # "    self.pool1 = nn.MaxPool2d(2,2)\n",
# # "    self.conv2 = nn.Conv2d(6,16,5)\n",
# # "    #self.conv2_bn = nn.BatchNorm2d(16)\n",
# # "\n",
# # "    self.fc1 = nn.Linear(16*5*5,200)\n",
# # "    self.fc2 = nn.Linear(200,150)\n",
# # "    self.fc3 = nn.Linear(150,10)\n",
# # "    \n",
# # "  def forward(self,x):\n",
# # "    x = self.pool1(self.conv1_bn(self.conv1(x)))\n",
# # "    x = self.pool1(self.conv2_bn(self.conv2(x)))\n",
# # "\n",
# # "    x = x.view(-1,16*5*5)\n",
# # "    x = F.relu(self.fc1(x))\n",
# # "    x = F.relu(self.fc2(x))\n",
# # "    x = F.softmax(self.fc3(x),dim=1)\n",
# # "    return x"
# #    ]
# #   }
# #  ],
# #  "metadata": {
# #   "language_info": {
# #    "name": "python"
# #   },
# #   "orig_nbformat": 4
# #  },
# #  "nbformat": 4,
# #  "nbformat_minor": 2
# # }
