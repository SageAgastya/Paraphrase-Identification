import torch
import torch.optim as optim
import torch.nn as nn
from Paraphrase import Network
from dataclass import loader
from sklearn.metrics import f1_score, accuracy_score
import json

with open("config.json") as json_data_file:
    data = json.load(json_data_file)

learning_rate = data["learning_rate"]
epochs = data["epochs"]
threshold = data["threshold"]

# print(epochs)
# print(learning_rate)
# print(threshold)
loader = loader()
train_loader = loader.train()
vad_loader = loader.vad()
test_loader = loader.test()


def train(epochs, learning_rate):

    global train_loader, vad_loader, threshold
    model = Network().train().cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=learning_rate)

    best_model_on_dev = None
    best_acc_on_dev = 0
    best_f1_on_dev = 0
    for epoch in range(epochs):
        running_loss = 0
        true_label, predicted_label = [], []
        for batch_num, batch in enumerate(train_loader):
            # print(batch[-1][0], batch[-2][0], batch[0][0])
            x1, x2, target = batch[-1][0], batch[-2][0], torch.tensor(int(batch[0][0])).float().cuda()
            print(x1,x2,target)
            prediction = model(x1, x2)
            optimizer.zero_grad()
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            true_label.append(target)
            predicted_label.append(float(prediction > threshold))

        print("running_loss:", running_loss)

        vad_pred = []
        vad_tar = []
        for dev_batch_num, dev_batch in enumerate(vad_loader):
            dev_x1, dev_X2, dev_target = dev_batch[-1][0], dev_batch[-2][0], torch.tensor(int(dev_batch[0][0])).float().cuda()
            dev_pred = model(dev_x1, dev_X2)[0]
            vad_pred.append(float(int(dev_pred) > 0.5))
            vad_tar.append(int(dev_target.detach()))

        vad_acc = accuracy_score(vad_tar, vad_pred)
        vad_f1 = f1_score(vad_tar, vad_pred, average=None)
        vad_f1_micro = f1_score(vad_tar, vad_pred, average="micro")
        vad_f1_macro = f1_score(vad_tar, vad_pred, average="macro")

        print("(vad_acc, vad_f1, micro, macro):", vad_acc, vad_f1, vad_f1_micro, vad_f1_macro)

        if (vad_acc > best_acc_on_dev):
            best_acc_on_dev = vad_acc
            best_model_on_dev = model
            best_f1_on_dev = vad_f1


    PATH = "weights/trainedWeight.pt"
    torch.save({
        'model_state_dict': best_model_on_dev.state_dict()
    }, PATH)

    return best_acc_on_dev, best_f1_on_dev

def TestOnData():
    global test_loader, threshold
    PATH = "weights/trainedWeight.pt"
    model = Network().eval().cuda()
    checkpnt = torch.load(PATH)
    model.load_state_dict(checkpnt["model_state_dict"])

    testing_pred = []
    for test_batch_num, test_batch in enumerate(test_loader):
        test_x1, test_X2 = test_batch[-1][0], test_batch[-2][0]
        test_pred = model(test_x1, test_X2)[0]
        testing_pred.append(float(int(test_pred) > threshold))

    return testing_pred

def TestOnRandomData(s1,s2):
    global test_loader, threshold
    PATH = "weights/trainedWeight.pt"
    model = Network().eval().cuda()
    checkpnt = torch.load(PATH)
    model.load_state_dict(checkpnt["model_state_dict"])
    pred = model(s1,s2)[0]
    return 1 if threshold<int(pred) else 0

# train_tup = train(epochs, learning_rate)
# test_tup = TestOnData()
# print(TestOnRandomData("I am here","hi how are you"))
