import torch
import torch.optim as optim
import torch.nn as nn
from Paraphrase import Network
from dataclass import loader
from sklearn.metrics import f1_score, accuracy_score

loader = loader()
train_loader = loader.train()
vad_loader = loader.vad()
test_loader = loader.test()
threshold = 0.5


def train(epochs = 100, learning_rate=0.0001):

    global train_loader, vad_loader, threshold
    model = Network().train().cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=learning_rate)

    best_model_on_dev = None
    best_acc_on_dev = 0
    best_f1_on_dev = 0
    for epoch in range(epochs):
        running_loss = 0
        true_label, predicted_label = [], []
        for batch_num, batch in enumerate(train_loader):
            x1, x2, target = batch[-1], batch[-2], torch.tensor(batch[0]).float()
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
            dev_x1, dev_X2, dev_target = dev_batch[-1], dev_batch[-2], torch.tensor(dev_batch[0]).float()
            dev_pred = model(dev_x1, dev_X2)[0]
            vad_pred.append(float(dev_pred > 0.5))
            vad_tar.append(dev_target[0])
        vad_acc = accuracy_score(vad_tar, vad_pred)
        vad_f1 = f1_score(vad_tar, vad_pred, average=None)

        print("(vad_acc, vad_f1):", vad_acc, vad_f1)

        if (vad_acc > best_acc_on_dev):
            best_acc_on_dev = vad_acc
            best_model_on_dev = model
            best_f1_on_dev = vad_f1

    testing_pred = []
    testing_tar = []
    best_model_on_dev.eval()
    for test_batch_num, test_batch in enumerate(test_loader):
        test_x1, test_X2, test_target = test_batch[-1], test_batch[-2], torch.tensor(test_batch[0]).float()
        test_pred = best_model_on_dev(test_x1, test_X2)[0]
        testing_pred.append(float(test_pred > 0.5))
        testing_tar.append(test_target[0])
    test_acc = accuracy_score(testing_tar, testing_pred)
    test_f1 = f1_score(testing_tar, testing_pred, average=None)
    test_f1_micro = f1_score(testing_tar, testing_pred, average="micro")
    test_f1_macro = f1_score(testing_tar, testing_pred, average="macro")
    print("(test_acc, test_f1, test_f1_macro, test_f1_micro):", test_acc, test_f1, test_f1_macro, test_f1_micro)

    PATH = "weights/trainedWeight.pt"
    torch.save({
        'model_state_dict': best_model_on_dev.state_dict()
    }, PATH)

    return best_acc_on_dev, best_f1_on_dev, test_acc, test_f1, test_f1_micro, test_f1_macro

def test(text1, text2):
    global test_loader, threshold
    PATH = "weights/trainedWeight.pt"
    model = Network().eval.cuda()
    checkpnt = torch.load(PATH)
    model.load_state_dict(checkpnt["model_state_dict"])

    output = model(text1, text2)
    return int(output[0]>threshold)

if __name__ == '__main__':
    train_tup = train()
    test_tup = test()