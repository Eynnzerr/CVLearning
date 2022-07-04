from AlexNet import AlexNet
from load_data import train_loader, classes
from model_utils import train_runner, Loss, Accuracy
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epoch_num = 20

for epoch in range(1, epoch_num+1):
    print("start_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    loss, acc = train_runner(model, device, train_loader, optimizer, epoch)
    Loss.append(loss)
    Accuracy.append(acc)
    # test_runner(model, device, testloader)
    print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')

print('Finished Training')
plt.subplot(2, 1, 1)
plt.plot(Loss)
plt.title('Loss')
plt.show()
plt.subplot(2, 1, 2)
plt.plot(Accuracy)
plt.title('Accuracy')
plt.show()

print(model)
torch.save(model, 'alexnet-catvsdog.pth')
