import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import ALL_LETTERS, N_LETTERS, letter_to_tensor, line_to_tensor, load_data, random_training_example

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size, device=device)
    
category_lines, all_categories = load_data()
n_categories = len(all_categories)

n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories).to(device)

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000
batch_size = 32  # Define a batch size

for i in range(n_iters // batch_size):
    batch_loss = 0
    for _ in range(batch_size):
        category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
        line_tensor, category_tensor = line_tensor.to(device), category_tensor.to(device)
        output, loss = train(line_tensor, category_tensor)
        batch_loss += loss
    current_loss += batch_loss / batch_size

    if (i+1) % (plot_steps // batch_size) == 0:
        all_losses.append(current_loss / (plot_steps // batch_size))
        current_loss = 0

    if (i+1) % (print_steps // batch_size) == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{(i+1) * batch_size} {(i+1) * batch_size / n_iters * 100:.2f}% {batch_loss / batch_size:.4f} {line} / {guess} {correct}")

plt.figure()
plt.plot(all_losses)
plt.show()

def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line).to(device)
        hidden = rnn.init_hidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        guess = category_from_output(output)
        print(guess)

while True:
    sentence = input("Input:")
    if sentence == "quit":
        break
    predict(sentence)
