import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ================================================================
# PART A: DATA PREPARATION
# ================================================================

BATCH_SIZE    = 64
LEARNING_RATE = 0.001
EPOCHS        = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./mnist_data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./mnist_data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples : {len(train_dataset):,}")
print(f"Testing samples  : {len(test_dataset):,}")

fig, axes = plt.subplots(2, 10, figsize=(15, 4))
fig.suptitle("Part A: MNIST Dataset Sample Images", fontsize=13, fontweight='bold')
for digit in range(10):
    indices = [i for i, (_, label) in enumerate(train_dataset) if label == digit]
    for row, idx in enumerate(indices[:2]):
        img, label = train_dataset[idx]
        ax = axes[row][digit]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'Digit {label}', fontsize=8)
        ax.axis('off')
plt.tight_layout()
plt.savefig('Lab6_PartA_MNIST_Samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: Lab6_PartA_MNIST_Samples.png")


# ================================================================
# PART B: MODEL ARCHITECTURE AND TRAINING
# ================================================================

class MLP(nn.Module):
    def __init__(self, hidden_layers):
        super(MLP, self).__init__()
        layers = []
        input_size = 784
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, test_loader, epochs, lr, name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses     = []
    train_accuracies = []
    test_accuracies  = []

    print(f"\nTraining: {name}  |  Parameters: {count_parameters(model):,}")

    for epoch in range(epochs):
        model.train()
        total_loss, total_batches = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss    += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        train_losses.append(avg_loss)

        # Calculate training accuracy
        model.eval()
        train_correct, train_total = 0, 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

        train_accuracy = train_correct / train_total
        train_accuracies.append(train_accuracy)

        # Calculate test accuracy
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        test_accuracies.append(test_accuracy)

        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy*100:.2f}% | Test Acc: {test_accuracy*100:.2f}%")

    return train_losses, train_accuracies, test_accuracies


configs = {
    'Config 1 (Under-param) 1L x 8' : {'hidden': [8],         'color': '#E74C3C'},
    'Config 2 (Balanced) 2L x 128'  : {'hidden': [128, 128], 'color': '#27AE60'},
    'Config 3 (Over-param) 5L x 512': {'hidden': [512]*5,    'color': '#2980B9'},
}

all_results = {}
for name, config in configs.items():
    model = MLP(config['hidden'])
    losses, train_accs, test_accs = train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE, name)
    all_results[name] = {
        'model'          : model,
        'losses'         : losses,
        'train_accs'     : train_accs,
        'test_accs'      : test_accs,
        'accuracies'     : test_accs,
        'color'          : config['color'],
        'params'         : count_parameters(model),
        'final_loss'     : losses[-1],
        'final_train_acc': train_accs[-1],
        'final_test_acc' : test_accs[-1],
        'final_acc'      : test_accs[-1]
    }

print("\n" + "=" * 90)
print(f"{'Configuration':<35} {'Params':>12} {'Train Acc':>15} {'Test Acc':>15}")
print("-" * 90)
for name, res in all_results.items():
    print(f"{name:<35} {res['params']:>12,} {res['final_train_acc']*100:>14.2f}% {res['final_test_acc']*100:>14.2f}%")
print("=" * 90)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Part B: Training Curves — All 3 Configurations\nELEC 442/6651 Lab 6",
             fontsize=13, fontweight='bold')
for name, res in all_results.items():
    axes[0].plot(range(1, EPOCHS+1), res['losses'],
                 marker='o', color=res['color'], label=name, linewidth=2)
    axes[1].plot(range(1, EPOCHS+1), [a*100 for a in res['test_accs']],
                 marker='s', color=res['color'], label=name, linewidth=2)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss per Epoch', fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Test Accuracy (%)')
axes[1].set_title('Test Accuracy per Epoch', fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Lab6_PartB_Training_Curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: Lab6_PartB_Training_Curves.png")


# ================================================================
# PART C: DETAILED EVALUATION OF BEST MODEL
# ================================================================

best_name  = max(all_results, key=lambda k: all_results[k]['final_test_acc'])
best_model = all_results[best_name]['model']
print(f"\nBest model: {best_name}")
print(f"Test Accuracy: {all_results[best_name]['final_test_acc']*100:.2f}%")

best_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(device)
        outputs = best_model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

fig, ax = plt.subplots(figsize=(9, 5))
best_losses = all_results[best_name]['losses']
best_accs   = all_results[best_name]['test_accs']
ax.plot(range(1, EPOCHS+1), best_losses, color='#27AE60',
        marker='o', linewidth=2.5, markersize=7, label='Training Loss')
ax.fill_between(range(1, EPOCHS+1), best_losses, alpha=0.15, color='#27AE60')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax.set_title(f'Part C: Training Loss Curve — {best_name}\nFinal Test Accuracy: {best_accs[-1]*100:.2f}%',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, EPOCHS+1))
plt.tight_layout()
plt.savefig('Lab6_PartC_Loss_Curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: Lab6_PartC_Loss_Curve.png")

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=range(10), yticklabels=range(10), linewidths=0.5)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(f'Part C: Confusion Matrix — {best_name}\nTest Accuracy: {accuracy_score(all_labels, all_preds)*100:.2f}%',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('Lab6_PartC_Confusion_Matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: Lab6_PartC_Confusion_Matrix.png")

print("\nClassification Report:")
print("=" * 60)
report = classification_report(
    all_labels, all_preds,
    target_names=[f'Digit {i}' for i in range(10)],
    digits=4
)
print(report)

with open('Lab6_PartC_Classification_Report.txt', 'w') as f:
    f.write(f"ELEC 442/6651 Lab 6 - Classification Report\n")
    f.write(f"Best Model: {best_name}\n")
    f.write("=" * 60 + "\n")
    f.write(report)
print("Saved: Lab6_PartC_Classification_Report.txt")

print("\nAll done! Files generated:")
print("  Lab6_PartA_MNIST_Samples.png")
print("  Lab6_PartB_Training_Curves.png")
print("  Lab6_PartC_Loss_Curve.png")
print("  Lab6_PartC_Confusion_Matrix.png")
print("  Lab6_PartC_Classification_Report.txt")