import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, title_prefix):
	epochs = list(range(0, len(train_losses)))

	plt.plot(epochs, train_losses, label="Training Loss")
	plt.plot(epochs, valid_losses, label="Validation Loss")
	plt.legend()
	plt.title(f"{title_prefix} Loss Curve")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.savefig(f"../output/plots/Loss_Curve.png")
	plt.close()

	plt.plot(epochs, train_accuracies, label="Training Accuracy")
	plt.plot(epochs, valid_accuracies, label="Validation Accuracy")
	plt.legend()
	plt.title(f"{title_prefix} Accuracy Curve")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.savefig(f"../output/plots/Accuracy_Curve.png")
	plt.close()


def plot_confusion_matrix(results, class_names, title_prefix):
	y_true, y_pred = zip(*results)

	confusionMatrix = confusion_matrix(y_true, y_pred, normalize="true")

	display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=class_names)
	display.plot(cmap="Blues")
	plt.title(f"{title_prefix} Normalized Confusion Matrix")
	plt.xlabel("Predicted")
	plt.xticks(rotation=45)
	plt.ylabel("True")
	plt.tight_layout()
	plt.savefig(f"../output/plots/Confusion_Matrix.png")
	plt.close()