from train import train_and_evaluate

if __name__ == "__main__":
    print("Training without attack...")
    clean_accuracy = train_and_evaluate(poisoned_clients=0)
    print(f"Clean Accuracy: {clean_accuracy:.2f}%")

    print("\nTraining with 1 poisoned client and KMeans defense...")
    poisoned_accuracy = train_and_evaluate(poisoned_clients=1)
    print(f"Poisoned Accuracy with KMeans Defense: {poisoned_accuracy:.2f}%")
