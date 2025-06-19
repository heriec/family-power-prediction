import torch


def model_train(model, criterion, optimizer, num_epochs, train_loader, test_loader, device, model_name):
    train_losses, eval_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        curr_loss = 0.0
        for i, (inputs, y) in enumerate(train_loader):
            inputs, y = inputs.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], loss: {curr_loss/len(train_loader):.4f}")
        train_losses.append(curr_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            eval_loss = 0.0
            for inputs, y in test_loader:
                inputs, y = inputs.to(device), y.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, y)
                eval_loss += loss.item()
            print(f"Evaluation Loss: {eval_loss/len(train_loader):.4f}")
            eval_losses.append(eval_loss / len(test_loader))
    print("Finished")
    model_path = f'results/{model_name}-{round(eval_losses[-1], 3)}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}\n')
    return train_losses, eval_losses
