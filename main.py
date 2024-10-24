from model_loader import load_model
from expression_tracking import predict_expression

BATCH_SIZE = 32
EPOCH_COUNT = 15
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-5

    
def main():
    while True:
        try:
            trainPrompt = int(input("\nEnter 0 or 1\n 0 - Train new model and overwrite pre-trained model\n 1 - Use pretrained model\n Prompt: "))
            if trainPrompt in [0, 1]:
                break
        except ValueError:
            print("Enter an integer")

    model = load_model(trainPrompt, BATCH_SIZE, EPOCH_COUNT, BASE_LR, WEIGHT_DECAY)  

    predict_expression(model)

if __name__ == "__main__":
    main()
