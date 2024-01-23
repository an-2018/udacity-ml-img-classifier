import argparse
from utility import load_checkpoint, process_image, predict
import json
import torch

def main():
    parser = argparse.ArgumentParser(description='Predict the class of an input image using a trained network')
    parser.add_argument('input', help='Path to the input image')
    parser.add_argument('checkpoint', help='Path to the trained model checkpoint')
    parser.add_argument('--top_k', dest='top_k', type=int, default=1, help='Top K most likely classes')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json',
                        help='Path to a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for inference')

    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)

    class_to_name = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_to_name = json.load(f)

    probs, classes = predict(args.input, model, args.top_k, device)

    print("Predictions:")
    for i in range(len(probs)):
       
        predict_class = str(classes[i].item())
        print(f"Prediction {i + 1}: {class_to_name[predict_class]} with probability {probs[i].item():.3f}")
    
if __name__ == "__main__":
    main()
