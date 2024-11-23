import argparse


def arg_parsing():
        
    parser = argparse.ArgumentParser(description="각종 Hyperparameters")

    parser.add_argument('--data_path', type=str, default='../../rotten_data/')
    parser.add_argument('--model_path', type=str, default='../../model_files/')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--embed_dims', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    return args