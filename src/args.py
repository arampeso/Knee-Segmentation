import argparse

def get_args():

    parser = argparse.ArgumentParser(description="Knee Segmentation Project")

    parser.add_argument("-csv_dir", default="data/csv")

    parser.add_argument("-out_dir", default="sessions")

    parser.add_argument("-epochs", type=int, default=30)

    parser.add_argument("-bs", type=int, default=16)

    parser.add_argument("-lr", type=float, default=1e-4,
                        choices=[1e-2, 1e-3, 1e-4, 1e-6])

    parser.add_argument("-wd", type=float, default=1e-5,
                        choices=[1e-5, 1e-7])
    
    args = parser.parse_args()
    return args