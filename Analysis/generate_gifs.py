"""
Generating gifs from plots etc
"""
import glob
from PIL import Image
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Generating Confusion Matrix')
    parser.add_argument('--run', help='Wandb run name')
    return parser.parse_args()


def make_gif(folder, target):

    # Create the frames
    frames = []
    imgs = glob.glob("{}/{}/*.png".format(folder, target))
    sorted_imgs = sorted(imgs)
    for i in sorted_imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save('{}/{}.gif'.format(folder, target), format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=200, loop=0)


# Specify paths for root dir, results respectively

if __name__ == '__main__':
    args = get_args()
    run_dir = args.run
    root_dir = '/Users/kelseydoerksen/Desktop/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/Results'
    results_dir = '{}/{}'.format(root_dir, run_dir)

    # Run gif maker for what we want to generate
    targets = ['precipitation', 'F1', 'prediction', 'confusion_matrix']
    for target in targets:
        print('Generating gifs for: {}'.format(target))
        make_gif(results_dir, target)
