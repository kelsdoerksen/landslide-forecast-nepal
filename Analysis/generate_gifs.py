"""
Generating gifs from plots etc
"""
import glob
from PIL import Image
from os import listdir

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
root_dir = '/Users/kelseydoerksen/Desktop/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/Results'
results_dir = '{}/honest-gorge-110_ForecastModelUKMO_EnsembleNum1'.format(root_dir)

# Run gif maker for what we want to generate
targets = ['F1']
#targets = ['precipitation', 'F1', 'prediction']
for target in targets:
    print('Generating gifs for: {}'.format(target))
    make_gif(results_dir, target)
