"""
Generate maps of Nepal from groundtruth and predictions
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse


def get_args():
  parser = argparse.ArgumentParser(description='Generate Plots from UNet Results')
  parser.add_argument('--save_dir', help='Save Directory')
  parser.add_argument('--threshold', help='Prediction Threshold')
  return parser.parse_args()

def plot_layer(array):
  plt.figure()
  plt.imshow(array)
  plt.gray()
  #plt.viridis()
  #plt.colorbar(ticks=[0,1])
  plt.show()

def combined_plot_layer(array):
  cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black","white","red"])
  plt.figure()
  plt.imshow(array, cmap=cmap, vmin=0, vmax=2)
  plt.colorbar(ticks=[0,1,2])
  plt.show()

if __name__ == '__main__':
  args = get_args()
  save_dir = args.save_dir
  threshold = args.threshold

  gt = np.load('{}/groundtruth_0.npy'.format(save_dir))
  pred = np.load('{}/pred_0.npy'.format(save_dir))

  gt0 = gt[0]
  pred0 = pred[0]

  # Apply thresholding
  pred0_thresh = (pred0 >= float(threshold)).astype(np.uint8)

  plt.subplot(1, 3, 1)  # row 1, column 2, count 1
  plt.imshow(pred0_thresh[0])
  plt.gray()
  plt.title('Threshold Model Output')

  # using subplot function and creating plot two
  # row 1, column 2, count 2
  plt.subplot(1, 3, 2)
  plt.imshow(gt0[0])
  plt.gray()
  plt.title('Groundtruth')

  # Combined overlap
  plt.subplot(1, 3, 3)
  combined = pred0_thresh[0] + gt0[0]
  cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black", "white", "red"])
  plt.imshow(combined, cmap=cmap, vmin=0, vmax=2)
  plt.title('Overlap in Red')

  # show plot
  plt.show()



