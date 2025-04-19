"""
Generate maps of Nepal from groundtruth and predictions
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
import re


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

def generate_plot(prediction, groundtruth, threshold, doy, save_dir):
  """
  Generate plot of gt, pred, threshold pred
  """
  # Apply thresholding
  pred0_thresh = (prediction[0] >= float(threshold)).astype(np.uint8)
  plt.figure(figsize=(10, 2))

  plt.subplot(1, 4, 1)  # row 1, column 3, count 1
  plt.imshow(prediction[0])
  plt.gray()
  plt.title('Raw Model Output')

  plt.subplot(1, 4, 2)  # row 1, column 3, count 1
  plt.imshow(pred0_thresh)
  plt.gray()
  plt.title('Threshold Model Output')

  # using subplot function and creating plot two
  # row 1, column 2, count 2
  plt.subplot(1, 4, 3)
  plt.imshow(groundtruth[0])
  plt.gray()
  plt.title('Groundtruth')

  # Combined overlap
  plt.subplot(1, 4, 4)
  combined = pred0_thresh + groundtruth[0]
  cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black", "white", "red"])
  plt.imshow(combined, cmap=cmap, vmin=0, vmax=2)
  plt.title('Overlap in Red')
  plt.suptitle('{} + 13 days outlook for threshold {}'.format(doy, threshold))

  # show plot
  #plt.show()

  plt.savefig('{}/{}_subplots_thr_{}.png'.format(save_dir, doy, threshold))
  plt.close()


if __name__ == '__main__':
  args = get_args()
  save_dir = args.save_dir
  threshold = args.threshold

  predictions = [f for f in os.listdir(save_dir) if 'pred' in f]
  predictions.sort()
  groundtruth = [f for f in os.listdir(save_dir) if 'groundtruth' in f]
  groundtruth.sort()

  # Extract the dates from the date list
  with open('{}/test_dates.txt'.format(save_dir), 'r') as file:
    content = file.read()
  # Use regular expression to find all sample names and dates (sample_YYYY-MM-DD)
  sample_pattern = r'sample_\d{4}-\d{2}-\d{2}'

  # Find all matching sample names
  samples = re.findall(sample_pattern, content)

  # Extract dates from the samples
  datelist = [sample.split('_')[1] for sample in samples]

  # Sort to make sure they are in order (they should be already)
  datelist.sort()

  #datelist = ['2023-04-01', '2023-04-02', '2023-04-03', '2023-04-04', '2023-04-05', '2023-04-06', '2023-04-07', '2023-04-08', '2023-04-09', '2023-04-10', '2023-04-11', '2023-04-12', '2023-04-13', '2023-04-14', '2023-04-15', '2023-04-16', '2023-04-17', '2023-04-18', '2023-04-19', '2023-04-20', '2023-04-21', '2023-04-22', '2023-04-23', '2023-04-24', '2023-04-25', '2023-04-26', '2023-04-27', '2023-04-28', '2023-04-29', '2023-04-30', '2023-05-01', '2023-05-02', '2023-05-03', '2023-05-04', '2023-05-05', '2023-05-06', '2023-05-07', '2023-05-08', '2023-05-09', '2023-05-10', '2023-05-11', '2023-05-12', '2023-05-13', '2023-05-14', '2023-05-15', '2023-05-16', '2023-05-17', '2023-05-18', '2023-05-19', '2023-05-20', '2023-05-21', '2023-05-22', '2023-05-23', '2023-05-24', '2023-05-25', '2023-05-26', '2023-05-27', '2023-05-28', '2023-05-29', '2023-05-30', '2023-05-31', '2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05', '2023-06-06', '2023-06-07', '2023-06-08', '2023-06-09', '2023-06-10', '2023-06-11', '2023-06-12', '2023-06-13', '2023-06-14', '2023-06-15', '2023-06-16', '2023-06-17', '2023-06-18', '2023-06-19', '2023-06-20', '2023-06-21', '2023-06-22', '2023-06-23', '2023-06-24', '2023-06-25', '2023-06-26', '2023-06-27', '2023-06-28', '2023-06-29', '2023-06-30', '2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04', '2023-07-05', '2023-07-06', '2023-07-07', '2023-07-08', '2023-07-09', '2023-07-10', '2023-07-11', '2023-07-12', '2023-07-13', '2023-07-14', '2023-07-15', '2023-07-16', '2023-07-17', '2023-07-18', '2023-07-19', '2023-07-20', '2023-07-21', '2023-07-22', '2023-07-23', '2023-07-24', '2023-07-25', '2023-07-26', '2023-07-27', '2023-07-28', '2023-07-29', '2023-07-30', '2023-07-31', '2023-08-01', '2023-08-02', '2023-08-03', '2023-08-04', '2023-08-05', '2023-08-06', '2023-08-07', '2023-08-08', '2023-08-09', '2023-08-10', '2023-08-11', '2023-08-12', '2023-08-13', '2023-08-14', '2023-08-15', '2023-08-16', '2023-08-17', '2023-08-18', '2023-08-19', '2023-08-20', '2023-08-21', '2023-08-22', '2023-08-23', '2023-08-24', '2023-08-25', '2023-08-26', '2023-08-27', '2023-08-28', '2023-08-29', '2023-08-30', '2023-08-31', '2023-09-01', '2023-09-02', '2023-09-03', '2023-09-04', '2023-09-05', '2023-09-06', '2023-09-07', '2023-09-08', '2023-09-09', '2023-09-10', '2023-09-11', '2023-09-12', '2023-09-13', '2023-09-14', '2023-09-15', '2023-09-16', '2023-09-17', '2023-09-18', '2023-09-19', '2023-09-20', '2023-09-21', '2023-09-22', '2023-09-23', '2023-09-24', '2023-09-25', '2023-09-26', '2023-09-27', '2023-09-28', '2023-09-29', '2023-09-30']

  count = 0
  for i in range(len(predictions)):
    pred = np.load('{}/{}'.format(save_dir, predictions[i]))
    gt = np.load('{}/{}'.format(save_dir, groundtruth[i]))
    for j in range(len(pred)):
      print('Plotting for doy: {}'.format(datelist[count]))
      pred_j = pred[j]
      gt_j = gt[j]
      generate_plot(pred_j, gt_j, threshold, datelist[count], save_dir)
      count +=1

