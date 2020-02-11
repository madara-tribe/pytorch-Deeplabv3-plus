import cv2
import matplotlib.pyplot as plt
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def plot(img):
  plt.imshow(img)
  plt.title('image/ overlay /segmentation')
  plt.tick_params(bottom=False, left=False,right=False, top=False)
  plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
  plt.show()
  #cv2.imwrite("original_image.png", hstacks)
