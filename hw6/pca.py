import os
import sys
import numpy as np
from skimage import io

X = []
size = None    
for file in sorted(os.listdir(sys.argv[1])):
    img = io.imread(os.path.join(sys.argv[1], file))
    size = img.shape
    X.append(img.flatten())
    img = np.array(X)


mean = np.mean(img, axis=0)
X_center = img - mean
U, S, V = np.linalg.svd(X_center, full_matrices=False)
eigenvector = V

img = io.imread(os.path.join(sys.argv[1], sys.argv[2]))
img = img.flatten()
project = np.dot(eigenvector[:4,:], img - mean)
recon = mean + np.dot(eigenvector[:4,:].T, project)
recon -= np.min(recon)
recon /= np.max(recon)
recon = (recon*255).astype(np.uint8)        
io.imsave('reconstruction.jpg', recon.reshape(size))    
	

