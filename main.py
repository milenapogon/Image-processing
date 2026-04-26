import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"

resp = urllib.request.urlopen(url)
img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# oryginał
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Oryginalny")
plt.axis("off")
plt.show()

# zmniejszenie
small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# grayscale
gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

# obrót
rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

# wynik
plt.imshow(rotated, cmap='gray')
plt.title("Wynik")
plt.axis("off")
plt.show()

# macierz
print(rotated[:5, :5])
