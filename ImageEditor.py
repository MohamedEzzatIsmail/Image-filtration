import customtkinter
from tkinter import filedialog
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import random


customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root=customtkinter.CTk()
root.title("Image Editor")
root.geometry('1000x600')

lable=customtkinter.CTkLabel(master=root, text='Image Editor', text_color='#095783', font=('', 40))
lable.pack(pady=10, padx=20)

tabview = customtkinter.CTkTabview(master=root, width=250)
tabview.pack(pady=20, padx=60, fill="both", expand=True)
tabview.add("Contrast")
tabview.add("Crop")
tabview.add("Gamma")
tabview.add("Gaussian")
tabview.add("Histogram")
tabview.add("Median Filter")
tabview.add("Negative")
tabview.add("Remove Noise")
tabview.add("Salt&Pepper")
tabview.add("Sharpener")

tabview.tab("Contrast").grid_columnconfigure(0, weight=1)
tabview.tab("Crop").grid_columnconfigure(0, weight=1)
tabview.tab("Gamma").grid_columnconfigure(0, weight=1)
tabview.tab("Gaussian").grid_columnconfigure(0, weight=1)
tabview.tab("Histogram").grid_columnconfigure(0, weight=1)
tabview.tab("Median Filter").grid_columnconfigure(0, weight=1)
tabview.tab("Negative").grid_columnconfigure(0, weight=1)
tabview.tab("Remove Noise").grid_columnconfigure(0, weight=1)
tabview.tab("Salt&Pepper").grid_columnconfigure(0, weight=1)
tabview.tab("Sharpener").grid_columnconfigure(0, weight=1)

lable = customtkinter.CTkLabel(master=tabview.tab("Contrast"), text='Choose Image', text_color='#095783', font=('', 20))
lable.pack(pady=10, padx=20)
lable = customtkinter.CTkLabel(master=tabview.tab("Crop"), text='Choose Image', text_color='#095783', font=('', 20))
lable.pack(pady=10, padx=20)
lable = customtkinter.CTkLabel(master=tabview.tab("Gamma"), text='Choose Image', text_color='#095783', font=('', 20))
lable.pack(pady=10, padx=20)
lable = customtkinter.CTkLabel(master=tabview.tab("Gaussian"), text='Choose Image', text_color='#095783', font=('', 20))
lable.pack(pady=10, padx=20)
lable = customtkinter.CTkLabel(master=tabview.tab("Histogram"), text='Choose Image', text_color='#095783', font=('', 20))
lable.pack(pady=10, padx=20)
lable = customtkinter.CTkLabel(master=tabview.tab("Median Filter"), text='Choose Image', text_color='#095783', font=('', 20))
lable.pack(pady=10, padx=20)
lable = customtkinter.CTkLabel(master=tabview.tab("Negative"), text='Choose Image', text_color='#095783', font=('', 20))
lable.pack(pady=10, padx=20)
lable = customtkinter.CTkLabel(master=tabview.tab("Remove Noise"), text='Choose Image', text_color='#095783', font=('', 20))
lable.pack(pady=10, padx=20)
lable = customtkinter.CTkLabel(master=tabview.tab("Salt&Pepper"), text='Choose Image', text_color='#095783', font=('', 20))
lable.pack(pady=10, padx=20)
lable = customtkinter.CTkLabel(master=tabview.tab("Sharpener"), text='Choose Image', text_color='#095783', font=('', 20))
lable.pack(pady=10, padx=20)


# main fuctions

def path():
    rootfile = filedialog.askopenfilename(initialdir='C:/Users/Mohamed/PycharmProjects/imageproc/Image'
                                          , title='Select Image'
                                          , filetypes=(("png files", "*.png"), ("jpg files", "*.jpg")
                                                       , ("all files", "*.*")))
    return rootfile


def contrast(rootfile):
    # read the image
    im = Image.open(rootfile)

    # image enhancer
    enhancer = ImageEnhance.Contrast(im)

    factor = 1  # gives original image
    im_output = enhancer.enhance(factor)
    im_output.show()

    factor = 0.5  # decrease constrast
    im_output = enhancer.enhance(factor)
    im_output.show()


    factor = 1.5  # increase contrast
    im_output = enhancer.enhance(factor)
    im_output.show()


def activecont():
    contrast(root_path)


def crop(rootfile, left, upper, right, lower):
    img = Image.open(rootfile)
    img.show()
    box = (left, upper, right, lower)
    img2 = img.crop(box)
    img2.show()


def setcrop():
    global left1
    global upper1
    global right1
    global lower1

    left1 = int(entry11.get())
    upper1 = int(entry12.get())
    right1 = int(entry13.get())
    lower1 = int(entry14.get())

    return left1, upper1, right1, lower1


def activecrop():
    crop(root_path, left1, upper1, right1, lower1)


def gamma(rootfile):
    invGamma = 1 / 3.3

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    img = cv2.imread(rootfile)
    gammaImg = cv2.LUT(img, table)

    cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original image', 1000, 1000)
    cv2.imshow('Original image', img)
    cv2.namedWindow('Gamma corrected image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gamma corrected image', 1000, 1000)
    cv2.imshow('Gamma corrected image', gammaImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def activegamma():
    gamma(root_path)


def gaussian(rootpath, r):
    # creating a image object
    im1 = Image.open(rootpath)
    im1.show()

    # applying the Gaussian Blur filter
    im2 = im1.filter(ImageFilter.GaussianBlur(radius=r))
    im2.show()


def setgau():
    global rd
    rd = int(entry21.get())
    return rd


def activegau():
    gaussian(root_path, rd)


def hist(rootf):
    img = cv2.imread(rootf)
    # alternative way to find histogram of an image
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


def activehist():
    hist(root_path)


def maid(rootfile):
    im1 = Image.open(rootfile)
    im1.show()

    # applying the median filter
    im2 = im1.filter(ImageFilter.MedianFilter(size=3))
    im2.show()


def activemaid():
    maid(root_path)


def negative(rootfile):
    img = Image.open(rootfile)
    img.show()
    im_invert = ImageOps.invert(img)
    im_invert.show()


def activeneg():
    negative(root_path)


def noise(rootfile):

    image = cv2.imread(rootfile, 1)
    noiseless_image = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)

    titles = ['Original Image', 'Image after removing the noise']
    images = [image, noiseless_image]
    plt.figure(figsize=(13, 5))
    for i in range(2):
        plt.subplot(2, 2, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


def activenoise():
    noise(root_path)


def slat(img):
    # Getting the dimensions of the image
    row, col = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


def saltp(rootfile):
    img = cv2.imread(rootfile, cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original image', 800, 800)
    cv2.imshow('Original image', img)
    # Storing the image
    cv2.namedWindow('salt-and-pepper', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('salt-and-pepper', 800, 800)
    cv2.imshow('salt-and-pepper', slat(img))

    cv2.waitKey(0)


def activesalt():
    saltp(root_path)


def sharpener(rootfile):
    im = Image.open(rootfile)

    enhancer = ImageEnhance.Sharpness(im)

    factor = 1
    im_s_1 = enhancer.enhance(factor)
    im_s_1.show()

    factor = 3
    im_s_1 = enhancer.enhance(factor)
    im_s_1.show()


def activesharp():
    sharpener(root_path)


def change():
    global root_path
    root_path = path()
    return root_path


# body
# contrast
btn1 = customtkinter.CTkButton(master=tabview.tab("Contrast"), text="show", command=activecont)
btn1.pack(pady=10)

# crop
btnf = customtkinter.CTkFrame(master=tabview.tab("Crop"))
btnf.columnconfigure(0, weight=1)
btnf.columnconfigure(1, weight=1)
btnf.columnconfigure(3, weight=1)
btnf.columnconfigure(4, weight=1)

lable = customtkinter.CTkLabel(master=btnf, text='Enter Size', text_color='#095783', font=('', 20))
lable.grid(row=0, column=0, pady=5, padx=5)
entry11 = customtkinter.CTkEntry(master=btnf, placeholder_text='Left')
entry11.grid(row=1, column=0, pady=5, padx=5)
entry12 = customtkinter.CTkEntry(master=btnf, placeholder_text='Upper')
entry12.grid(row=1, column=1, pady=5, padx=5)
entry13 = customtkinter.CTkEntry(master=btnf, placeholder_text='Right')
entry13.grid(row=1, column=2, pady=5, padx=5)
entry14 = customtkinter.CTkEntry(master=btnf, placeholder_text='Lower')
entry14.grid(row=1, column=3, pady=5, padx=5)

btnf.pack(pady=10)

btn1 = customtkinter.CTkButton(master=tabview.tab("Crop"), text="set", command=setcrop)
btn1.pack(pady=10)
btn1 = customtkinter.CTkButton(master=tabview.tab("Crop"), text="show", command=activecrop)
btn1.pack(pady=10)

# gamma
btn1 = customtkinter.CTkButton(master=tabview.tab("Gamma"), text="show", command=activegamma)
btn1.pack(pady=10)

# Gaussian
entry21 = customtkinter.CTkEntry(master=tabview.tab("Gaussian"), placeholder_text='Enter The Blur Value')
entry21.pack(pady=10, padx=20)
btn1 = customtkinter.CTkButton(master=tabview.tab("Gaussian"), text="set", command=setgau)
btn1.pack(pady=10)
btn1 = customtkinter.CTkButton(master=tabview.tab("Gaussian"), text="show", command=activegau)
btn1.pack(pady=10)

# hist
btn1 = customtkinter.CTkButton(master=tabview.tab("Histogram"), text="show", command=activehist)
btn1.pack(pady=10)

# maiden
btn1 = customtkinter.CTkButton(master=tabview.tab("Median Filter"), text="show", command=activemaid)
btn1.pack(pady=10)

# negative
btn1 = customtkinter.CTkButton(master=tabview.tab("Negative"), text="show", command=activeneg)
btn1.pack(pady=10)

# noise
btn1 = customtkinter.CTkButton(master=tabview.tab("Remove Noise"), text="show", command=activenoise)
btn1.pack(pady=10)

# salt
btn1 = customtkinter.CTkButton(master=tabview.tab("Salt&Pepper"), text="show", command=activesalt)
btn1.pack(pady=10)


# sharp
btn1 = customtkinter.CTkButton(master=tabview.tab("Sharpener"), text="show", command=activesharp)
btn1.pack(pady=10)

# path
btn1 = customtkinter.CTkButton(master=tabview.tab("Contrast"), text="Choose Image", command=change)
btn1.pack(pady=10)
btn1 = customtkinter.CTkButton(master=tabview.tab("Crop"), text="Choose Image", command=change)
btn1.pack(pady=10)
btn1 = customtkinter.CTkButton(master=tabview.tab("Gamma"), text="Choose Image", command=change)
btn1.pack(pady=10)
btn1 = customtkinter.CTkButton(master=tabview.tab("Gaussian"), text="Choose Image", command=change)
btn1.pack(pady=10)
btn1 = customtkinter.CTkButton(master=tabview.tab("Histogram"), text="Choose Image", command=change)
btn1.pack(pady=10)
btn1 = customtkinter.CTkButton(master=tabview.tab("Median Filter"), text="Choose Image", command=change)
btn1.pack(pady=10)
btn1 = customtkinter.CTkButton(master=tabview.tab("Negative"), text="Choose Image", command=change)
btn1.pack(pady=10)
btn1 = customtkinter.CTkButton(master=tabview.tab("Remove Noise"), text="Choose Image", command=change)
btn1.pack(pady=10)
btn1 = customtkinter.CTkButton(master=tabview.tab("Salt&Pepper"), text="Choose Image", command=change)
btn1.pack(pady=10)
btn1 = customtkinter.CTkButton(master=tabview.tab("Sharpener"), text="Choose Image", command=change)
btn1.pack(pady=10)


def change_appearance_mode_event(new_appearance_mode: str):
    customtkinter.set_appearance_mode(new_appearance_mode)


appearance_mode_label = customtkinter.CTkLabel(root, text="Appearance Mode:", anchor="w")
appearance_mode_label.pack()
appearance_mode_optionemenu = customtkinter.CTkOptionMenu(root, values=["Light", "Dark"]
                                                          , command=change_appearance_mode_event)
appearance_mode_optionemenu.pack(pady=10)

appearance_mode_optionemenu.set("Dark")

root.mainloop()

