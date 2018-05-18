* Name: Simran Rawlani
* Project 1: Linear Scaling and Histogram Equalization in Luv domain
* NetID   : sxr174130


I. Purpose
	1. To perform linear scaling and histogram equalization in Luv domain of any color image, using a small window.
	2. Input parameters are following:
   	w1 h1 w2 h2 input-image output-image

II. Files List
	Luvscaling.py --- A program that performs Linear scaling, by converting the given image to Luv format and then scaling the values of L,
	and then converting it back to rgb format to display the image.

	Luvhistogram.py --- A program that performs histogram equalization by converting the given image to Luv format and then equalizing it 
	using the lookup table, and then converting it back to rgb format to display the image.

	readme.txt --- A report of the files in the unzipped folders.

	testimg.bmp- the image on which the program was tested.   
	pencil.png- image whose output generated bad image.
	pencil-output.png- image which looked bad.

III. How to Run the Program:
	Python 3 Version
	Open a terminal(command prompt- cmd), go to the current path. 
	Type(for scaling):  python Luvscaling.py 0.1 0.3 0.6 0.7 testimg.bmp output.bmp
	Type(for equalization): python Luvhistogram.py 0.1 0.3 0.6 0.7 testimg.bmp output.bmp 

IV. Output:    
	The program will print the scaled image and eqalized image, along with the values of RGB and Luv for each pixel of the input image.

V. Handling of any kind of errors(like division by zero, etc):
	1. By using hallucinated constant 0.01 and adding it to value of L, in order to avoid divide by 0 error, or zero by zero error.
	2. By applying conditions to values of r, g, b. 
	    Condition like checking NaN error, if it is met, the values(r,g,b) are equated to 0.

VI. Situation when image looked bad:
	See the image named- 'pencil-output'
	This output was displayed when histogram equalization was performed on image- 'pencil.png'
	It shows that, the image is not equalized completely and so, it made the picture looked green(where input image has 
	purple colour) in one part of image.