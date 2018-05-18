import cv2
import numpy as np
import sys
import math

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.imshow("input image: " + name_input, inputImage)

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

tmp = np.copy(inputImage)
#Declared the quantities used in Histogram Equalization and initialized to zero.
height_of_histogram=np.zeros(101)
cummulative=np.zeros(101)
lookup_table=np.zeros(101)


for i in range(H1, H2) :
    for j in range(W1, W2) :
        b, g, r = inputImage[i, j]
        gray = round(0.3*r + 0.6*g + 0.1*b + 0.5)
        
        #Start the code here
        #normalizing
        r=r/255
        g=g/255
        b=b/255

        if(r<0.03928):
        	r=r/12.92
        else:
        	r=(np.power(((r+0.055)/1.055),2.4))

        if(g<0.03928):
        	g=g/12.92
        else:
        	g=(np.power(((g+0.055)/1.055),2.4))

        if(b<0.03928):
        	b=b/12.92
        else:
        	b=(np.power(((b+0.055)/1.055),2.4))

        rgb=[]
        rgb.append(r)
        rgb.append(g)
        rgb.append(b)

        #Convert to XYZ
        matrix = [[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]]
        xyz=[]
        xyz= np.dot(matrix,rgb)
        x=xyz[0]
        y=xyz[1]
        z=xyz[2]

        #Converting to LUV
        Xw=0.95
        Yw=1.0
        Zw=1.09
        Uw=((4*Xw)/(Xw+15*Yw+3*Zw))
        Vw=((9*Yw)/(Xw+15*Yw+3*Zw))

        t=y/Yw

        if(t>0.008856):
        	L=((116*(np.power(t,(1/3))))-16)
        else:
        	L=903.3*t
        if(L>100):
            L=100
        elif(L<0):            
            L=0
        height_of_histogram[(int)(L+0.5)]+=1
       
        tmp[i, j] = [L,Uw,Vw]
        print(L,Uw,Vw)

cv2.imshow('tmp', tmp)

# end of example of going over window
cummulative[0]=height_of_histogram[0]
# Calculating the f(i) values
for i in range(1,101):
    cummulative[i]=cummulative[i-1]+ height_of_histogram[i]

range_of_values=101
number_of_pixels=cummulative[100]
lookup_table[0]=(cummulative[0] * (float)(range_of_values))/ (2*number_of_pixels)

for i in range(1,101):
    lookup_table[i]=(cummulative[i-1] + cummulative[i])*((float)(range_of_values))/(2*number_of_pixels)

outputImage = np.zeros([rows, cols, bands], dtype=np.uint8)

for i in range(0, rows) :
    for j in range(0, cols) :
        b, g, r = inputImage[i, j]
        r=r/255
        g=g/255
        b=b/255

        if(r<0.03928):
        	r=r/12.92
        else:
        	r=(np.power(((r+0.055)/1.055),2.4))

        if(g<0.03928):
        	g=g/12.92
        else:
        	g=(np.power(((g+0.055)/1.055),2.4))

        if(b<0.03928):
        	b=b/12.92
        else:
        	b=(np.power(((b+0.055)/1.055),2.4))

        rgb=[]
        rgb.append(r)
        rgb.append(g)
        rgb.append(b)


        #Convert to XYZ
        matrix = [[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]]
        xyz=[]
        xyz= np.dot(matrix,rgb)
        x=xyz[0]
        y=xyz[1]
        z=xyz[2]
        

        #Converting to LUV
        Xw=0.95
        Yw=1.0
        Zw=1.09

        Uw=((4*Xw)/(Xw+15*Yw+3*Zw))
        Vw=((9*Yw)/(Xw+15*Yw+3*Zw))

        t=y/Yw

        if(t>0.008856):
        	L=((116*(np.power(t,(1/3))))-16)
        else:
        	L=903.3*t

        d=x+15*y+3*z

        u1=((4*x)/d)
        v1=((9*y)/d)
       
        u=13*L*(u1-Uw)
        v=13*L*(v1-Vw)
        L= lookup_table[(int)(L+0.5)]

        #LUV to XYZ
        #Divide by zero issue, by adding a hallucinated constant in L, i.e. 0.01
        u1=(u+(13.0*Uw)*(L+0.01))/(13.0*(L+0.01))
        v1=((v+(13.0*Vw)*(L+0.01))/(13*(L+0.01)))

        if(L>7.9996):
        	y=np.power(((L+16)/116),3)*Yw
        else:
        	y=(L/903.3)*Yw
       
        if(v1==0):
        	x=0
        	z=0
        else:
        	x=y*(2.25*(u1/v1))
        	z=((y*(3-(0.75*u1)-(5*v1)))/v1)       

        #Convert to linear sRGB
        Rsrgb=(3.240479*x)+(-1.53715*y)+(-0.498535*z)
        Gsrgb=(-0.969256*x)+(1.875991*y)+(0.041556*z)
        Bsrgb=(0.055648*x)+(-0.204043*y)+(1.057311*z)

        if(Rsrgb<0):
        	Rsrgb=0
        if(Gsrgb<0):
        	Gsrgb=0
        if(Bsrgb<0):
        	Bsrgb=0
        if(Rsrgb>1):
        	Rsrgb=1
        if(Gsrgb>1):
        	Gsrgb=1
        if(Bsrgb>1):
        	Bsrgb=1

        if(Rsrgb<0.00304):
        	Rsrgb=12.92*Rsrgb
        else:
        	Rsrgb=((1.055*(np.power(Rsrgb,(1/2.4))))-0.055)

        if(Gsrgb<0.00304):
        	Gsrgb=12.92*Gsrgb
        else:
        	Gsrgb=((1.055*(np.power(Gsrgb,(1/2.4))))-0.055)
        if(Bsrgb<0.00304):
        	Bsrgb=12.92*Bsrgb
        else:
        	Bsrgb=((1.055*(np.power(Bsrgb,(1/2.4))))-0.055)

        r=Rsrgb*255
        b=Bsrgb*255
        g=Gsrgb*255

        if (math.isnan(r)==True):
            r=0
        if (math.isnan(g)==True):
            g=0
        if (math.isnan(b)==True):
            b=0 

        print(b,g,r)


        outputImage[i,j] = [b, g, r]
cv2.imshow("output:", outputImage)
cv2.imwrite(name_output, outputImage);

# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
