import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 

# BASIC GRAPH

# resize graph # Resize your Graph (dpi specifies pixels per inch. When saving probably should use 300 if possible)
plt.figure(figsize=(5,3), dpi = 100)

plt.plot([1,2,4],[2,5,3],label = '3x',color = 'red', linewidth = '3',markersize = 10,markeredgecolor = 'blue',linestyle = '--')
# cach khac : 
# use shorthand notation 
# fmt = '[color] [marker] [line]
plt.plot([1,2,4],[2,5,3],'b^--',label = '3x')

# cach khac 
x = [0,1,2,3,4]
y = [0,2,4,6,8]
plt.plot(x,y, label = '2x',color = 'yellow')

#line number two 
x2 = np.arange(0,4.5 , 0.5)
print(x2) 
# [0 0.5 1. 1.5 2. 2.5 3. 3.5 4.  ]
plt.plot(x2[:4],x2[:4]**2,'r',label = 'x^2')
plt.plot(x2[3:],x2[3:]**2,'r--')
# them title 
plt.title("Our First Graph",fontdict = {'fontname': 'Comic Sans MS','fontsize' : 20})
plt.xlabel('X Axis!',fontdict = {'fontname': 'Comic Sans MS'})
plt.ylabel('Y Axis!')

plt.xticks([0,1,2,3,4])
plt.yticks([0,2,4,6,7.5,8,10])

# add a legend 
plt.legend()

# Save figure (dpi 300 is good when saving so graph has high resolution)
plt.savefig('mygraph.png',dpi = 300)

plt.show()