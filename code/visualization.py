import os
import numpy as np
import matplotlib.pyplot as plt

output_folder = "../results"

show3classes = False

def plot_features(dane, outfname, figtitle):

    aa_in_line = 50
    n_aa = len(dane)
    nrows = int(np.ceil(n_aa/aa_in_line))
    
    xsize = 10
    yspacing = 0.45
    ysize = 3.2 * yspacing

    if (show3classes):
        ysize += yspacing

    topmarg = 0.8
    lefttxt = -0.5
    yfigsize = topmarg+nrows*ysize

    fig = plt.figure(figsize=(xsize, yfigsize))
    ax = fig.add_subplot()
    
    # Set titles for the figure and the subplot respectively
    #fig.suptitle(figtitle, fontsize=14, fontweight='bold')
    fig.tight_layout()
    #ax.set_title('axes title')
    #ax.set_xlabel('xlabel')
    #ax.set_ylabel('ylabel')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Set both x- and y-axis limits to [0, 10] instead of default [0, 1]
    ax.axis([lefttxt, 10, 0, yfigsize])
    ax.text((xsize+lefttxt)/2, yfigsize-topmarg/2, figtitle,
         horizontalalignment='center', verticalalignment='bottom',
         color = "blue" , fontsize=16)

    colors8 = {
        "H": "red",
        "G": "orange",
        "I": "yellow",
        "B": "cyan",
        "E": "blue",
        "C": "green",
        "T": "grey",
        "S": "purple"
    }

    colors3 = {
        "H": "red",
        "E": "blue",
        "C": "green"
    }

    ss_dict = {'H': 'H', 'G': 'H', 'I': 'H',
               'B': 'E', 'E': 'E',
               'T': 'C', 'S': 'C',
               'C': 'C'}    
    
    yorigin = yfigsize - ysize - topmarg
    ypos = 0
    
    if show3classes:
        ysub = 0
    else:
        ysub = 1
    
    for row in range(nrows):
        for i in range(aa_in_line):
            xpos = xsize * i*1.0/aa_in_line
            elem_idx = i+row*aa_in_line
            if elem_idx>=n_aa:
                break

            ypos = (3.8-ysub) * yspacing + yorigin - row*ysize
            if ((i==0) or (elem_idx%5==4)):
                ax.text(xpos, ypos, str(elem_idx+1),
                    horizontalalignment='center',fontsize=7)

            ypos = (3-ysub) * yspacing + yorigin - row*ysize
            ax.text(xpos, ypos, dane['resname'].values[elem_idx],
                    horizontalalignment='center', fontsize=12)
            if(i==0):
                ax.text(lefttxt, ypos, 'AA', horizontalalignment='right',
                        color = 'black', fontsize=12, fontweight='bold')   

            code8 = dane['pred_SS8'].values[elem_idx]    
            ypos = (2-ysub) * yspacing + yorigin - row*ysize
            ax.text(xpos, ypos, code8,
                    horizontalalignment='center', color = colors8.get(code8), fontsize=12)   
            if(i==0):
                ax.text(lefttxt, ypos, 'ss8', horizontalalignment='right',
                        color = 'black', fontsize=12, fontweight='bold')   


            if show3classes:
                code3 = ss_dict.get(code8)  
                ypos = (1-ysub) * yspacing + yorigin - row*ysize
                ax.text(xpos, ypos, code3,
                        horizontalalignment='center', color = colors3.get(code3) , fontsize=12)   
                if(i==0):
                    ax.text(lefttxt, ypos, 'ss3', horizontalalignment='right',
                        color = 'black', fontsize=12, fontweight='bold')   


            plt.plot([lefttxt, 10],[ypos - 0.6*yspacing, ypos-0.6*yspacing], marker='', color='red')

    ypos = 1*yspacing + yorigin - (-1)*ysize
    plt.plot([lefttxt, 10],[ypos - 0.6*yspacing, ypos-0.6*yspacing], marker='', color='red')
    
    
    plt.savefig(os.path.join(output_folder, outfname + '.png'),dpi=300)
    plt.close()
    
