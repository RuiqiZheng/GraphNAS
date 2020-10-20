# import numpy as np
# data = np.loadtxt("../Cora_macrosub_manager_logger_file_1602862015.8844802.txt",delimiter = ";", dtype='str')
file_path = "/home/qul/PenghuiRuan/GNASZRQ/GraphNAS/Cora_macrosub_manager_logger_file_1603120998.9757247.txt"
dataset = "Cora"
parameter = "generation100"
import matplotlib.pyplot as plt
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


data=open(file_path)
line=data.readline()
accuracy_generation = []
individual_time = []
individual_accuracy = []
while line:
    if line[0] == '[':
        line = line.split(';')
        individual_time.append(float(line[3][:-1]))
        individual_accuracy.append(float(line[2]))
        line=data.readline()
        continue
    if line[0] == 'F':
        line = line.split(':')
        accuracy_generation.append(float(line[1][:-1]))
        line=data.readline()
        continue
    line = data.readline()
    continue

plt.plot(individual_time,label="individual_time")
plt.xlabel('individual index')
plt.ylabel('time cost(s)')
plt.title('individual_time')
set_size(8,3)
plt.savefig('../images/individual_time_'+dataset+'_'+parameter+'.png',dpi=1000)
plt.show()

plt.plot(individual_accuracy,label="individual_accuracy")
plt.xlabel('individual index')
plt.ylabel('accuracy')
plt.title('individual_accuracy')
set_size(8,3)

plt.savefig('../images/individual_accuracy_'+dataset+'_'+parameter+'.png',dpi=1000)
plt.show()

plt.plot(accuracy_generation,label="accuracy_generation")
plt.xlabel('generation index')
plt.ylabel('best accuracy')
plt.title('best accuracy per generation')
plt.savefig('../images/accuracy_generation_'+dataset+'_'+parameter+'.png',dpi=1000)
plt.show()

