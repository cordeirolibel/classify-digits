#========================================================
# CordeiroLibel 2017 - https://github.com/cordeirolibel/
# 
#========================================================

import psutil
import time


def cpu_count(min):
	out = list()
	k=1
	for x in range(int(min*60)):

		cpus = psutil.cpu_percent(interval=1, percpu=True)
		mem = psutil.virtual_memory()
		t = time.time()

		out.append([int(t)]+cpus+[mem.percent])

		if k%10 == 0:
			print(str(k)+'s')
		k+=1
	return out


out = cpu_count(7*60)

file = open('data/DadosPCAs_CPU','w')
file.write('timestamp\tCPU1\tCPU2\tCPU3\tCPU4\tmem(%)\n')

for o in out:
	line = ''
	for i in o:
		line += str(i)+'\t'

	file.write(line+'\n')