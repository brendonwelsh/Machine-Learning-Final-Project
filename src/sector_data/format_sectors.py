import csv
import os

sectors = {}
finance = {}
health = {}
re = {}
tech = {}
energy = {}
k = 1
for filename in os.listdir('Financial'):
	with open('Financial/'+filename) as fin:
		reader=csv.reader(fin, skipinitialspace=True, quotechar="'")
		security = filename.split('.')[0]
		temp_dict = {}
		for row in reader:
			temp_dict[row[0]]=row[1:]
	#add temp dict to complete finance dict
	print('temp length = '+str(len(temp_dict))+" for the security"+security)
	finance[security] = temp_dict
	k = k+1
print('finance length = '+str(len(finance)))
#print(finance)

for filename in os.listdir('Health'):
        with open('Health/'+filename) as fin:
                reader=csv.reader(fin, skipinitialspace=True, quotechar="'")
                security = filename.split('.')[0]
                temp_dict = {}
                for row in reader:
                        temp_dict[row[0]]=row[1:]
        #add temp dict to complete finance dict
        print('temp length = '+str(len(temp_dict))+" for the security"+security)
        health[security] = temp_dict
        k = k+1
print('health length = '+str(len(health)))

for filename in os.listdir('RE'):
        with open('RE/'+filename) as fin:
                reader=csv.reader(fin, skipinitialspace=True, quotechar="'")
                security = filename.split('.')[0]
                temp_dict = {}
                for row in reader:
                        temp_dict[row[0]]=row[1:]
        #add temp dict to complete finance dict
        print('temp length = '+str(len(temp_dict))+" for the security"+security)
        re[security] = temp_dict
        k = k+1
print('Real Estate length = '+str(len(re)))

for filename in os.listdir('Tech'):
        with open('Tech/'+filename) as fin:
                reader=csv.reader(fin, skipinitialspace=True, quotechar="'")
                security = filename.split('.')[0]
                temp_dict = {}
                for row in reader:
                        temp_dict[row[0]]=row[1:]
        #add temp dict to complete finance dict
        print('temp length = '+str(len(temp_dict))+" for the security"+security)
        tech[security] = temp_dict
        k = k+1
print('tech length = '+str(len(tech)))

for filename in os.listdir('Energy'):
        with open('Energy/'+filename) as fin:
                reader=csv.reader(fin, skipinitialspace=True, quotechar="'")
                security = filename.split('.')[0]
                #print("parsing energy security "+security)
                temp_dict = {}
                for row in reader:
                        temp_dict[row[0]]=row[1:]
        #add temp dict to complete finance dict
        print('temp length = '+str(len(temp_dict))+" for the security"+security)
        energy[security] = temp_dict
        k = k+1
print('energy length = '+str(len(energy)))
#merge all 5 dicts
sectors["energy"] = energy
sectors["health"] = health
sectors["re"] = re
sectors["finance"] = finance
sectors["tech"] = tech
print('sectors length = '+str(len(sectors)))
