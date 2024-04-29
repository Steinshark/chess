import sys 
if len(sys.argv) < 1:
	print(f"specify filename")
	exit()
filename = sys.argv[1]
a = open(filename,"r").readlines()
total 	= 0 
elp = [] 
encounter 	= False
for item in a[1:]:
	if "tottime" in item:
		encounter = True 
	elif not encounter:
		continue

	c = item.split(" ")
	c = [l for l in c if not l == ""]
	try:
		c[1] = float(c[1])
		total += c[1]
		if c[1] > .5:
			elp.append((c[1],"".join(c[5:])))
	except ValueError:
		print("err")
		pass
	except IndexError:
		print("err")
		pass
import pprint 
print(f"Total: {total}")
pprint.pp(sorted(elp,reverse=True))