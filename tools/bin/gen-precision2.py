#py predicate-file feature-file
import sys,os

thre = 0.5
if (len(sys.argv) > 3):
	thre = float(sys.argv[3])
print "thre: %f"%thre
f = open(sys.argv[1])
line = f.readline()
idx = 1
if (line.split()[1] == '0'):
	idx = 2
predicate_list = f.readlines()
predicate_list = [str(int(float(item.strip().split()[idx]) > thre)) for item in predicate_list]

label_list = []
for line in open(sys.argv[2]):
	l = line.strip().split()
	if (l[0] == '+1'):
		l[0] = '1'
	label_list.append(l[0])

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(label_list)):
	if (predicate_list[i] == '1'):
		if (label_list[i] == '1'):
			tp += 1
		else:
			fp += 1
	else:
		if (label_list[i] == '1'):
			fn += 1
		else:
			tn += 1


print '%s %s'%(tp, fn)
print '%s %s'%(fp, tn)

print "For label 1: POS"
print "Precision: %f"%(tp * 1.0 / (tp + fp))
print "Recall: %f"%(tp * 1.0 / (tp + fn))

print "For label 0: NEG"
print "Precision: %f"%(tn * 1.0 / (tn + fn))
print "Recall: %f"%(tn * 1.0 / (tn + fp))

print "Total precision: %f"%((tp + tn) * 1.0 / (tp + tn + fp + fn))
