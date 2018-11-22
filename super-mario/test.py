from multiprocessing import Process, Value, Array
import time


class A:
	def __init__(self, id):
		self.id1 = id
		self.id2 = 0

def change_id2(a,ctr,b):
	b[ctr] = a.id1

if __name__ == '__main__':
	num_proc = 10
	processes = [None] * num_proc
	A_objs = [A(i) for i in range(num_proc)]
	B = Array('d', range(num_proc))

	print ("before---->")
	for a in A_objs:
		print(a.id1, a.id2)

	for i,p in enumerate(processes):
		processes[i] = Process(target=change_id2, args=(A_objs[i],i,B,))
		processes[i].start()

	for i,p in enumerate(processes):
		print(p.join())

	print("after----->")
	for b in B:
		print(b)