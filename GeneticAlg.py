import numpy as np
import math as m
import random as rd
from numpy import linalg as LA
from multiprocessing import Process, Queue
import copy as cp


def error(A, B): 
    """
    Input : A, B are square matrices
	Output : The L2 norm (Frobenius norm) of AB-Id

    Used to check how "good" B is the inverse of A
    If B is the inverse of A it should return 0.
    The lower the better.
    """

	P = np.dot(A,B)
	P = P-np.eye(np.shape(P)[0], np.shape(P)[1]) #Substract the identity

	return LA.norm(P) #L2 norm of P (= Frobenius norm)


def calculateInverseNumpy(A):
	# Numpy way of calculating the inverse of a matrix
	# Can be very long for high dimensions
	# Retuns the an "exact" inverse.
	return LA.inv(A)

def evaluate(A, gen, Q):
    """
    Input : A the matrix we want to inverse
    		gen the generation of approximate inverse of A
    		Q queue for multithreading...
	Output : List of matrices with the reward/error.
			[(M1, error1), (M2, error2)...]

	We calculate the error for each matrix (we duplicate them)
    """

	rewards = cp.deepcopy(gen) #[M1, M2, ...]
	N = len(gen)
	for k in range(N):
		rewards[k] = (rewards[k], error(A, rewards[k]))

	Q.put(rewards) #Sorted by reward later in the main function

def evaluateMultithreaded(A, gen, numberOfProcess = 6):
	# The multithreaded version of evaluate(A, gen, Q)
	# We start all the thread and we merge the rewards lists
	# We then sort that list by error, smallest first.

	rewards = []
	processList = []
	Q = Queue()
	N = len(gen)

	for i in range(numberOfProcess-1):
		#Creating the process & starting  
		p = Process(
		        target=evaluate,
		        args=(A, gen[i*int(N/numberOfProcess):(i+1)*int(N/numberOfProcess)], Q)) 
		        #divide the workload in slices of the generation
		processList.append(p)
		p.start()
	
	#The last process takes the remainder
	p = Process(
		        target=evaluate,
		        args=(A, gen[(numberOfProcess-1)*int(N/numberOfProcess):], Q))
	processList.append(p)
	p.start()

	#Extract the rewards
	for i in range(len(processList)):
		rewards.extend(Q.get())
	            
	# Wait for the processes to finish...
	for p in processList:
		p.join()

	return sorted(rewards, key=lambda x: x[1]) #Sort by error, smallest first


def calculateInverseGenetic(A, e = 0.05, maxIter = 2000, N = 4000, keep = 0.1, merge = 0.2, mutateBig = 0.2, mutateSmall = 0.2):
	#e the requested error if it can be satisfied within maxIter iterations.
	#N is the size of our population at each iteration
	#keep is the percentage of matrices kept for mutation and merge.

	#I choose to merge and mutate matrices from the kept set (other systems would work).
	#If merge + mutateBig + mutateSmall < 1 we add random matrices to complete the new generation

	#merge the percentage of matrices merged (reproduction) inside the kept set
	#mutateBig the percentage of matrices with big mutation 3/4/5 mutations
	#mutateSmall the percentage of matrices with small mutation 1/2

	#CRITICS: 
	#1 - I could have imagined a genetic algorithm evolutive with the error.
	#That is to say an exploration phase with a lot of randomness at the begining and
	#a growing mutateSmall towards the end of the convergence to find the optimal.
	
	shape = np.shape(A)

	#Create the first generation
	newGen = [np.random.rand(shape[1], shape[0])*2-1 for _ in range(N)] #List of random matrices in [-1,1]
	reward = evaluateMultithreaded(A, newGen, numberOfProcess = 6)
	eFound = reward[0][1]
	oldEFound = -1
	iterationCount = 1

	while eFound > e:
		if oldEFound != eFound:
			oldEFound = eFound
			print("Iteration #{}, error={}".format(iterationCount, eFound))

		if maxIter < iterationCount:
			break
		iterationCount += 1

		newGen = reward[:int(N*keep)] #[(M1, error1), (M2, error2)...]
		newGen = [x[0] for x in newGen] #[M1, M2,...]

		#----------- Big mutations
		for _ in range(int(N*mutateBig)):
			M1 = cp.deepcopy(newGen[rd.randint(0,int(N*keep)-1)])
			nbMutation = rd.randint(3,5)
			while nbMutation != 0:
				nbMutation -= 1
				i = rd.randint(0,shape[0]-1)
				j = rd.randint(0,shape[1]-1)
				
				M1[i][j] = M1[i][j] + (rd.random()*0.5-0.25) # mutation between [-0.25,0.25]
				
			newGen.append(M1)

		#----------- Small mutations
		for _ in range(int(N*mutateSmall)):
			M1 = cp.deepcopy(newGen[rd.randint(0,int(N*keep)-1)])
			nbMutation = rd.randint(1,2)
			while nbMutation != 0:
				nbMutation -= 1
				i = rd.randint(0,shape[0]-1)
				j = rd.randint(0,shape[1]-1)
				
				M1[i][j] = M1[i][j] + (rd.random()*0.2-0.1) # mutation between [-0.1,0.1]
				
			newGen.append(M1)

		#----------- Reproduction
		for _ in range(int(merge*N)):
			parent1 = newGen[rd.randint(0,int(N*keep)-1)]
			parent2 = newGen[rd.randint(0,int(N*keep)-1)]
			child = [[parent1[i][j] if rd.random()<0.5 else parent2[i][j] for i in range (shape[1])] for j in range(shape[0])]
			
			newGen.append(child)


		#------------ random
		n = len(newGen)
		for k in range(n,N):
			newGen.append(np.random.rand(shape[1], shape[0])*2-1)

		reward = evaluateMultithreaded(A, newGen, numberOfProcess = 6)
		eFound = reward[0][1]

	return reward[0][0] 






if __name__ == "__main__":
	import time
	A = np.random.rand(2,2)*200-100 # in [-100,100]

	#Timer for calculateInverseNumpy
	start = time.time()
	M1 = calculateInverseNumpy(A)
	T1 = time.time() - start
	print("calculateInverseNumpy took {}s and the error is {}".format(T1, error(A, M1)))
	print(np.dot(A,M1))


	#Timer for calculateInverseGenetic
	start = time.time()
	M2 = calculateInverseGenetic(A)
	T2 = time.time() - start
	print("calculateInverseGenetic took {}s and the error is {}".format(T2, error(A,M2)))
	print(np.dot(A,M2))



	
