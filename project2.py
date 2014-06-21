# CSCI 2824 Project 2 - Jesse Greenberg
# 12/16/2013

# This project involves finding a solution to the Travelling
# Salesman problem.  In this project we are asked to complete 
# three of the four algorithms discussed in class.  I chose 
# to implement the Nearest Neighbor Algorithm, the Repeated
# Nearest Neighbor Algorithm, and the Cheapest Link Algorithm.

# Import the NetworkX module.  This is a package designed 
# to do graph work.
import networkx as nx

# Import a python timer:
import time

# Declare Python lists:
traveled = [] # Holds list of nodes traveled to.
# Holds the list of totals for each iteration of RNN
rneighbortots = [] 

# Declare empty graphs:
chart = nx.Graph() # This will hold the main weighted graph.
# These will hold empty graphs for the cheapest link algorithm.
# Chart2 holds the algorithm with the final hamilton circuit.
# Chart3 holds a graph which keeps track of the graph state
# to determine valid edges.
chart2 = nx.Graph() 
chart3 = nx.Graph()

# Read the data from a txt file into a matrix
# The data I used was from the link provided in the project description:
# people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html
# I chose to use the GR17 data set, which is a set of 17 nodes
# It is provided that the minimum tour has lenght 2085.
data = open("gr17_d.txt")
matrix = [map(int, line.split()) for line in data]


##############################################################
# Function for the nearest neighbor algorithms.
# nearest() finds the next valid nearest node for the algorithm.
# If the node has already been travelled to it is not a 
# valid node. nearest() also finds the new value for 
# the total cost of the walk through the circuit
def nearest( initnode, total, cheapest ):
	check = 0 # Initialize control flow variable.
	nextnode = 0 # Initialize the next node variable.
	for i in range(0, len(matrix)): 
		# If the weight of an edge is 0, skip that edge.
		# This represents a loop to the same node.
		if(chart[initnode][i]["weight"] == 0):
			continue
		# If the node has already been traveled to, set the
		# control variable to 1.  This means that the node
		# has already been traveled to, and we cannot travel there
		# again.
		for k in range(0, len(traveled)):
			if(i == traveled[k]):
				check = 1
		# If we have traveled to all nodes, return to the initial node
		# This completes the hamilton circuit. This is the last edge
		# in the algorithm.
		if len(matrix) == len(traveled):
			origin = traveled[0]
			cheapest = chart[initnode][origin]["weight"]
			nextnode = origin
		# If the edge is thus far valid, this finds the cheapest
		# edge and sets the next node equal to the node connected 
		# by that edge.
		if(chart[initialnode][i]["weight"] < cheapest and check == 0):
			cheapest = chart[initialnode][i]["weight"]
			nextnode = i
		check = 0
	# Calculate the new total
	# Return a tuple of the Next node to travel to, and the new
	# Cost of travel
	total = total + cheapest
	return (nextnode, total)
	
######################################################################
# Function for the cheapest link algorithm.  One of the requirements
# for the cheapest link algorithm is that no edge in the final walk 
# through the graph can create a mini circuit.  This algorithm
# makes sure that no edge will create a circuit until the final
# edge is placed.  Returns 1 if the edge is valid. Returns 0 if the 
# edge creates a circuit.  This method exploits the fact that 
# each node can only have at most 2 edges.  If a node has three
# edges, this function will break.

def checkconnect( init, final, graph):
	connected = [] # List of items connected to the initial node.
	traveled2 = [] # List of nodes we have already checked.
	while True:
		check3 = 0 # Initiate control flow variable
		# list which holds nodes connected to the initial node.
		traveled2.append(init)
		# imm is a list of immediate neighbors to the initial node.
		imm = graph.neighbors(init)
		
		# If the node has no connected neighbors, we immediately know
		# it is valid.
		if len(imm) == 0:
			return 1
		
		# If we have already checked one of the neighbors of the 
		# current node, take it out of the list of immediate neighbors.
		# Again, this exploits the fact that the list of immediate
		# neighbors can only have a length of 2.
		for i in range(0, len(traveled2)):
			if imm[0] == traveled2[i]:
				imm.pop(0)
				break
			if len(imm) > 2:	
				if imm[1] == traveled2[i]:
					imm.pop(1)
					break
		
		for i in range(0, len(traveled2)):
			if final == traveled2[i]:
				# This means that we have connected the initial
				# to the desired final node and an edge placement is
				# illegal.
				check3 = 1 # Set the control flow variable to 1.
		if check3 == 1:
			return 0;
		if len(imm) > 0:
			init = imm[0]
		else:
			return 1


# Iterate throught the matrix and use its information 
# to create the edges of the graph.  While iterating
# through the loops, also print an adjacency table
# which holds the weight of each edge to visualize what the 
# graph looks like.  
string = ""
for i in range(0, len(matrix)):
	for j in range(0, len(matrix)):
		string = string + "  %d  " % matrix[i][j]
		# Add an edge to the graph.
		chart.add_edge(i, j, weight = matrix[i][j])
	# If you want to print the data:	
	# print string
	string = ""
 
##############################################################
##############################################################

# NEAREST NEIGHBOR ALGORITHM:
# This algorithme works as follows:
# 1) Start at some initial node in the graph.
# 2) Find the cheapest edge from the initial node and travel to it.
# 3) From that node, travel to the next node along the next cheapest
#		edge, making sure that you only travel to nodes that you
# 		have not yet been to.
# 4) Continue until a hamilton circuit is constructed out of the graph.
print ""
print "NEAREST NEIGHBOR APPROXIMATION:"
# Adding Node 0 as Initial Node
start_time = time.time()
traveled.append(0)
initialnode = 0

# Total is the sum of edge weights.  It is the desired value.
# Initialize to zero.
total = 0

# Nearest() returns the the next "initial" node, and the new total
# cost.
while len(traveled) < len(matrix)+1:
	# Some arbitrary large value
	cheapest = (1 << 31)-1 # Max value for a 32 bit integer.
	temp = nearest(initialnode, total, cheapest)
	# Append the next node to the list of nodes traveled to.
	traveled.append(temp[0])
	# Set the initial node to the next node.
	initialnode = temp[0]
	# Find the new total.
	total = temp[1]
	# Now we repeat until the list of nodes traveled to is equal 
	# to the length of the matrix + 1. We add the one because
	# we have to return to the node of origin.
print "Nearest Neighbor walk through the graph: "
print traveled
print "Total cost for nearest neighbor approximation: "
print total
print "Time taken for Algorithm to complete: "
print time.time() - start_time, "seconds"

print ""
print ""

########################################################################
########################################################################

# REPEATED NEAREST NEIGHBOR ALGORITHM:
# The Repeated Nearest Neighbor Algorithm works like this:
# 1) Call the Nearest Neighbor Algorithm on some arbitrary node.
# 2) Call the Nearest Neighbor Algorithm on the rest of the nodes 
# 		in the graph until all nodes have been called as a node
#		of origin.
# 3) Save the total cost from call to the nearest neighbor algorithm
#		and find the cheapest starting node.
# 4) The cost from the cheapest node is the solution
print "REPEATED NEAREST NEIGHBOR APPROXIMATION:"
start_time = time.time()
costs = [] # Holds the N costs for each walk through the graph.
walks = [] # Holds list of walks by the RNN algorithm
del traveled[:] # Clear the list of nodes traveled to.

for i in range (0, len(matrix)):
	del traveled[:]
	initialnode = i
	total = 0
	traveled.append(i)
	
	
	# Call the Nearest Node Algorithm on each node i
	while len(traveled) < len(matrix)+1:
		cheapest = (1 << 31) - 1
		temp = nearest(initialnode, total, cheapest)
		traveled.append(temp[0])
		initialnode = temp[0]
		total = temp[1]
	#print "For Initial Node %d:" %i
	#print traveled
	#print total
	costs.append(total)
	walks.append(traveled[:])


cheapest = costs[0]
for i in range(0, len(costs)):
	if (costs[i] < cheapest):
		cheapest = costs[i]
		cheapesti = i
print "Cheapest walk and cost by Repeated Nearest Neighbor Algorithm:"
print walks[cheapesti]
print cheapest
print "Time taken for Algorithm to complete:"
print time.time() - start_time, "seconds"
	
########################################################################
########################################################################

# CHEAPEST LINK ALGORITHM
print ""
print "CHEAPEST LINK APPROXIMATION"
start_time = time.time()
# The cheapest link algorithm works like this:
# 1) Find the absolute cheapest edge in the graph and mark it.
# 2) Find the next cheapest edge in the graph and mark it.
# 3) Continue finding the next cheapest eges, only marking edges that
# 		a) do NOT make a node exceed a degree of 2
#		b) do NOT create a premature circuit in graph.

# My solution accomplishes this by creating temporary graphs
# to keep track of the conditions listed above.  This algorithm
# only adds valid edges to chart2, and adds already checked and 
# invalid edges to chart3.  In this way, the hamilton circuit 
# can be built using the checkconnect() method defined above to 
# ensure that no edge creates a circuit. Invalid edges are added
# to chart 3 to ensure that the degree of every node in the solution
# graph is less than three.

# Holds list of nodes already traveled to.
traveled3 = []
# Add all nodes to the graph copy without edges.  We will add the 
# desired edges to this graph.
for i in range(0, len(matrix)):
	chart2.add_node(i)

# Find the cheapest link in the graph:
for i in range(0, len(matrix)):
	for j in range(0, len(matrix)):
		if matrix[i][j] > 0 and matrix[i][j] < cheapest:
			cheapest = matrix[i][j]
			init = i
			final = j
# Add that edge to the graph copy
chart2.add_edge(init, final, weight = matrix[init][final])
cost = matrix[init][final]
chart3.add_edge(init, final)
traveled3.append(init)
traveled3.append(final)
# Find the next cheapest link in the graph
cheapest = (1<<31) -1 # Arbitrarily large value to reset cheapest.
for i in range(0, len(matrix)):
	for j in range(0, len(matrix)):
		if matrix[i][j] > 0 and matrix[i][j] < cheapest:
			# If that edge has NOT already been added...
			if(chart2.has_edge(i, j) == False):
				cheapest = matrix[i][j]
				init = i
				final = j
			else:
				continue
chart2.add_edge(init, final, weight = matrix[init][final])
cost = cost + matrix[init][final]
chart3.add_edge(init, final)
traveled3.append(init)
traveled3.append(final)

# Now iterate for the rest of the walk:
while len(traveled3) < (2*len(matrix)-2):
	cheapest = (1 << 31) - 1 # Reset cheapest to an arbitrary large num
	init = 0
	final = 0
	
	# Find the next cheapest edge, making sure that the edge
	# does not already exist on the graph, and that we have 
	# not already marked it as an INVALID edge.
	for i in range(0, len(matrix)):
		for j in range(0, len(matrix)):	
			if matrix[i][j] > 0 and matrix[i][j] < cheapest:
				if chart3.has_edge(i, j) == False:
					cheapest = matrix[i][j]
					init = i
					final = j
				
	# Make sure that an edge between init and final does not 
	# create a premature circuit.
	circuit = checkconnect(init, final, chart2)
	
	# If everything is valid according to the algorithm listed above:
	if chart2.degree(init) < 2 and chart2.degree(final) < 2 and circuit == 1:
		cost = cost + cheapest # Calculate the new tour length
		chart2.add_edge(init, final, weight = matrix[init][final])
		chart3.add_edge(init, final) # Add edge to checking graph
		traveled3.append(init) # List of nodes we have been to. 
		traveled3.append(final)
		#traveled3 = list(set(traveled3))
		
	else:
		# This makes sure that this is no longer available for an edge.
		chart3.add_edge(init, final)
# Now we can add the final edge.  For the final two nodes, their
# degree must be one.
degreecheck = []
for i in range(0, len(matrix)):
	deg = chart2.degree(i)
	if deg == 1:
		degreecheck.append(i)

# Place the final edge:
node1 = degreecheck[0]
node2 = degreecheck[1]
chart2.add_edge(node1, node2, weight = matrix[node1][node2])
cost = cost + chart2[node1][node2]["weight"]

print "Total cost by the Cheapest Link Approximation:"
print cost
print "Time taken for Algorithm to complete:"
print time.time() - start_time, "seconds"

########################################################################
########################################################################

# RESULTS:
# It is provided in the data set that the optimal solution has 
# a weight of 2085.  For the three Algorithms that I chose to 
# implement, I never found this solution.  However, I  came close.
# My most successful algorithm was the Repeated Nearest Neighbor.
# It found that the cheapest circuit had a length of 2178.
# Surprisingly, my least successful algorithm was Cheapest link.
# I would have thought this would be more successful as it attempts
# to keep only the cheapest links from the graph.  The total length
# of this algorithm is 2189.  In terms of timing, as expected, the 
# Nearest Neighbor Algorithm took the least amount of time.  The 
# Cheapest link algorithm took the most amount of time.  As expected,
# the repeated nearest neighbor took roughly N time longer than the 
# the nearest neighbor algorithm, where n is the number of nodes.
# We can see that the cheapest link algorithm takes the longest
# as it requires several checks and calls to functions for each 
# node and edge in the graph.



