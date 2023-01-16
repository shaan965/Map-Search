from hashlib import new
from search.algorithms import State
import heapq
from search.map import Map
import getopt
import math
import sys

def main():
    """
    Function for testing your implementation. Run it with a -help option to see the options available. 
    """
    optlist, _ = getopt.getopt(sys.argv[1:], 'h:m:r:', ['testinstances', 'plots', 'help'])

    plots = False
    for o, _ in optlist:
        if o in ("-help"):
            print("Examples of Usage:")
            print("Solve set of test instances: main.py --testinstances")
            print("Solve set of test instances and generate plots: main.py --testinstances --plots")
            exit()
        elif o in ("--plots"):
            plots = True
    test_instances = "test-instances/testinstances.txt"
    gridded_map = Map("dao-map/brc000d.map")
    
    nodes_expanded_biastar = []   
    nodes_expanded_astar = []   
    nodes_expanded_mm = []
    
    start_states = []
    goal_states = []
    solution_costs = []
       
    file = open(test_instances, "r")
    for instance_string in file:
        list_instance = instance_string.split(",")
        start_states.append(State(int(list_instance[0]), int(list_instance[1])))
        goal_states.append(State(int(list_instance[2]), int(list_instance[3])))
        
        solution_costs.append(float(list_instance[4]))
    file.close()
        
    for i in range(0, len(start_states)):   

        start = start_states[i]
        goal = goal_states[i]
    
        cost, expanded_astar = a_star(start, goal, gridded_map.successors) # Replace None, None with a call to your implementation of A*
        
        nodes_expanded_astar.append(expanded_astar)

        if cost != solution_costs[i]:
            print("There is a mismatch in the solution cost found by A* and what was expected for the problem:")
            print("Start state: ", start)
            print("Goal state: ", goal)
            print("Solution cost encountered: ", cost)
            print("Solution cost expected: ", solution_costs[i])
            print()

        cost, expanded_mm = mm(start, goal, gridded_map.successors) # Replace None, None with a call to your implementation of MM
        nodes_expanded_mm.append(expanded_mm)
        
        if cost != solution_costs[i]:
            print("There is a mismatch in the solution cost found by MM and what was expected for the problem:")
            print("Start state: ", start)
            print("Goal state: ", goal)
            print("Solution cost encountered: ", cost)
            print("Solution cost expected: ", solution_costs[i])
            print()

        cost, expanded_biastar = bi_astar(start, goal, gridded_map.successors) # Replace None, None with a call to your implementation of Bi-A*
        nodes_expanded_biastar.append(expanded_biastar)
        
        if cost != solution_costs[i]:
            print("There is a mismatch in the solution cost found by Bi-A* and what was expected for the problem:")
            print("Start state: ", start)
            print("Goal state: ", goal)
            print("Solution cost encountered: ", cost)
            print("Solution cost expected: ", solution_costs[i])
            print()
    
    print('Finished running all tests. The implementation of an algorithm is likely correct if you do not see mismatch messages for it.')

    if plots:
        from search.plot_results import PlotResults
        plotter = PlotResults()
        plotter.plot_results(nodes_expanded_mm, nodes_expanded_astar, "Nodes Expanded (MM)", "Nodes Expanded (A*)", "nodes_expanded_mm_astar")
        plotter.plot_results(nodes_expanded_mm, nodes_expanded_biastar, "Nodes Expanded (MM)", "Nodes Expanded (Bi-A*)", "nodes_expanded_mm_biastar")


def a_star(sinit: 'State', sgoal: 'State', succ):

    open_list = []
    # try with get_g()
    start_fval = (h_val(sinit, sgoal)+0)
    sinit.set_cost(start_fval)
    heapq.heappush(open_list, sinit)
    closed_list = {sinit.state_hash(): sinit}
    
    counter = 0  

    while len(open_list) != 0:
        node = heapq.heappop(open_list)
        counter +=1
        if node == sgoal:
            return node.get_cost(), counter
        
        for new_node in succ(node): 
            new_node_h_val = h_val(new_node, sgoal)
            new_node_f_val = new_node_h_val + new_node.get_g()
            new_node.set_cost(new_node_f_val)
            if new_node.state_hash() not in closed_list: 
                new_node.set_cost(new_node_f_val) 
                heapq.heappush(open_list, new_node)
                closed_list[new_node.state_hash()] = new_node
                                
            if new_node.state_hash() in closed_list and new_node_f_val < (h_val(closed_list[new_node.state_hash()], sgoal) + closed_list[new_node.state_hash()].get_g()):
                closed_list[new_node.state_hash()].set_g(new_node.get_g())
                closed_list[new_node.state_hash()].set_cost(new_node.get_cost())
                heapq.heapify(open_list)
                    
    return -1, counter

def bi_astar(sinit: 'State', sgoal: 'State', succ):
    open_list_f = []
    f_start_fval = (h_val(sinit, sgoal)+0)
    sinit.set_cost(f_start_fval)
    heapq.heappush(open_list_f, sinit)

    open_list_b = []
    b_start_fval = (h_val(sgoal, sinit)+0)
    sgoal.set_cost(b_start_fval)
    heapq.heappush(open_list_b, sgoal)

    closed_list_f = {sinit.state_hash(): sinit}
    closed_list_b = {sgoal.state_hash(): sgoal}

    u_value = math.inf
    counter = 0
    
    while len(open_list_f) != 0 and len(open_list_b) != 0:

        if u_value <= (min(open_list_f[0].get_cost(), open_list_b[0].get_cost())):
            return u_value, counter

        if open_list_f[0].get_cost() < open_list_b[0].get_cost():      
            node = heapq.heappop(open_list_f)
            counter += 1

            for new_node in succ(node): 
                new_node_h_val = h_val(new_node, sgoal)
                new_node_f_val = new_node_h_val + new_node.get_g()
                new_node.set_cost(new_node_f_val)

                if new_node.state_hash() in closed_list_b:
                    n_new_node_fval = (h_val(closed_list_b[new_node.state_hash()], sgoal) + closed_list_b[new_node.state_hash()].get_g())
                    u_value = min(u_value, (n_new_node_fval + new_node.get_cost()))

                if (new_node.state_hash() in closed_list_f) and new_node_f_val < (h_val(closed_list_f[new_node.state_hash()], sgoal) + closed_list_f[new_node.state_hash()].get_g()):
                    closed_list_f[new_node.state_hash()].set_g(new_node.get_g())
                    closed_list_f[new_node.state_hash()].set_cost(new_node.get_cost())
                    heapq.heapify(open_list_f) 

                if new_node.state_hash() not in closed_list_f: 
                    new_node.set_cost(new_node_f_val) 
                    heapq.heappush(open_list_f, new_node)
                    closed_list_f[new_node.state_hash()] = new_node 
        else:
            node = heapq.heappop(open_list_b)
            counter += 1

            for new_node in succ(node): 
                new_node_h_val = h_val(new_node, sinit)
                new_node_f_val = new_node_h_val + new_node.get_g()
                new_node.set_cost(new_node_f_val)

                if new_node.state_hash() in closed_list_f:

                    n_new_node_fval = (h_val(closed_list_f[new_node.state_hash()], sgoal) + closed_list_f[new_node.state_hash()].get_g())
                    u_value = min(u_value, (n_new_node_fval + new_node.get_cost()))
                    
                if (new_node.state_hash() in closed_list_b) and new_node_f_val < (h_val(closed_list_b[new_node.state_hash()], sinit) + closed_list_b[new_node.state_hash()].get_g()):
                    closed_list_b[new_node.state_hash()].set_g(new_node.get_g())
                    closed_list_b[new_node.state_hash()].set_cost(new_node.get_cost())
                    heapq.heapify(open_list_b) 

                if new_node.state_hash() not in closed_list_b: 
                    new_node.set_cost(new_node_f_val) 
                    heapq.heappush(open_list_b, new_node)
                    closed_list_b[new_node.state_hash()] = new_node

    return -1, counter

def mm(sinit: 'State', sgoal: 'State', succ):
    open_list_f = []
    f_start_fval = (h_val(sinit, sgoal)+0)
    f_start_Pval = max(f_start_fval, 2*sinit.get_g())
    sinit.set_cost(f_start_Pval)
    heapq.heappush(open_list_f, sinit)

    open_list_b = []
    b_start_fval = (h_val(sgoal, sinit)+0)
    b_start_Pval = max(b_start_fval, 2*sgoal.get_g())
    sgoal.set_cost(b_start_Pval)
    heapq.heappush(open_list_b, sgoal)

    closed_list_f = {sinit.state_hash(): sinit}
    closed_list_b = {sgoal.state_hash(): sgoal}

    u_value = math.inf
    counter = 0

    while len(open_list_f) != 0 and len(open_list_b) != 0:

        if u_value <= (min(open_list_f[0].get_cost(), open_list_b[0].get_cost())):
            return u_value, counter
        
        if open_list_f[0].get_cost() < open_list_b[0].get_cost():
            node = heapq.heappop(open_list_f)
            counter += 1
        
            for new_node in succ(node): 
                    new_node_h_val = h_val(new_node, sgoal)
                    new_node_f_val = new_node_h_val + new_node.get_g()
                    new_node_P = max(new_node_f_val, 2*new_node.get_g())
                    new_node.set_cost(new_node_P)

                    if new_node.state_hash() in closed_list_b:
                        u_value = min(u_value, closed_list_b[new_node.state_hash()].get_g() + new_node.get_g())
                    
                    if (new_node.state_hash() in closed_list_f) and (new_node < closed_list_f[new_node.state_hash()]):
                        closed_list_f[new_node.state_hash()].set_g(new_node.get_g())
                        closed_list_f[new_node.state_hash()].set_cost(new_node.get_cost())
                        heapq.heapify(open_list_f)
                    
                    if new_node.state_hash() not in closed_list_f:
                        new_node.set_cost(new_node_P)
                        heapq.heappush(open_list_f, new_node)   
                        closed_list_f[new_node.state_hash()] = new_node
        else:
            node = heapq.heappop(open_list_b)
            counter += 1
        
            for new_node in succ(node): 
                    new_node_h_val = h_val(new_node, sinit)
                    new_node_f_val = new_node_h_val + new_node.get_g()
                    new_node_P = max(new_node_f_val, 2*new_node.get_g())
                    new_node.set_cost(new_node_P)

                    if new_node.state_hash() in closed_list_f:
                        u_value = min(u_value, closed_list_f[new_node.state_hash()].get_g() + new_node.get_g())
                    
                    if (new_node.state_hash() in closed_list_b) and (new_node.get_g() < closed_list_b[new_node.state_hash()].get_g()):
                        closed_list_b[new_node.state_hash()].set_g(new_node.get_g())
                        closed_list_b[new_node.state_hash()].set_cost(new_node.get_cost())
                        heapq.heapify(open_list_b)
                    
                    if new_node.state_hash() not in closed_list_b:
                        new_node.set_cost(new_node_P)
                        heapq.heappush(open_list_b, new_node)   
                        closed_list_b[new_node.state_hash()] = new_node

    return -1, counter

def h_val(current_state, goal_state):
    """
    returns the h_value of a state
    """
    del_x = abs(goal_state.get_x() - current_state.get_x())
    del_y = abs(goal_state.get_y() - current_state.get_y())

    h_value = 1.5*(min(del_x, del_y)) + abs(del_x - del_y)

    return h_value

if __name__ == "__main__":
    main()