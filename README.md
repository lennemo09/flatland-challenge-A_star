# Multi-agent A* Tree Search for Flatland challenge.
A rather inefficient solution to the Flatland challenge using A* Tree Search for path planning.
Rather than using A* for actual 'path finding' for each agent, this implementation naively relies on the value of f(n) for each global state to implements A* tree search to find the shortest "state trajectory"  for a schedule.
Due to this, at every level of node expansion the problem grows by (possibly) 5^n times (as in 5 possible actions for n agents).
