# if __name__ == '__main__':
# 	if __package__ is None:
import sys
from os import path
print(path.dirname( path.dirname( path.abspath(__file__) ) ))
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
from prents import my_add
# 	else:
# 		from ..prents import my_add
# 
# from prents import my_add

a = my_add(1,2)
print(a)