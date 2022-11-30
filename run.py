
import random

reputation = 0
r = 0 
s = 0


def calculate_reputation(r,s):
    return (r+1) / (r+s+2)
         
def update_reputation(is_fulfilled):
    global r,s
    w = random.uniform(.45, .55)
    λ = 0.98
    threshold = (r - r*λ + s*λ)/(s+1)
    if is_fulfilled:    
        if w > threshold: 
            r *= λ 
            s *= λ 
            r += w
        # else: 
            # print(w, threshold)
    else: 
        r *= λ 
        s *= λ    
        s += w   
    

for i in range(0,1000): 
    if random.uniform(0, 1) < 0.9:
        update_reputation(True)
    else: 
        update_reputation(False)



print(calculate_reputation(r,s))