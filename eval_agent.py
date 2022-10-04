def  correct_agent(pred,act):
    correct = 0

    total = len(pred)
    temp_pred = pred[0]
    temp_act = act[0]

    for i,p in enumerate(pred[1:]):

        if temp_pred > p and temp_act > act[i+1]:
            correct += 1
        elif temp_pred < p and temp_act < act[i+1]:
            correct += 1
        temp_act = act[i+1]
        temp_pred = p

    return correct/total

def  profit_agent(pred,act,qty = 100):
    

    earning = 0
    
    temp_pred = pred[0]

    for i,p in enumerate(pred[1:]):

        if act[i-1] > p:
            earning += act[i]*qty
        elif act[i-1] < p:
            earning -= act[i]*qty
        
        

    return earning
    

        
            