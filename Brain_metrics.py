def metrics(cnf_matrix):
    TN = cnf_matrix[0,0]
    FN = cnf_matrix[1,0]
    TP = cnf_matrix[1,1]
    FP = cnf_matrix[0,1]
    
    DSC = (2*TP)/((2*TP)+FP+FN)
    JC = DSC/(2-DSC)
 
        
    return (DSC,JC)

