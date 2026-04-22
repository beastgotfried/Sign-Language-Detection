import numpy as np 

def distance(p1,p2):
    """
    Args:
        p1(list,tuple):[x,y,z]
        p2(list,tuple):[x,y,z]
        
    Returns:
        float
    """
    return np.linalg.norm(np.array(p1)-np.array(p2))

def angle(p1,p2,p3):
    """
    Args:
        p1(list,tuple):[x,y,z]
        p2(list,tuple):[x,y,z]
        p3(list,tuple):[x,y,z]
    
    Returns:
        float
    """
    a=np.array(p1)
    b=np.array(p2)
    c=np.array(p3)


    vector_ba=b-a
    vector_cb=b-c

    dot_product=np.dot(vector_ba,vector_cb)
    dist_ba=np.linalg.norm(vector_ba)
    dist_cb=np.linalg.norm(vector_cb)

    if dist_ba==0 or dist_cb==0:
        return 0.0

    cosine_angle=dot_product/(dist_ba*dist_cb)
    
    cosine_angle=np.clip(cosine_angle, -1.0, 1.0)
    
    angle_degrees=np.degrees(np.arccos(cosine_angle))
    
    return angle_degrees



    