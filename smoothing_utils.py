import numpy as np

def smooth_chaikin(points, iterations=1, closed=True):
    """
    Suavizado de Chaikin para polígonos.
    Corta las esquinas iterativamente para redondear la forma.
    points: array (N, 2)
    """
    if len(points) < 3:
        return points
    
    pts = points.copy()
    if closed and not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
        
    for _ in range(iterations):
        new_pts = []
        # Para cada segmento P_i -> P_i+1
        # Generar Q_i = 0.75*P_i + 0.25*P_i+1
        # Generar R_i = 0.25*P_i + 0.75*P_i+1
        # Q_i es el nuevo punto "cerca" de P_i
        # R_i es el nuevo punto "cerca" de P_i+1
        
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i+1]
            
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            
            new_pts.append(q)
            new_pts.append(r)
            
        pts = np.array(new_pts)
        
        if closed:
             # Unir el último con el primero para cerrar el loop
             # El algoritmo genera 2N puntos, el último R conecta con el primer Q efectivamente
             # Pero en implementación lineal, el loop se cierra solo si tratamos el último segmento
             pass
             
    return pts
