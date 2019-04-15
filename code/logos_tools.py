
def safe_convert(x, f, default):
    try: 
        return f(x)
    except:
        return default

def handle_divided_by0(x,y):
    return round(x/y,6) if y>0 else 0