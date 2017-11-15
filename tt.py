try :
    raise ValueError("invalid")
except ValueError as ve:
    print ve.message
except:
    print "Error"
else:
    print "Else"
finally:
    print "finally"
print "python"