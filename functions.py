# Functions

def standardize_field(x, m, s):
    return (x-m)/s

def destandardize_field(x, m, s):
    return x*s + m