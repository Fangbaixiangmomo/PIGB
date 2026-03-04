#!/usr/bin/env python3
import numpy as np

def test_environment():
    x = np.linspace(0,1,10)
    print("Environment working:", x.mean())

if __name__ == "__main__":
    test_environment()
