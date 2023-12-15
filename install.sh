#!/bin/bash

# Author: Pratik Shiveshwar, Sukruthi Chidananda
# Date: December 14, 2023
# Course: ROB 498: RObot Localization

echo "Checking for Matplotlib, Pybullet, and NumPy..."

# Check for Matplotlib
if python3 -c "import matplotlib" &> /dev/null; then
    echo "Matplotlib is installed."
else
    echo "Matplotlib is not installed."
fi

# Check for Pybullet
if python3 -c "import pybullet" &> /dev/null; then
    echo "Pybullet is installed."
else
    echo "Pybullet is not installed."
fi

# Check for NumPy
if python3 -c "import numpy" &> /dev/null; then
    echo "NumPy is installed."
else
    echo "NumPy is not installed."
fi
