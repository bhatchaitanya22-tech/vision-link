# Vision-Link

Autonomous part perception system for CNC machine tending.

## Problem
Manual loading and unloading of CNC machines is repetitive,
slow, and error-prone. One worker per machine, all day.

## Solution
A deep learning perception system that detects cylindrical
parts on a table, estimates their position (X, Y) and
orientation (theta), and outputs robot-ready coordinates
for autonomous grasping.

## Industry Partner
Unipro Fine CNC, Bangalore

## Stack
- YOLOv8 — part detection
- Depth Anything v2 — depth estimation  
- ROS2 — robot middleware
- MoveIt2 — arm control
- TensorRT — edge deployment

## Status
Phase 1 — dataset collection and model training
