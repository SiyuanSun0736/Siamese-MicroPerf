#!/bin/bash

VARIANT="${VARIANT:-O3-bolt}"

sudo rm -rf ./test/$VARIANT/*
sudo rm -rf ./data/$VARIANT/*
sudo rm -rf ./bin/$VARIANT/*
sudo rm -rf ./bolt_profiles/$VARIANT/*
sudo rm -rf ../log/*
