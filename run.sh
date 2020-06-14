#!/bin/bash

#All challenge data should be in DATA_PATH
PROJECT_PATH="/data/recsys2020/"
DATA_PATH="${PROJECT_PATH}Data/"
XGB_PATH="${PROJECT_PATH}Models/XGB/"

mkdir "${PROJECT_PATH}Models"
mkdir "${PROJECT_PATH}Models/XGB"

mvn clean compile

#need at least 30GB of RAM to run
export MAVEN_OPTS="-Xmx250g -Xms250g"

#parse all data
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20DataParser" -Dexec.args="${DATA_PATH}"

#extract tweet text
python get_tweet_text.py "${DATA_PATH}"

#parse tweet text
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Text" -Dexec.args="${DATA_PATH}"

#extract libsvm feature file for each engagement
export MAVEN_OPTS="-Xmx230g -Xms230g"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Model" -Dexec.args="trainLIBSVM:ALL ${PROJECT_PATH}"

#train XGB models
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Model" -Dexec.args="trainXGB:ALL:true ${PROJECT_PATH}"

#generate LB predictions, output will be in XGB_PATH
export MAVEN_OPTS="-Xmx220g -Xms220g"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Model" -Dexec.args="submit:ALL ${PROJECT_PATH}"

#generate TEST predictions, output will be in XGB_PATH
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Model" -Dexec.args="test:ALL ${PROJECT_PATH}"
