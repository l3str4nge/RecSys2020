#!/bin/bash

#All challenge data should be in DATA_PATH
PROJECT_PATH="/data/recsys2020/"
DATA_PATH="${PROJECT_PATH}Data/"
XGB_PATH="${PROJECT_PATH}Models/XGB/"
DL_PATH="${PROJECT_PATH}Models/DL/"
DL_H_PATH="${PROJECT_PATH}Models/DL_history/"

mkdir "${PROJECT_PATH}Models"
mkdir "${PROJECT_PATH}Models/XGB"
mkdir "${PROJECT_PATH}Models/DL"
mkdir "${PROJECT_PATH}Models/DL_history"
mvn clean compile

#this was tested on a box with 250GB of RAM
export MAVEN_OPTS="-Xmx230g -Xms230g"

#parse all data
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20DataParser" -Dexec.args="${DATA_PATH}"

#extract tweet text
python3 python/tweets/get_tweet_text.py "${DATA_PATH}"

#parse tweet text
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Text" -Dexec.args="${DATA_PATH}"

#extract libsvm feature file for each engagement
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Model" -Dexec.args="trainLIBSVM:ALL ${PROJECT_PATH}"

#train XGB models
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Model" -Dexec.args="trainXGB:ALL:true ${PROJECT_PATH}"

#prediction uses more memory outside of the JVM
export MAVEN_OPTS="-Xmx220g -Xms220g"

#generate LB predictions, output will be in XGB_PATH
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Model" -Dexec.args="submit:ALL ${PROJECT_PATH}"

#generate TEST predictions, output will be in XGB_PATH
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Model" -Dexec.args="test:ALL ${PROJECT_PATH}"

#extract libsvm feature file for DL pipeline
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20ModelMLP" -Dexec.args="train ${PROJECT_PATH}"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20ModelMLP" -Dexec.args="valid ${PROJECT_PATH}"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20ModelMLP" -Dexec.args="submit ${PROJECT_PATH}"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20ModelMLP" -Dexec.args="test ${PROJECT_PATH}"

#extract libsvm feature file for DL (history) pipeline
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20ModelHistory" -Dexec.args="train ${PROJECT_PATH}"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20ModelHistory" -Dexec.args="valid ${PROJECT_PATH}"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20ModelHistory" -Dexec.args="submit ${PROJECT_PATH}"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20ModelHistory" -Dexec.args="test ${PROJECT_PATH}"
